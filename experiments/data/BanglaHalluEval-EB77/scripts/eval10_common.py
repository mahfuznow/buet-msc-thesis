"""Shared utilities for the 10% rebuttal evaluation of BanglaLLaMA-13B and
TituLLM-3B on GQA / Summarization / Reasoning / Codemix (baseline + CoT).

Design goals:
  * Same sampled subset for both models (deterministic — seed=42).
  * Zero "unknown" labels — first-token constrained decoding on the two
    yes/no vocab IDs guarantees a valid verdict for baseline; for CoT
    we do free-form generation, then a forced constrained-yes/no
    completion if parsing fails.
  * No evidence truncation. Only tokenizer safety cap at 3072 tokens.
  * Incremental CSV writes so a crash never loses completed rows.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
SAMPLE_DIR = ROOT / "Sampling" / "10pct"


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

SOURCES: Dict[str, Tuple[str, int]] = {
    # sample_key -> (source_path, sample_size)
    "qa":            ("Hallucination Generated Answers/qa_4000.csv", 400),
    "summ_3000":     ("Hallucination Generated Answers/summarization_3000_corrected.csv", 300),
    "summ_1000":     ("Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv", 100),
    "reason_hallu":  ("Hallucination Generated Answers/reasoning_1000.csv", 100),
    "reason_gt":     ("Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv", 100),
    "codemix":       ("Hallucination Generated Answers/codemix_4000_gpt4_1_mini_labeled.csv", 400),
}


def build_samples(force: bool = False) -> None:
    """Materialize deterministic 10% subsets into Sampling/10pct/*.csv."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for key, (rel_path, n) in SOURCES.items():
        out = SAMPLE_DIR / f"{key}.csv"
        if out.exists() and not force:
            print(f"[sample] {key}: already exists ({out}), skip")
            continue
        src = ROOT / rel_path
        df = pd.read_csv(src, encoding="utf-8", on_bad_lines="skip")
        n = min(n, len(df))
        sub = df.sample(n=n, random_state=SEED).reset_index(drop=True)
        sub.to_csv(out, index=False, encoding="utf-8")
        print(f"[sample] {key}: wrote {len(sub)} rows -> {out.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Track config — each track = (sample_file, builder_fn, output_name, expected_label)
# ─────────────────────────────────────────────────────────────────────────────

def _s(x) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


BASELINE_TASKS = [
    ("gqa_gt",       "qa",           "right_answer",         "gqa"),
    ("gqa_hallu",    "qa",           "hallucinated_answer",  "gqa"),
    ("summ_gt",      "summ_1000",    "summary",              "summ_gt"),
    ("summ_hallu",   "summ_3000",    "hallucinated_summary", "summ_hallu"),
    ("reason_gt",    "reason_gt",    "answer",               "reason"),
    ("reason_hallu", "reason_hallu", "hallucinated_chain",   "reason"),
    ("codemix_gt",   "codemix",      "codemix_answer",       "codemix"),
    ("codemix_hallu","codemix",      "hallucinated_answer",  "codemix"),
]

# CoT: same list minus codemix (per user).
COT_TASKS = [t for t in BASELINE_TASKS if not t[0].startswith("codemix")]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders  (task_kind determines context/question field mapping)
# ─────────────────────────────────────────────────────────────────────────────

def build_baseline_prompt(row: dict, answer_field: str, task_kind: str) -> str:
    ans = _s(row.get(answer_field))
    if task_kind == "gqa":
        return (
            "You are an evaluator. Decide whether the model answer is hallucinated "
            "relative to the context and question. Reply with exactly one word: yes or no.\n\n"
            f"Context: {_s(row.get('context'))}\n"
            f"Question: {_s(row.get('question'))}\n"
            f"Model answer: {ans}\n\nAnswer:"
        )
    if task_kind == "summ_gt":
        # summ_1000 file has only 'question' (source doc) and 'summary'
        return (
            "You are an evaluator. Decide whether the summary is hallucinated relative "
            "to the document. Reply with exactly one word: yes or no.\n\n"
            f"Document: {_s(row.get('question'))}\n"
            f"Summary: {ans}\n\nAnswer:"
        )
    if task_kind == "summ_hallu":
        return (
            "You are an evaluator. Decide whether the summary is hallucinated relative "
            "to the document. Reply with exactly one word: yes or no.\n\n"
            f"Document: {_s(row.get('document'))}\n"
            f"Summary: {ans}\n\nAnswer:"
        )
    if task_kind == "reason":
        return (
            "You are an evaluator for Bengali mathematical reasoning. Decide whether the "
            "reasoning chain is hallucinated (incorrect or fabricated). Reply with exactly "
            "one word: yes or no.\n\n"
            f"Question: {_s(row.get('question'))}\n"
            f"Reasoning chain: {ans}\n\nAnswer:"
        )
    if task_kind == "codemix":
        return (
            "You are an evaluator. Decide whether the model answer is hallucinated "
            "relative to the code-mixed context and question. Reply with exactly one "
            "word: yes or no.\n\n"
            f"Context: {_s(row.get('codemix_context'))}\n"
            f"Question: {_s(row.get('codemix_question'))}\n"
            f"Model answer: {ans}\n\nAnswer:"
        )
    raise ValueError(f"unknown task_kind {task_kind}")


def build_cot_prompt(row: dict, answer_field: str, task_kind: str) -> str:
    base = build_baseline_prompt(row, answer_field, task_kind)
    # Replace the terse ending with a CoT instruction
    prefix = base.rsplit("\n\nAnswer:", 1)[0]
    return (
        prefix
        + "\n\nAnalyze step by step:\n"
        "Step 1: Identify the key claims made.\n"
        "Step 2: Check each claim against the source.\n"
        "Step 3: Decide the final verdict.\n\n"
        "Write your reasoning, then on the LAST line write exactly one word: yes or no"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

_YES_WORDS = ("yes", "y", "হ্যাঁ", "হ্যা", "হা", "Yes", "YES")
_NO_WORDS  = ("no",  "n", "না", "নয়", "No", "NO")
_YES_RE = re.compile(r"\byes\b", re.I)
_NO_RE  = re.compile(r"\bno\b",  re.I)


def parse_yesno(raw: str) -> str:
    """Return 'yes', 'no', or 'unknown'."""
    if not raw:
        return "unknown"
    text = raw.strip()

    # Last-line first (CoT convention)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        low = line.lower().strip(" .,!?:;\"'`*[](){}<>")
        if low in ("yes", "y"):
            return "yes"
        if low in ("no", "n"):
            return "no"

    # JSON style
    try:
        obj = json.loads(text)
        val = str(obj.get("is_hallucinated") or obj.get("answer") or "").strip().lower()
        if val.startswith("y"):
            return "yes"
        if val.startswith("n"):
            return "no"
    except Exception:
        pass
    m = re.search(r'"is_hallucinated"\s*:\s*"(yes|no)"', text, re.I)
    if m:
        return m.group(1).lower()

    # Bengali fallback
    if any(w in text for w in ("হ্যাঁ", "হ্যা")):
        return "yes"
    if "না" in text and "নয়" not in text:
        return "no"

    # Last yes/no occurrence anywhere (last wins)
    yes_hits = list(_YES_RE.finditer(text))
    no_hits  = list(_NO_RE.finditer(text))
    if yes_hits and (not no_hits or yes_hits[-1].start() > no_hits[-1].start()):
        return "yes"
    if no_hits:
        return "no"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Constrained yes/no decoding
# ─────────────────────────────────────────────────────────────────────────────

def yes_no_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """Collect token IDs that produce 'yes'/'no' as the first content word.
    Uses several surface forms to survive tokenizer quirks (leading space,
    capitalization). Returns (yes_ids, no_ids)."""
    yes, no = set(), set()
    for w in ("yes", " yes", "Yes", " Yes", "YES", " YES"):
        for tid in tokenizer.encode(w, add_special_tokens=False):
            yes.add(tid)
    for w in ("no", " no", "No", " No", "NO", " NO"):
        for tid in tokenizer.encode(w, add_special_tokens=False):
            no.add(tid)
    # keep only single-token surface forms to avoid overshooting
    def _single(word: str) -> Optional[int]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    yes_ids = sorted({tid for w in ("yes", " yes", "Yes", " Yes") if (tid := _single(w)) is not None})
    no_ids  = sorted({tid for w in ("no",  " no",  "No",  " No" ) if (tid := _single(w)) is not None})
    return yes_ids, no_ids


def constrained_yesno(model, tokenizer, prompt_text: str,
                      yes_ids: List[int], no_ids: List[int]) -> str:
    """Greedy 1-token decode restricted to yes/no vocab. Returns 'yes' or 'no'."""
    import torch
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    with torch.inference_mode():
        out = model(**enc)
    logits = out.logits[0, -1, :]
    yes_score = float(logits[yes_ids].max()) if yes_ids else float("-inf")
    no_score  = float(logits[no_ids ].max()) if no_ids  else float("-inf")
    return "yes" if yes_score >= no_score else "no"


# ─────────────────────────────────────────────────────────────────────────────
# Chat template helper — some tokenizers barf; fall back gracefully
# ─────────────────────────────────────────────────────────────────────────────

def apply_chat(tokenizer, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return prompt


# ─────────────────────────────────────────────────────────────────────────────
# CSV I/O — incremental append per row, keeps original columns + new ones
# ─────────────────────────────────────────────────────────────────────────────

def open_writer(out_path: Path, base_fieldnames: List[str]) -> Tuple[csv.DictWriter, list, set]:
    """Return (writer, existing_rows, done_ids). Resumes from existing file."""
    fieldnames = list(base_fieldnames)
    for extra in ("raw_response", "is_hallucinated"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    done_ids = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
        done_ids = {(r.get("id") or r.get("question_id") or r.get("source_id") or "").strip()
                    for r in existing if (r.get("is_hallucinated") or "").strip() in ("yes", "no")}
        f_out = open(out_path, "a", encoding="utf-8", newline="")
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    else:
        f_out = open(out_path, "w", encoding="utf-8", newline="")
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
    writer._f = f_out  # attach so caller can close
    return writer, existing, done_ids


def row_id(row: dict) -> str:
    for k in ("id", "question_id", "source_id"):
        v = row.get(k)
        if v:
            return str(v).strip()
    return ""
