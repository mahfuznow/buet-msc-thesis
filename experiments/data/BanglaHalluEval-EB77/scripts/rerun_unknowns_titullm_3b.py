#!/usr/bin/env python3
"""Re-process only the rows where TituLLM-3B produced an unknown/empty
verdict, using sampling + a shortened prompt to avoid the empty-output
failure mode we saw on long-context CoT.

Reads:  scripts/results_titullm_3b_cot/<task>_cot.csv     (CoT files)
        scripts/results_titullm_3b/gqa_gt_labeled.csv     (broken baseline)
Writes: same file — updates rows in place after making a .bak backup.

Strategy for the empty-output bug:
  1. Truncate the evidence field (context/document/reference) to the
     first ~600 words so the prompt fits comfortably in the model's
     effective attention window.
  2. Use sampling with temperature 0.5, retry up to 4 times if the
     model still emits nothing.
  3. Increase max_new_tokens to 800 (was 512) so long CoT traces
     have room to end in "yes"/"no".

Usage:
    # Rerun CoT unknowns for a specific task+track
    python3 scripts/rerun_unknowns_titullm_3b.py --file scripts/results_titullm_3b_cot/gqa_gt_cot.csv

    # Rerun ALL known-unknown files (in order)
    python3 scripts/rerun_unknowns_titullm_3b.py --all
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise SystemExit("Missing deps. Run: pip install torch transformers accelerate")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "hishab/titulm-llama-3.2-3b-v1.1"

# ── Prompt builders (same as evaluate_cot_titullm_3b.py, just with
#    a hard truncation on the evidence field to avoid empty-output bug) ──

MAX_EVIDENCE_WORDS = 600


def _truncate(text: str, max_words: int = MAX_EVIDENCE_WORDS) -> str:
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " …"


def build_qa_cot(r) -> str:
    return (
        "You are an evaluator checking whether a model answer is hallucinated.\n\n"
        f"Context: {_truncate(r.get('context', ''))}\n"
        f"Question: {r.get('question', '')}\n"
        f"Model Answer: {r.get('correct_answer') or r.get('hallucinated_answer', '')}\n\n"
        "Analyze step by step:\n"
        "Step 1: What factual claims does the answer make?\n"
        "Step 2: Are these claims supported by the context?\n"
        "Step 3: Based on steps 1-2, is the answer hallucinated?\n\n"
        "Final answer (write only this word on the last line): yes or no"
    )


def build_summ_cot(r) -> str:
    source = r.get("document") or r.get("question", "")
    summary = r.get("summary") or r.get("hallucinated_summary", "")
    return (
        "You are an evaluator checking whether a summary is hallucinated relative to a document.\n\n"
        f"Document: {_truncate(source)}\n"
        f"Summary: {summary}\n\n"
        "Analyze step by step:\n"
        "Step 1: List the key claims made in the summary.\n"
        "Step 2: For each claim, check whether it is supported by the document.\n"
        "Step 3: Based on steps 1-2, decide your final answer.\n\n"
        "Final answer (write only this word on the last line): yes or no"
    )


def build_reasoning_cot(r) -> str:
    chain = r.get("answer") or r.get("hallucinated_chain", "")
    return (
        "You are an expert evaluator for Bengali mathematical reasoning tasks.\n\n"
        f"Question: {r.get('question', '')}\n"
        f"Reasoning chain: {_truncate(chain, 800)}\n\n"
        "Analyze step by step:\n"
        "Step 1: Is each calculation or logical step mathematically correct?\n"
        "Step 2: Does the final answer follow logically?\n"
        "Step 3: Based on steps 1-2, is this reasoning chain hallucinated?\n\n"
        "Final answer (write only this word on the last line): yes or no"
    )


def build_baseline_qa(r) -> str:
    return (
        "You are an evaluator.\n"
        "Decide whether the provided model answer is hallucinated relative to the context and question.\n"
        "Reply with exactly one word: yes or no.\n\n"
        f"Context: {_truncate(r.get('context', ''))}\n"
        f"Question: {r.get('question', '')}\n"
        f"Model answer: {r.get('correct_answer') or r.get('hallucinated_answer', '')}\n\n"
        "Answer:"
    )


FILE_HANDLERS = {
    # CoT files
    "gqa_gt_cot.csv":        {"builder": build_qa_cot,        "max_tokens": 800},
    "gqa_hallu_cot.csv":     {"builder": build_qa_cot,        "max_tokens": 800},
    "summ_gt_cot.csv":       {"builder": build_summ_cot,      "max_tokens": 800},
    "summ_hallu_cot.csv":    {"builder": build_summ_cot,      "max_tokens": 800},
    "reason_gt_cot.csv":     {"builder": build_reasoning_cot, "max_tokens": 800},
    "reason_hallu_cot.csv":  {"builder": build_reasoning_cot, "max_tokens": 800},
    # Baseline file (only gqa_gt was corrupted)
    "gqa_gt_labeled.csv":    {"builder": build_baseline_qa,   "max_tokens": 32},
}


YES_RE = re.compile(r"\byes\b", re.I)
NO_RE  = re.compile(r"\bno\b",  re.I)


def parse_verdict(raw: str) -> str:
    """Smarter parser than the original — scans whole response for
    a clean yes/no token, checking last line first."""
    if not raw or not raw.strip():
        return "unknown"
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for line in reversed(lines):
        low = line.lower().strip(" .,!?:;")
        if low in ("yes", "no", "y", "n"):
            return "yes" if low.startswith("y") else "no"
        if low.startswith("yes") and len(low) < 20:
            return "yes"
        if low.startswith("no") and len(low) < 20:
            return "no"
    # Bengali fallback
    if "হ্যাঁ" in raw or "হ্যা" in raw:
        return "yes"
    if "না" in raw and "নয়" not in raw:
        return "no"
    # Whole-string regex fallback (last occurrence wins)
    yes_hits = list(YES_RE.finditer(raw))
    no_hits = list(NO_RE.finditer(raw))
    if yes_hits and (not no_hits or yes_hits[-1].start() > no_hits[-1].start()):
        return "yes"
    if no_hits:
        return "no"
    return "unknown"


class TituLLM:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        print(f"[rerun] loading {model_id} in bfloat16 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.model.eval()
        print(f"[rerun] ready on {self.model.device}\n")

    @torch.no_grad()
    def generate_batch(self, prompts: list[str], max_new_tokens: int, temperature: float = 0.5) -> list[str]:
        prompt_texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            prompt_texts.append(self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            ))
            
        enc = self.tokenizer(prompt_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=2048).to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        input_len = enc.input_ids.shape[1]
        results = []
        for o in out:
            gen = o[input_len:]
            results.append(self.tokenizer.decode(gen, skip_special_tokens=True).strip())
        return results


def process_file(path: Path, llm: TituLLM, limit: Optional[int] = None) -> None:
    handler = FILE_HANDLERS.get(path.name)
    if handler is None:
        print(f"[skip] {path.name}: no handler registered")
        return

    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    if "raw_response" not in fieldnames:
        fieldnames.append("raw_response")
    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")

    unknowns = [i for i, r in enumerate(rows)
                if (r.get("is_hallucinated") or "").strip().lower() not in ("yes", "no")]

    print(f"\n=== {path.name} ===")
    print(f"  total rows: {len(rows)}   currently unknown: {len(unknowns)}")
    if limit:
        unknowns = unknowns[:limit]
        print(f"  processing (limit): {len(unknowns)}")
    if not unknowns:
        print("  nothing to do.")
        return

    # Backup once
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backup: {bak.name}")

    fixed = 0
    still_unk = 0
    t0 = time.time()
    batch_size = 4
    for batch_start in range(0, len(unknowns), batch_size):
        batch_idx = unknowns[batch_start:batch_start + batch_size]
        batch = [rows[idx] for idx in batch_idx]
        prompts = [handler["builder"](r) for r in batch]
        max_new = handler["max_tokens"]
        
        raws = [""] * len(batch)
        labels = ["unknown"] * len(batch)
        active_indices = list(range(len(batch)))
        
        for attempt, temp in enumerate((0.5, 0.7, 0.9, 0.3), 1):
            if not active_indices:
                break
                
            active_prompts = [prompts[i] for i in active_indices]
            active_raws = llm.generate_batch(active_prompts, max_new_tokens=max_new, temperature=temp)
            
            new_active_indices = []
            for i, raw in zip(active_indices, active_raws):
                label = parse_verdict(raw)
                raws[i] = raw
                labels[i] = label
                if label not in ("yes", "no"):
                    new_active_indices.append(i)
            
            active_indices = new_active_indices

        for i, idx in enumerate(batch_idx):
            if labels[i] in ("yes", "no"):
                fixed += 1
            else:
                still_unk += 1
            rows[idx]["raw_response"] = raws[i]
            rows[idx]["is_hallucinated"] = labels[i]

        k = min(batch_start + batch_size, len(unknowns))
        elapsed = time.time() - t0
        rate = k / max(elapsed, 1e-6)
        eta = (len(unknowns) - k) / max(rate, 1e-6)
        print(f"  {k}/{len(unknowns)}   fixed={fixed}   still_unk={still_unk}   "
              f"rate={rate:.2f}/s   eta={eta/60:.1f}min")

    # Rewrite the file
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  rewrote {path.name}: fixed {fixed} / {len(unknowns)} unknowns "
          f"({still_unk} still unknown)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, help="Path to one output CSV to rerun.")
    ap.add_argument("--all", action="store_true",
                    help="Rerun unknowns across every registered file.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="Sanity: rerun at most N unknowns per file.")
    args = ap.parse_args()

    llm = TituLLM(args.model)

    if args.all:
        for name in FILE_HANDLERS:
            for pattern in (
                f"scripts/results_titullm_3b_cot/{name}",
                f"scripts/results_titullm_3b/{name}",
            ):
                path = ROOT / pattern
                if path.exists():
                    process_file(path, llm, limit=args.limit)
    elif args.file:
        process_file(Path(args.file).resolve(), llm, limit=args.limit)
    else:
        ap.error("Pass --file <path> or --all")


if __name__ == "__main__":
    main()
