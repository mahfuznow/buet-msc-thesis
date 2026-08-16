#!/usr/bin/env python3
"""Baseline hallucination-detection evaluation with TituLLM-3B (Hishab).

Mirrors scripts/evaluate_tigerllm.py in structure but adds a fourth task
(Codemixed QA) and covers both tracks (ground-truth and hallucinated) for
every task. Streams rows to CSV with `--resume` behavior — re-invoking the
same command skips rows already labeled.

Model:
  hishab/titulm-llama-3.2-3b-v1.1        (change via --model)

Tasks x tracks (16,000 rows total per full run):
  gqa_gt        (1,000)   gqa_hallu       (4,000)
  summ_gt       (1,000)   summ_hallu      (3,000)
  reason_gt     (1,000)   reason_hallu    (1,000)
  codemix_gt    (1,000)   codemix_hallu   (4,000)

Usage:
    python scripts/evaluate_titullm_3b.py --task all
    python scripts/evaluate_titullm_3b.py --task gqa_hallu
    python scripts/evaluate_titullm_3b.py --task codemix_gt --limit 100  # sanity
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

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
SLUG = "titullm_3b"

# ── Prompts (mirror the closed-book "reply yes or no" style from the paper) ──

QA_PROMPT = (
    "You are an evaluator.\n"
    "Decide whether the provided model answer is hallucinated relative to the context and question.\n"
    "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
    "Interpretation: 'yes' means the answer contains information not supported by the context "
    "or contradicts it (hallucinated).\n"
    "Provide the answer in English only: yes or no.\n\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Model answer: {answer}\n\n"
    "Answer now:"
)

SUMM_PROMPT = (
    "You are an evaluator.\n"
    "Decide whether the provided model summary is hallucinated relative to the document.\n"
    "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
    "Interpretation: 'yes' means the summary contains information not supported by the document "
    "or contradicts it (hallucinated).\n"
    "Provide the answer in English only: yes or no.\n\n"
    "Document: {document}\n"
    "Summary: {summary}\n\n"
    "Answer now:"
)

REASONING_PROMPT = (
    "You are an expert evaluator for Bengali mathematical reasoning tasks.\n"
    "Decide whether the provided reasoning chain is hallucinated (incorrect or fabricated).\n"
    "Only reply with a single token: yes or no. No explanation, no punctuation, no extra text.\n"
    "Interpretation: 'yes' means the reasoning chain contains incorrect calculations, wrong "
    "intermediate values, or a wrong final answer.\n\n"
    "Question: {question}\n"
    "Reasoning chain: {chain}\n\n"
    "Answer now:"
)

CODEMIX_PROMPT = (
    "You are an evaluator for code-mixed Bengali-English question answering.\n"
    "Decide whether the provided model answer is hallucinated relative to the code-mixed context "
    "and question. Only reply with a single token: yes or no. No explanation, no punctuation.\n"
    "Interpretation: 'yes' means the answer contains information not supported by the context "
    "or contradicts it (hallucinated).\n"
    "Provide the answer in English only: yes or no.\n\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Model answer: {answer}\n\n"
    "Answer now:"
)


# ── Task registry: (input_csv, output_csv, candidate_field, prompt_builder) ─
def build_qa_prompt(r):        return QA_PROMPT.format(context=r["context"], question=r["question"], answer=r["_candidate"])
def build_summ_prompt(r):      return SUMM_PROMPT.format(document=r["_document"], summary=r["_candidate"])
def build_reasoning_prompt(r): return REASONING_PROMPT.format(question=r["question"], chain=r["_candidate"])
def build_codemix_prompt(r):   return CODEMIX_PROMPT.format(context=r["codemix_context"], question=r["codemix_question"], answer=r["_candidate"])


TASKS = {
    "gqa_gt":       {"in": "BanglaHalluEval Datasets/banglahallueval_qa_1000.csv",
                     "out": f"scripts/results_{SLUG}/gqa_gt_labeled.csv",
                     "candidate": "correct_answer",
                     "prompt": build_qa_prompt},
    "gqa_hallu":    {"in": "Hallucination Generated Answers/qa_4000.csv",
                     "out": f"scripts/results_{SLUG}/gqa_hallu_labeled.csv",
                     "candidate": "hallucinated_answer",
                     "prompt": build_qa_prompt},

    "summ_gt":      {"in": "Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv",
                     "out": f"scripts/results_{SLUG}/summ_gt_labeled.csv",
                     "candidate": "summary",
                     "doc":       "question",         # source query IS the document for summ_gt
                     "prompt": build_summ_prompt},
    "summ_hallu":   {"in": "Hallucination Generated Answers/summarization_3000_corrected.csv",
                     "out": f"scripts/results_{SLUG}/summ_hallu_labeled.csv",
                     "candidate": "hallucinated_summary",
                     "doc":       "document",
                     "prompt": build_summ_prompt},

    "reason_gt":    {"in": "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv",
                     "out": f"scripts/results_{SLUG}/reason_gt_labeled.csv",
                     "candidate": "answer",
                     "prompt": build_reasoning_prompt},
    "reason_hallu": {"in": "Hallucination Generated Answers/reasoning_1000.csv",
                     "out": f"scripts/results_{SLUG}/reason_hallu_labeled.csv",
                     "candidate": "hallucinated_chain",
                     "prompt": build_reasoning_prompt},

    "codemix_gt":   {"in": "Codemix/Main dataset/codemix_1000.csv",
                     "out": f"scripts/results_{SLUG}/codemix_gt_labeled.csv",
                     "candidate": "codemix_answer",
                     "prompt": build_codemix_prompt},
    "codemix_hallu":{"in": "Hallucination Generated Answers/codemix_4000.csv",
                     "out": f"scripts/results_{SLUG}/codemix_hallu_labeled.csv",
                     "candidate": "hallucinated_answer",
                     "prompt": build_codemix_prompt},
}


# ── Model wrapper ─────────────────────────────────────────────────────────────

class TituLLM:
    def __init__(self, model_id: str):
        print(f"[titullm] loading {model_id} in bfloat16 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print(f"[titullm] ready on {self.model.device}\n")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 16) -> str:
        # Two-step to stay compatible with both old and new `transformers`:
        # apply_chat_template can return BatchEncoding (dict-like) in >=4.44,
        # so we ask for text and tokenize explicitly.
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen = out[0][enc.input_ids.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


# ── Answer parsing (yes/no normaliser) ────────────────────────────────────────
YES_RE = re.compile(r"\byes\b", re.I)
NO_RE  = re.compile(r"\bno\b",  re.I)

def parse_yes_no(raw: str) -> str:
    if not raw:
        return "unknown"
    s = raw.strip()
    # Take first non-empty line
    first = next((ln for ln in s.splitlines() if ln.strip()), "").strip().lower()
    if first.startswith("y") or "হ্যাঁ" in first: return "yes"
    if first.startswith("n") or "না" in first:   return "no"
    if YES_RE.search(s): return "yes"
    if NO_RE.search(s):  return "no"
    return "unknown"


# ── Streaming CSV runner with resume ──────────────────────────────────────────
def run_task(task_key: str, llm: TituLLM, limit: Optional[int] = None) -> None:
    cfg = TASKS[task_key]
    inp = ROOT / cfg["in"]
    out = ROOT / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(inp, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")

    # Load already-done ids for resume
    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                sid = r.get("id") or r.get("source_id") or r.get("question_id")
                lbl = r.get("is_hallucinated", "").lower()
                if sid and lbl in ("yes", "no"):
                    done.add(sid)

    pending = []
    for i, r in enumerate(rows):
        sid = r.get("id") or r.get("source_id") or r.get("question_id") or str(i)
        if sid in done:
            continue
        # Attach candidate + document for prompt builders
        r["_candidate"] = r.get(cfg["candidate"], "") or ""
        if "doc" in cfg:
            r["_document"] = r.get(cfg["doc"], "") or ""
        pending.append((i, sid, r))
        if limit and len(pending) >= limit:
            break

    print(f"[{SLUG}] {task_key}  total={len(rows)}  done={len(done)}  pending={len(pending)}"
          + (f"  (limit={limit})" if limit else ""))
    if not pending:
        return

    write_header = (not out.exists()) or out.stat().st_size == 0
    t0 = time.time()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for k, (i, sid, r) in enumerate(pending, 1):
            raw = llm.generate(cfg["prompt"](r), max_new_tokens=16)
            label = parse_yes_no(raw)
            out_row = {k2: r[k2] for k2 in fieldnames if k2 != "is_hallucinated" and k2 in r}
            out_row["is_hallucinated"] = label
            w.writerow(out_row)
            f.flush()
            if k % 20 == 0 or k == len(pending):
                elapsed = time.time() - t0
                rate = k / max(elapsed, 1e-6)
                eta = (len(pending) - k) / max(rate, 1e-6)
                print(f"  [{SLUG}] {task_key}  {k}/{len(pending)}  rate={rate:.2f}/s  eta={eta/60:.1f}min  last={label}")
    print(f"  [{SLUG}] {task_key} -> {out.relative_to(ROOT)}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", *TASKS.keys()])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="Sanity: only N pending rows.")
    args = ap.parse_args()

    tasks = list(TASKS.keys()) if args.task == "all" else [args.task]
    llm = TituLLM(args.model)
    for t in tasks:
        run_task(t, llm, limit=args.limit)


if __name__ == "__main__":
    main()
