#!/usr/bin/env python3
"""Chain-of-Thought (CoT) hallucination-detection evaluation with TituLLM-3B.

Mirrors scripts/evaluate_cot_tigerllm.py in structure. Matches the paper's
scope: CoT is applied only to the three tasks where every other judge was
also evaluated with CoT — GQA, Summarization, and Reasoning. Codemixed QA
does NOT get CoT here (consistent with the rest of the benchmark).

Both tracks (ground-truth + hallucinated) are covered for each task.

Model:  hishab/titulm-llama-3.2-3b-v1.1  (change via --model)

Outputs land under: scripts/results_titullm_3b_cot/<task>_cot.csv

Usage:
    python scripts/evaluate_cot_titullm_3b.py --task all
    python scripts/evaluate_cot_titullm_3b.py --task summ_hallu
    python scripts/evaluate_cot_titullm_3b.py --task reason_gt --limit 100
"""

from __future__ import annotations

import argparse
import csv
import re
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
SLUG = "titullm_3b"

# ── CoT prompts (mirror the paper's plain-CoT style used in evaluate_cot_*) ──

QA_COT_PROMPT = (
    "You are an evaluator checking whether a model answer is hallucinated.\n\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Model Answer: {answer}\n\n"
    "Analyze step by step:\n"
    "Step 1: What factual claims does the answer make?\n"
    "Step 2: Are these claims supported by or inferable from the context?\n"
    "Step 3: Based on steps 1-2, is the answer hallucinated?\n\n"
    "Final answer (write only this word on the last line): yes or no\n"
    "(yes = answer is hallucinated, no = answer is not hallucinated)"
)

SUMM_COT_PROMPT = (
    "You are an evaluator checking whether a summary is hallucinated relative to a document.\n\n"
    "Document: {document}\n"
    "Summary: {summary}\n\n"
    "Analyze step by step:\n"
    "Step 1: List the key claims made in the summary.\n"
    "Step 2: For each claim, check whether it is directly supported by the document.\n"
    "Step 3: Based on steps 1-2, decide your final answer.\n\n"
    "Final answer (write only this word on the last line): yes or no\n"
    "(yes = summary is hallucinated, no = summary is not hallucinated)"
)

REASONING_COT_PROMPT = (
    "You are an expert evaluator for Bengali mathematical reasoning tasks.\n\n"
    "Question: {question}\n"
    "Reasoning chain: {chain}\n\n"
    "Analyze step by step:\n"
    "Step 1: Is each calculation or logical step in the reasoning chain mathematically correct?\n"
    "Step 2: Does the final answer follow logically from the reasoning chain?\n"
    "Step 3: Based on steps 1-2, is this reasoning chain hallucinated (incorrect or fabricated)?\n\n"
    "Final answer (write only this word on the last line): yes or no"
)


def build_qa_cot(r):        return QA_COT_PROMPT.format(context=r["context"], question=r["question"], answer=r["_candidate"])
def build_summ_cot(r):      return SUMM_COT_PROMPT.format(document=r["_document"], summary=r["_candidate"])
def build_reasoning_cot(r): return REASONING_COT_PROMPT.format(question=r["question"], chain=r["_candidate"])


TASKS = {
    "gqa_gt":       {"in": "BanglaHalluEval Datasets/banglahallueval_qa_1000.csv",
                     "out": f"scripts/results_{SLUG}_cot/gqa_gt_cot.csv",
                     "candidate": "correct_answer",
                     "prompt": build_qa_cot},
    "gqa_hallu":    {"in": "Hallucination Generated Answers/qa_4000.csv",
                     "out": f"scripts/results_{SLUG}_cot/gqa_hallu_cot.csv",
                     "candidate": "hallucinated_answer",
                     "prompt": build_qa_cot},

    "summ_gt":      {"in": "Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv",
                     "out": f"scripts/results_{SLUG}_cot/summ_gt_cot.csv",
                     "candidate": "summary",
                     "doc":       "question",
                     "prompt": build_summ_cot},
    "summ_hallu":   {"in": "Hallucination Generated Answers/summarization_3000_corrected.csv",
                     "out": f"scripts/results_{SLUG}_cot/summ_hallu_cot.csv",
                     "candidate": "hallucinated_summary",
                     "doc":       "document",
                     "prompt": build_summ_cot},

    "reason_gt":    {"in": "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv",
                     "out": f"scripts/results_{SLUG}_cot/reason_gt_cot.csv",
                     "candidate": "answer",
                     "prompt": build_reasoning_cot},
    "reason_hallu": {"in": "Hallucination Generated Answers/reasoning_1000.csv",
                     "out": f"scripts/results_{SLUG}_cot/reason_hallu_cot.csv",
                     "candidate": "hallucinated_chain",
                     "prompt": build_reasoning_cot},
    # NOTE: No codemix here on purpose. Paper's CoT setup only covered
    # GQA, Summarization, and Reasoning; Codemixed QA stays baseline-only.
}


class TituLLM:
    def __init__(self, model_id: str):
        print(f"[titullm-cot] loading {model_id} in bfloat16 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.model.eval()
        print(f"[titullm-cot] ready on {self.model.device}\n")

    @torch.no_grad()
    def generate_batch(self, prompts: list[str], max_new_tokens: int = 512) -> list[str]:
        prompt_texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            prompt_texts.append(prompt_text)
            
        enc = self.tokenizer(
            prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(self.model.device)
        
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        input_len = enc.input_ids.shape[1]
        results = []
        for o in out:
            gen = o[input_len:]
            results.append(self.tokenizer.decode(gen, skip_special_tokens=True).strip())
        return results


# Extract "yes"/"no" from a multi-line CoT response — prefer the last non-empty line.
YES_RE = re.compile(r"\byes\b", re.I)
NO_RE  = re.compile(r"\bno\b",  re.I)

def parse_cot_verdict(raw: str) -> str:
    if not raw:
        return "unknown"
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for line in reversed(lines):
        low = line.lower()
        if low in ("yes", "no"): return low
        if low.startswith("yes"): return "yes"
        if low.startswith("no"):  return "no"
    # Fallback: last regex match anywhere
    last_yes = list(YES_RE.finditer(raw))
    last_no  = list(NO_RE.finditer(raw))
    if last_yes and (not last_no or last_yes[-1].start() > last_no[-1].start()):
        return "yes"
    if last_no:
        return "no"
    return "unknown"


def run_task(task_key: str, llm: TituLLM, limit: Optional[int] = None) -> None:
    cfg = TASKS[task_key]
    inp = ROOT / cfg["in"]
    out = ROOT / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(inp, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for extra in ("raw_response", "is_hallucinated"):
        if extra not in fieldnames:
            fieldnames.append(extra)

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
        r["_candidate"] = r.get(cfg["candidate"], "") or ""
        if "doc" in cfg:
            r["_document"] = r.get(cfg["doc"], "") or ""
        pending.append((i, sid, r))
        if limit and len(pending) >= limit:
            break

    print(f"[{SLUG}-cot] {task_key}  total={len(rows)}  done={len(done)}  pending={len(pending)}"
          + (f"  (limit={limit})" if limit else ""))
    if not pending:
        return

    write_header = (not out.exists()) or out.stat().st_size == 0
    t0 = time.time()
    batch_size = 16  # using a decent batch size to speed things up
    
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
            
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            prompts = [cfg["prompt"](r) for i, sid, r in batch]
            
            raws = llm.generate_batch(prompts, max_new_tokens=512)
            
            for (i, sid, r), raw in zip(batch, raws):
                label = parse_cot_verdict(raw)
                out_row = {kk: r.get(kk, "") for kk in fieldnames if kk not in ("raw_response", "is_hallucinated")}
                out_row["raw_response"] = raw
                out_row["is_hallucinated"] = label
                w.writerow(out_row)
            
            f.flush()
            k = min(batch_start + batch_size, len(pending))
            elapsed = time.time() - t0
            rate = k / max(elapsed, 1e-6)
            eta = (len(pending) - k) / max(rate, 1e-6)
            print(f"  [{SLUG}-cot] {task_key}  {k}/{len(pending)}  rate={rate:.2f}/s  eta={eta/60:.1f}min  last={label}")
            
    print(f"  [{SLUG}-cot] {task_key} -> {out.relative_to(ROOT)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", *TASKS.keys()])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tasks = list(TASKS.keys()) if args.task == "all" else [args.task]
    llm = TituLLM(args.model)
    for t in tasks:
        run_task(t, llm, limit=args.limit)


if __name__ == "__main__":
    main()
