#!/usr/bin/env python3
"""CoT hallucination evaluation for BanglaLLM/bangla-llama-13b-instruct-v0.1.

This mirrors evaluate_cot_tigerllm.py, but uses BanglaLLama and defaults to a
4-bit load so it can run on a cost-efficient 24 GB RunPod GPU.

Tasks (default: all):
  qa_hallu     → QA/Results/qa_cot_hallu_banglallama.csv
  qa_gt        → QA/Results/qa_cot_gt_banglallama.csv
  summ_hallu   → Summarization/Results/summ_3000_cot_banglallama.csv
  summ_gt      → Summarization/Results/summ_1000_cot_banglallama.csv
  reason_hallu → Reasoning/Results/reasoning_cot_banglallama.csv
  reason_gt    → Reasoning/Results/reasoning_gt_cot_banglallama.csv

Usage:
    python scripts/evaluate_cot_banglallama.py --task all
    python scripts/evaluate_cot_banglallama.py --task reason_hallu
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    raise SystemExit("Run: pip install transformers torch accelerate bitsandbytes")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_cot_ollama import (  # noqa: E402
    QA_COT_PROMPT,
    SUMM_COT_PROMPT,
    REASONING_COT_PROMPT,
    parse_summ_label,
    parse_reasoning_label,
)

MODEL_ID = "BanglaLLM/bangla-llama-13b-instruct-v0.1"
SLUG = "banglallama_13b"


def _format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        return prompt


class BanglaLLama:
    def __init__(self, model_id: str = MODEL_ID, quantize: bool = True):
        print(f"Loading {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if quantize:
            print("Using 4-bit quantization for a 24 GB GPU")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        self.model.eval()
        print(f"Model loaded on {next(self.model.parameters()).device}\n")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        prompt_text = _format_prompt(self.tokenizer, prompt)
        enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen = out[0][enc.input_ids.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def run_csv_task(input_file, output_file, build_prompt_fn, parse_fn, llm, max_new_tokens=512):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")

    completed = set()
    if Path(output_file).exists():
        with open(output_file, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                sid = r.get("id") or r.get("source_id")
                lbl = r.get("is_hallucinated", "")
                if sid and lbl in ("yes", "no", "Yes", "No"):
                    completed.add(sid)
        print(f"  Resuming — {len(completed)} rows already labeled.")

    pending = [
        (i, r) for i, r in enumerate(rows)
        if (r.get("id") or r.get("source_id") or str(i)) not in completed
    ]
    print(f"  Pending: {len(pending)}")

    write_header = not Path(output_file).exists() or Path(output_file).stat().st_size == 0
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for i, r in pending:
            sid = r.get("id") or r.get("source_id") or str(i)
            raw = llm.generate(build_prompt_fn(r), max_new_tokens)
            label = parse_fn(raw)
            r_out = dict(r)
            r_out["is_hallucinated"] = label
            writer.writerow(r_out)
            f.flush()
            print(f"  {i}: {sid} -> {label}")

    print(f"  Saved → {output_file}\n")


def run_reasoning_task(input_file, output_file, is_groundtruth, llm, max_new_tokens=512):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_file)

    if is_groundtruth:
        df = df.rename(columns={"question_id": "id"})
        df["chain"] = df["answer"]
        df["ans_col"] = ""
    else:
        df["chain"] = df["hallucinated_chain"]
        df["ans_col"] = df["hallucinated_answer"]

    if os.path.exists(output_file):
        print(f"  Resuming from {output_file}...")
        df = pd.read_csv(output_file)
        df["is_hallucinated"] = df["is_hallucinated"].fillna("")
    else:
        df["is_hallucinated"] = ""

    pending = df[~df["is_hallucinated"].isin(["Yes", "No"])].index.tolist()
    print(f"  Total: {len(df)} | Pending: {len(pending)}")

    chain_col = "chain" if "chain" in df.columns else "hallucinated_chain"
    ans_col = "ans_col" if "ans_col" in df.columns else "hallucinated_answer"

    for idx in pending:
        prompt = REASONING_COT_PROMPT.format(
            question=df.at[idx, "question"],
            chain=df.at[idx, chain_col],
            answer=df.at[idx, ans_col],
        )
        raw = llm.generate(prompt, max_new_tokens)
        label = parse_reasoning_label(raw)
        df.at[idx, "is_hallucinated"] = label
        print(f"  {idx}: {df.at[idx, 'id']} -> {label}")
        df.to_csv(output_file, index=False)

    print(f"  Saved → {output_file}\n")


def get_tasks(llm) -> dict:
    return {
        "qa_hallu": lambda: run_csv_task(
            "Hallucination Generated Answers/qa_4000.csv",
            f"QA/Results/qa_cot_hallu_{SLUG}.csv",
            lambda r: QA_COT_PROMPT.format(
                question=r.get("question", ""),
                answer=r.get("hallucinated_answer", ""),
            ),
            parse_summ_label, llm,
        ),
        "qa_gt": lambda: run_csv_task(
            "BanglaHalluEval Datasets/banglahallueval_qa_1000.csv",
            f"QA/Results/qa_cot_gt_{SLUG}.csv",
            lambda r: QA_COT_PROMPT.format(
                question=r.get("question", ""),
                answer=r.get("correct_answer", ""),
            ),
            parse_summ_label, llm,
        ),
        "summ_hallu": lambda: run_csv_task(
            "Hallucination Generated Answers/summarization_3000_corrected.csv",
            f"Summarization/Results/summ_3000_cot_{SLUG}.csv",
            lambda r: SUMM_COT_PROMPT.format(
                document=r.get("document", "") or r.get("question", ""),
                summary=r.get("hallucinated_summary", "") or r.get("summary", ""),
            ),
            parse_summ_label, llm,
        ),
        "summ_gt": lambda: run_csv_task(
            "Summarization/1000 Selected Samples/banglahallueval_summarization_dataset_1000.csv",
            f"Summarization/Results/summ_1000_cot_{SLUG}.csv",
            lambda r: SUMM_COT_PROMPT.format(
                document=r.get("question", ""),
                summary=r.get("summary", ""),
            ),
            parse_summ_label, llm,
        ),
        "reason_hallu": lambda: run_reasoning_task(
            "Hallucination Generated Answers/reasoning_1000.csv",
            f"Reasoning/Results/reasoning_cot_{SLUG}.csv",
            is_groundtruth=False, llm=llm,
        ),
        "reason_gt": lambda: run_reasoning_task(
            "Reasoning/1000 Selected Samples/somadhan_1000_main_ordered.csv",
            f"Reasoning/Results/reasoning_gt_cot_{SLUG}.csv",
            is_groundtruth=True, llm=llm,
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        choices=["all", "qa_hallu", "qa_gt", "summ_hallu", "summ_gt", "reason_hallu", "reason_gt"],
        default="all",
    )
    p.add_argument("--quantize", dest="quantize", action="store_true", default=True)
    p.add_argument("--no-quantize", dest="quantize", action="store_false")
    args = p.parse_args()

    llm = BanglaLLama(quantize=args.quantize)
    tasks = get_tasks(llm)
    to_run = list(tasks.keys()) if args.task == "all" else [args.task]

    for task in to_run:
        print(f"\n{'='*60}")
        print(f"Model: {MODEL_ID} | Task: {task}")
        print(f"{'='*60}")
        tasks[task]()

    print("All done.")


if __name__ == "__main__":
    main()