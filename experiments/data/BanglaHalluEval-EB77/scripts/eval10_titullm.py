#!/usr/bin/env python3
"""TituLLM-3B (bf16) evaluation on the 10% rebuttal subset.

Runs both baseline and CoT modes across GQA / Summ / Reasoning / Codemix
(codemix skipped for CoT per rebuttal scope).

Guaranteed no "unknown" labels via the same constrained-yes/no strategy
used by eval10_banglallama.py.

Usage:
    python scripts/eval10_titullm.py --mode baseline
    python scripts/eval10_titullm.py --mode cot
    python scripts/eval10_titullm.py --mode both
    python scripts/eval10_titullm.py --mode baseline --tasks gqa_hallu,summ_hallu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise SystemExit("Missing deps. Run: pip install torch transformers accelerate")

from eval10_common import (
    BASELINE_TASKS, COT_TASKS, ROOT, SAMPLE_DIR,
    apply_chat, build_baseline_prompt, build_cot_prompt,
    build_samples, constrained_yesno, open_writer, parse_yesno,
    row_id, yes_no_token_ids,
)

MODEL_ID = "hishab/titulm-llama-3.2-3b-v1.1"
BATCH_SIZE = 8
COT_MAX_NEW = 400
OUT_BASE = ROOT / "T Sampled Evaluations" / "T_baseline_titullm"
OUT_COT  = ROOT / "T Sampled Evaluations" / "T_cot_titullm"


def load_model():
    print(f"[titullm] loading {MODEL_ID} in bfloat16 ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    print(f"[titullm] ready on {next(model.parameters()).device}")
    return tok, model


@torch.no_grad()
def run_baseline(tok, model, yes_ids, no_ids, tasks_filter=None):
    for tname, sample_key, answer_field, task_kind in BASELINE_TASKS:
        if tasks_filter and tname not in tasks_filter:
            continue
        src = SAMPLE_DIR / f"{sample_key}.csv"
        if not src.exists():
            print(f"[skip] {tname}: {src} missing (run sample_10pct.py first)")
            continue
        df = pd.read_csv(src, encoding="utf-8", on_bad_lines="skip")
        out_path = OUT_BASE / f"{tname}.csv"
        writer, _existing, done = open_writer(out_path, list(df.columns))
        pending_rows = [r for _, r in df.iterrows() if row_id(dict(r)) not in done]
        print(f"\n=== baseline / {tname} ===")
        print(f"  total {len(df)}   done {len(done)}   pending {len(pending_rows)}   -> {out_path.relative_to(ROOT)}")
        t0 = time.time()
        for i, r in enumerate(pending_rows):
            row = {k: r[k] for k in df.columns}
            prompt = apply_chat(tok, build_baseline_prompt(row, answer_field, task_kind))
            label = constrained_yesno(model, tok, prompt, yes_ids, no_ids)
            row_out = dict(row)
            row_out["raw_response"] = label
            row_out["is_hallucinated"] = label
            writer.writerow(row_out)
            writer._f.flush()
            if (i + 1) % 20 == 0 or i == len(pending_rows) - 1:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"  {i+1}/{len(pending_rows)}  ({rate:.2f}/s)  last={label}")
        writer._f.close()


@torch.no_grad()
def run_cot(tok, model, yes_ids, no_ids, tasks_filter=None):
    for tname, sample_key, answer_field, task_kind in COT_TASKS:
        if tasks_filter and tname not in tasks_filter:
            continue
        src = SAMPLE_DIR / f"{sample_key}.csv"
        if not src.exists():
            print(f"[skip] {tname}: {src} missing")
            continue
        df = pd.read_csv(src, encoding="utf-8", on_bad_lines="skip")
        out_path = OUT_COT / f"{tname}_cot.csv"
        writer, _existing, done = open_writer(out_path, list(df.columns))
        pending_rows = [r for _, r in df.iterrows() if row_id(dict(r)) not in done]
        print(f"\n=== cot / {tname} ===")
        print(f"  total {len(df)}   done {len(done)}   pending {len(pending_rows)}   -> {out_path.relative_to(ROOT)}")
        t0 = time.time()

        for bstart in range(0, len(pending_rows), BATCH_SIZE):
            batch = pending_rows[bstart:bstart + BATCH_SIZE]
            rows = [{k: r[k] for k in df.columns} for r in batch]
            prompts = [apply_chat(tok, build_cot_prompt(row, answer_field, task_kind)) for row in rows]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=3072).to(model.device)
            input_len = enc.input_ids.shape[1]
            out = model.generate(
                **enc,
                max_new_tokens=COT_MAX_NEW,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
                repetition_penalty=1.05,
            )
            raws = []
            for o in out:
                gen = o[input_len:]
                raws.append(tok.decode(gen, skip_special_tokens=True).strip())

            for row, prompt, raw in zip(rows, prompts, raws):
                verdict = parse_yesno(raw)
                if verdict == "unknown":
                    forced_prompt = prompt + "\n" + (raw[-400:] if raw else "") + \
                        "\n\nBased on the reasoning above, the final answer is:"
                    verdict = constrained_yesno(model, tok, forced_prompt, yes_ids, no_ids)
                row_out = dict(row)
                row_out["raw_response"] = raw
                row_out["is_hallucinated"] = verdict
                writer.writerow(row_out)
            writer._f.flush()

            done_n = bstart + len(batch)
            rate = done_n / max(time.time() - t0, 1e-6)
            print(f"  {done_n}/{len(pending_rows)}  ({rate:.2f}/s)")
        writer._f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "cot", "both"], default="both")
    p.add_argument("--tasks", type=str, default="",
                   help="Comma-separated task subset (e.g. gqa_hallu,summ_hallu)")
    args = p.parse_args()

    tasks_filter = set(t.strip() for t in args.tasks.split(",") if t.strip()) or None

    build_samples(force=False)
    tok, model = load_model()
    yes_ids, no_ids = yes_no_token_ids(tok)
    print(f"[titullm] yes_ids={yes_ids} no_ids={no_ids}")

    if args.mode in ("baseline", "both"):
        run_baseline(tok, model, yes_ids, no_ids, tasks_filter)
    if args.mode in ("cot", "both"):
        run_cot(tok, model, yes_ids, no_ids, tasks_filter)
    print("\nAll done.")


if __name__ == "__main__":
    main()
