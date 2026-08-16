#!/usr/bin/env python3
"""Compute per-judge BenHalluScore on the full benchmark.

For every judge with E-CoT outputs in full_ecot_run/results/<slug>/:
  • A-err: yes-fraction on Track A (GT) outputs (false alarm rate)
  • B-err: not-yes-fraction on Track B (hallu) outputs (miss rate)
  • BHS = 0.5 * (A-err + B-err) * 100

Also computes Baseline and CoT BHS for the same row IDs from the canonical
result files used by extract_baseline_metrics.py / extract_cot_metrics.py
so the three columns are directly comparable.

Outputs:
  full_ecot_run/results/<slug>/metrics.csv          (per-judge, per-task)
  full_ecot_run/results/_all_judges_metrics.csv     (concatenated)

Usage:
  python full_ecot_run/scripts/03_compute_metrics.py [--judges qwen2_5_32b ...]
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
RUN_RESULTS = ROOT / "full_ecot_run" / "results"

# Registry of (baseline, cot) per-task input files per judge — mirrors
# extract_baseline_metrics.py and extract_cot_metrics.py registries.
JUDGE_FILES = {
    "tigerllm_9b": {
        "qa":           {"baseline_gt":    "QA/Results/tigerllm_qa_1000_ground_truth_eval_results.csv",
                         "baseline_hallu": "QA/Results/tigerllm_qa_4000_eval_results.csv",
                         "cot_gt":         "QA/Results/qa_cot_gt_tigerllm_9b.csv",
                         "cot_hallu":      "QA/Results/qa_cot_hallu_tigerllm_9b.csv"},
        "summarization":{"baseline_gt":    "Summarization/Results/tigerllm_summarization_1000_ground_truth_eval_results.csv",
                         "baseline_hallu": "Summarization/Results/tigerllm_summarization_3000_eval_results.csv",
                         "cot_gt":         "Summarization/Results/summ_1000_cot_tigerllm_9b.csv",
                         "cot_hallu":      "Summarization/Results/summ_3000_cot_tigerllm_9b.csv"},
        "reasoning":    {"baseline_gt":    "Reasoning/Results/tigerllm_reasoning_1000_ground_truth_eval_results.csv",
                         "baseline_hallu": "Reasoning/Results/tigerllm_reasoning_1000_eval_results.csv",
                         "cot_gt":         "Reasoning/Results/reasoning_gt_cot_tigerllm_9b.csv",
                         "cot_hallu":      "Reasoning/Results/reasoning_cot_tigerllm_9b.csv"},
    },
    "gpt4_1_mini": {
        "qa":           {"baseline_gt":    "Evaluation/qa_1000_eval_gpt_4_1_mini.csv",
                         "baseline_hallu": "Evaluation/qa_4000_eval_gpt_4_1_mini.csv",
                         "cot_gt":         "QA/Results/qa_cot_gt_gpt4_1_mini.csv",
                         "cot_hallu":      "QA/Results/qa_cot_hallu_gpt4_1_mini.csv"},
        "summarization":{"baseline_gt":    "Summarization/Results/summarization_dataset_1000_eval_gpt_4_1_mini.csv",
                         "baseline_hallu": "Summarization/Results/summarization_3000_eval_gpt_4_1_mini.csv",
                         "cot_gt":         "Summarization/Results/summ_1000_cot_gpt4_1_mini.csv",
                         "cot_hallu":      "Summarization/Results/summ_3000_cot_gpt4_1_mini.csv"},
        "reasoning":    {"baseline_gt":    "Reasoning/Results/reasoning_main_1000_eval_gpt_4_1_mini.csv",
                         "baseline_hallu": "Reasoning/Results/reasoning_1000_eval_gpt_4_1_mini.csv",
                         "cot_gt":         "Reasoning/Results/reasoning_gt_cot_gpt4_1_mini.csv",
                         "cot_hallu":      "Reasoning/Results/reasoning_cot_gpt4_1_mini.csv"},
    },
    # The following judges may need their `baseline_*` paths confirmed; CoT
    # paths follow the same Summarization/Reasoning convention as above.
    "qwen2_5_32b":      {"qa": {}, "summarization": {}, "reasoning": {}},
    "deepseek_r1_14b":  {"qa": {}, "summarization": {}, "reasoning": {}},
    "gemma2_27b":       {"qa": {}, "summarization": {}, "reasoning": {}},
    "mistral_nemo":     {"qa": {}, "summarization": {}, "reasoning": {}},
    "llama3_1_8b":      {"qa": {}, "summarization": {}, "reasoning": {}},
}

# Join keys used by each task. Reasoning baseline historically lacks an id
# column, so we join on the question text.
JOIN_KEY = {"qa": "id", "summarization": "id", "reasoning": "question"}


def yes_no(s) -> str:
    if s is None:
        return "unknown"
    t = str(s).strip().lower()
    if t.startswith("y"):
        return "yes"
    if t.startswith("n"):
        return "no"
    return "unknown"


def load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, on_bad_lines="skip")


def err_track_A(df: pd.DataFrame, col: str = "is_hallucinated") -> tuple[int, int, float]:
    n = len(df); y = sum(1 for v in df[col].apply(yes_no) if v == "yes")
    return y, n, round(y / n * 100, 2) if n else float("nan")


def err_track_B(df: pd.DataFrame, col: str = "is_hallucinated") -> tuple[int, int, float]:
    n = len(df); w = sum(1 for v in df[col].apply(yes_no) if v != "yes")
    return w, n, round(w / n * 100, 2) if n else float("nan")


def metrics_for_judge(slug: str) -> list[dict]:
    rows = []
    for task in ("qa", "summarization", "reasoning"):
        ecot_gt    = RUN_RESULTS / slug / f"{task}_gt_ecot.csv"
        ecot_hallu = RUN_RESULTS / slug / f"{task}_hallu_ecot.csv"
        if not (ecot_gt.exists() and ecot_hallu.exists()):
            print(f"  [skip] {slug}/{task}: missing E-CoT output")
            continue

        df_gt    = pd.read_csv(ecot_gt)
        df_hallu = pd.read_csv(ecot_hallu)
        yA, nA, a_err = err_track_A(df_gt)
        wB, nB, b_err = err_track_B(df_hallu)
        bhs = round(0.5 * (a_err + b_err), 2)

        # Optional baseline + CoT joins, when paths are configured.
        files = JUDGE_FILES.get(slug, {}).get(task, {}) or {}
        bl_bhs = ct_bhs = float("nan")
        bl_a = bl_b = ct_a = ct_b = float("nan")
        key = JOIN_KEY[task]

        def sub_and_err(df_outer: pd.DataFrame | None, df_inner: pd.DataFrame, track: str):
            if df_outer is None: return float("nan"), 0
            df_outer = df_outer.copy()
            df_outer[key] = df_outer[key].astype(str).str.strip()
            df_inner = df_inner.copy()
            df_inner[key] = df_inner[key].astype(str).str.strip()
            ids = df_inner[key].tolist()
            sub = df_outer[df_outer[key].isin(ids)]
            n = len(sub)
            if n == 0: return float("nan"), 0
            preds = sub["is_hallucinated"].apply(yes_no)
            if track == "A":
                return round(sum(1 for v in preds if v == "yes") / n * 100, 2), n
            return round(sum(1 for v in preds if v != "yes") / n * 100, 2), n

        bl_gt    = load(ROOT / files.get("baseline_gt", ""))    if files.get("baseline_gt") else None
        bl_hallu = load(ROOT / files.get("baseline_hallu", "")) if files.get("baseline_hallu") else None
        ct_gt    = load(ROOT / files.get("cot_gt", ""))         if files.get("cot_gt") else None
        ct_hallu = load(ROOT / files.get("cot_hallu", ""))      if files.get("cot_hallu") else None

        bl_a, _ = sub_and_err(bl_gt, df_gt, "A")
        bl_b, _ = sub_and_err(bl_hallu, df_hallu, "B")
        ct_a, _ = sub_and_err(ct_gt, df_gt, "A")
        ct_b, _ = sub_and_err(ct_hallu, df_hallu, "B")
        bl_bhs = round(0.5 * (bl_a + bl_b), 2) if bl_a == bl_a and bl_b == bl_b else float("nan")
        ct_bhs = round(0.5 * (ct_a + ct_b), 2) if ct_a == ct_a and ct_b == ct_b else float("nan")

        rows.append({
            "judge": slug, "task": task,
            "n_A": nA, "n_B": nB,
            "baseline_a_err": bl_a, "baseline_b_err": bl_b, "baseline_bhs": bl_bhs,
            "cot_a_err":      ct_a, "cot_b_err":      ct_b, "cot_bhs":      ct_bhs,
            "ecot_a_err":     a_err, "ecot_b_err":    b_err, "ecot_bhs":     bhs,
            "delta_vs_baseline": round(bhs - bl_bhs, 2) if bl_bhs == bl_bhs else float("nan"),
            "delta_vs_cot":      round(bhs - ct_bhs, 2) if ct_bhs == ct_bhs else float("nan"),
        })
        print(f"  [{slug:18s}] {task:14s} BHS  base={bl_bhs}  cot={ct_bhs}  ecot={bhs}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="*", default=None, help="Subset of judge slugs to score.")
    args = ap.parse_args()

    judges = args.judges or sorted(p.name for p in RUN_RESULTS.iterdir() if p.is_dir())
    all_rows = []
    for slug in judges:
        print(f"\n=== {slug} ===")
        rows = metrics_for_judge(slug)
        if not rows:
            continue
        per = RUN_RESULTS / slug / "metrics.csv"
        with open(per, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        all_rows.extend(rows)
        print(f"  -> wrote {per.relative_to(ROOT)}")

    if all_rows:
        out = RUN_RESULTS / "_all_judges_metrics.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        print(f"\nAggregated -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
