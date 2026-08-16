#!/usr/bin/env python3
"""
Extract Yes/No counts and compute hallucination-detection metrics from all COT evaluation files.

Ground truth per file:
  *_hallu_* / reasoning_cot_*  → all rows are hallucinated  (expected label = yes)
  *_gt_*    / reasoning_gt_cot → all rows are non-hallucinated (expected label = no)

The `is_hallucinated` column in every COT output file holds the model's prediction
(the original ground truth label was overwritten by the evaluation script).

Outputs (written to Evaluations/):
  cot_per_file_counts.csv   — yes/no counts per file
  cot_per_model_metrics.csv — precision / recall / F1 / accuracy per model × task
"""

import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# File registry
# Each entry: (relative_path, model_slug, task, expected_label)
#   expected_label = "yes" for hallucinated data, "no" for ground-truth data
# ---------------------------------------------------------------------------

FILES = [
    # ── QA hallucinated (4 000 rows, expected = yes) ─────────────────────
    ("QA/Results/qa_cot_hallu_llama3_1_8b.csv",      "llama3_1_8b",      "QA", "yes"),
    ("QA/Results/qa_cot_hallu_mistral_nemo.csv",      "mistral_nemo",     "QA", "yes"),
    ("QA/Results/qa_cot_hallu_qwen2_5_32b.csv",       "qwen2_5_32b",      "QA", "yes"),
    ("QA/Results/qa_cot_hallu_deepseek_r1_14b.csv",   "deepseek_r1_14b",  "QA", "yes"),
    ("QA/Results/qa_cot_hallu_gemma2_27b.csv",        "gemma2_27b",       "QA", "yes"),
    ("QA/Results/qa_cot_hallu_tigerllm_9b.csv",       "tigerllm_9b",      "QA", "yes"),
    ("QA/Results/qa_cot_hallu_gpt4_1_mini.csv",       "gpt4_1_mini",      "QA", "yes"),
    # ── QA ground truth (1 000 rows, expected = no) ───────────────────────
    ("QA/Results/qa_cot_gt_llama3_1_8b.csv",          "llama3_1_8b",      "QA", "no"),
    ("QA/Results/qa_cot_gt_mistral_nemo.csv",          "mistral_nemo",     "QA", "no"),
    ("QA/Results/qa_cot_gt_qwen2_5_32b.csv",           "qwen2_5_32b",      "QA", "no"),
    ("QA/Results/qa_cot_gt_deepseek_r1_14b.csv",       "deepseek_r1_14b",  "QA", "no"),
    ("QA/Results/qa_cot_gt_gemma2_27b.csv",            "gemma2_27b",       "QA", "no"),
    ("QA/Results/qa_cot_gt_tigerllm_9b.csv",           "tigerllm_9b",      "QA", "no"),
    ("QA/Results/qa_cot_gt_gpt4_1_mini.csv",           "gpt4_1_mini",      "QA", "no"),
    # ── Summarization hallucinated (3 000 rows, expected = yes) ───────────
    ("Summarization/Evaluation_Results/summ_3000_cot_llama3_1_8b.csv",     "llama3_1_8b",     "Summarization", "yes"),
    ("Summarization/Evaluation_Results/summ_3000_cot_mistral_nemo.csv",     "mistral_nemo",    "Summarization", "yes"),
    ("Summarization/Evaluation_Results/summ_3000_cot_qwen2_5_32b.csv",      "qwen2_5_32b",     "Summarization", "yes"),
    ("Summarization/Evaluation_Results/summ_3000_cot_deepseek_r1_14b.csv",  "deepseek_r1_14b", "Summarization", "yes"),
    ("Summarization/Evaluation_Results/summ_3000_cot_gemma2_27b.csv",       "gemma2_27b",      "Summarization", "yes"),
    ("Summarization/Evaluation_Results/summ_3000_cot_tigerllm_9b.csv",      "tigerllm_9b",     "Summarization", "yes"),
    ("Summarization/Evaluation_Results/summ_3000_cot_gpt4_1_mini.csv",      "gpt4_1_mini",     "Summarization", "yes"),
    # ── Summarization ground truth (1 000 rows, expected = no) ────────────
    ("Summarization/Evaluation_Results/summ_1000_cot_llama3_1_8b.csv",     "llama3_1_8b",     "Summarization", "no"),
    ("Summarization/Evaluation_Results/summ_1000_cot_mistral_nemo.csv",     "mistral_nemo",    "Summarization", "no"),
    ("Summarization/Evaluation_Results/summ_1000_cot_qwen2_5_32b.csv",      "qwen2_5_32b",     "Summarization", "no"),
    ("Summarization/Evaluation_Results/summ_1000_cot_deepseek_r1_14b.csv",  "deepseek_r1_14b", "Summarization", "no"),
    ("Summarization/Evaluation_Results/summ_1000_cot_gemma2_27b.csv",       "gemma2_27b",      "Summarization", "no"),
    ("Summarization/Evaluation_Results/summ_1000_cot_tigerllm_9b.csv",      "tigerllm_9b",     "Summarization", "no"),
    ("Summarization/Evaluation_Results/summ_1000_cot_gpt4_1_mini.csv",      "gpt4_1_mini",     "Summarization", "no"),
    # ── Reasoning hallucinated (~1 000 rows, expected = yes) ──────────────
    ("Reasoning/Results/reasoning_cot_llama3_1_8b.csv",     "llama3_1_8b",     "Reasoning", "yes"),
    ("Reasoning/Results/reasoning_cot_mistral_nemo.csv",     "mistral_nemo",    "Reasoning", "yes"),
    ("Reasoning/Results/reasoning_cot_qwen2_5_32b.csv",      "qwen2_5_32b",     "Reasoning", "yes"),
    ("Reasoning/Results/reasoning_cot_deepseek_r1_14b.csv",  "deepseek_r1_14b", "Reasoning", "yes"),
    ("Reasoning/Results/reasoning_cot_gemma2_27b.csv",       "gemma2_27b",      "Reasoning", "yes"),
    ("Reasoning/Results/reasoning_cot_tigerllm_9b.csv",      "tigerllm_9b",     "Reasoning", "yes"),
    ("Reasoning/Results/reasoning_cot_gpt4_1_mini.csv",      "gpt4_1_mini",     "Reasoning", "yes"),
    # ── Reasoning ground truth (~1 000 rows, expected = no) ───────────────
    ("Reasoning/Results/reasoning_gt_cot_llama3_1_8b.csv",     "llama3_1_8b",     "Reasoning", "no"),
    ("Reasoning/Results/reasoning_gt_cot_mistral_nemo.csv",     "mistral_nemo",    "Reasoning", "no"),
    ("Reasoning/Results/reasoning_gt_cot_qwen2_5_32b.csv",      "qwen2_5_32b",     "Reasoning", "no"),
    ("Reasoning/Results/reasoning_gt_cot_deepseek_r1_14b.csv",  "deepseek_r1_14b", "Reasoning", "no"),
    ("Reasoning/Results/reasoning_gt_cot_gemma2_27b.csv",       "gemma2_27b",      "Reasoning", "no"),
    ("Reasoning/Results/reasoning_gt_cot_tigerllm_9b.csv",      "tigerllm_9b",     "Reasoning", "no"),
    ("Reasoning/Results/reasoning_gt_cot_gpt4_1_mini.csv",      "gpt4_1_mini",     "Reasoning", "no"),
]

MODELS = ["llama3_1_8b", "mistral_nemo", "qwen2_5_32b",
          "deepseek_r1_14b", "gemma2_27b", "tigerllm_9b", "gpt4_1_mini"]
TASKS  = ["QA", "Summarization", "Reasoning"]


def read_predictions(path: Path) -> list[str]:
    preds = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            preds.append(row.get("is_hallucinated", "").strip().lower())
    return preds


def fmt(v: float) -> str:
    return f"{v:.4f}"


def main():
    out_dir = BASE / "Evaluations"
    out_dir.mkdir(exist_ok=True)

    per_file_rows = []
    # Accumulate TP/FP/TN/FN per (task, model)
    counters: dict[tuple, dict] = {
        (t, m): {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        for t in TASKS for m in MODELS
    }

    for rel_path, model, task, expected in FILES:
        path = BASE / rel_path
        subset = "hallucinated" if expected == "yes" else "ground_truth"

        if not path.exists():
            print(f"  [MISSING] {rel_path}")
            row = dict(file=rel_path, task=task, subset=subset, model=model,
                       expected=expected, total=0,
                       yes_count=0, no_count=0, unparsed=0,
                       yes_pct="", no_pct="", note="FILE MISSING")
            per_file_rows.append(row)
            continue

        preds = read_predictions(path)
        total = len(preds)
        yes   = sum(1 for p in preds if p == "yes")
        no    = sum(1 for p in preds if p == "no")
        other = total - yes - no

        yes_pct = f"{yes/total*100:.2f}%" if total else ""
        no_pct  = f"{no/total*100:.2f}%"  if total else ""

        row = dict(file=rel_path, task=task, subset=subset, model=model,
                   expected=expected, total=total,
                   yes_count=yes, no_count=no, unparsed=other,
                   yes_pct=yes_pct, no_pct=no_pct, note="")
        per_file_rows.append(row)

        c = counters[(task, model)]
        if expected == "yes":        # hallucinated data
            c["TP"] += yes
            c["FN"] += no + other    # missed hallucinations
        else:                        # ground-truth (non-hallucinated) data
            c["TN"] += no
            c["FP"] += yes + other   # false alarms

        print(f"  {task:15s} | {model:20s} | {subset:12s} | "
              f"yes={yes:5d} ({yes_pct:7s}) | no={no:5d} ({no_pct:7s}) | "
              f"unparsed={other}")

    # ── Write per-file counts ────────────────────────────────────────────────
    out1 = out_dir / "cot_per_file_counts.csv"
    fields1 = ["file", "task", "subset", "model", "expected",
                "total", "yes_count", "no_count", "unparsed",
                "yes_pct", "no_pct", "note"]
    with open(out1, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields1)
        w.writeheader()
        w.writerows(per_file_rows)
    print(f"\nPer-file counts  -> {out1}")

    # ── Compute per-model metrics ────────────────────────────────────────────
    metrics_rows = []
    for task in TASKS:
        for model in MODELS:
            c = counters[(task, model)]
            TP, FP, TN, FN = c["TP"], c["FP"], c["TN"], c["FN"]
            total = TP + FP + TN + FN
            if total == 0:
                continue

            accuracy  = (TP + TN) / total
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

            metrics_rows.append(dict(
                task=task, model=model,
                TP=TP, FP=FP, TN=TN, FN=FN, total=total,
                accuracy=fmt(accuracy),
                precision=fmt(precision),
                recall=fmt(recall),
                f1=fmt(f1),
                specificity=fmt(specificity),
            ))

    out2 = out_dir / "cot_per_model_metrics.csv"
    fields2 = ["task", "model", "TP", "FP", "TN", "FN", "total",
               "accuracy", "precision", "recall", "f1", "specificity"]
    with open(out2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        w.writerows(metrics_rows)
    print(f"Per-model metrics -> {out2}")

    # ── Print a quick summary table ──────────────────────────────────────────
    print("\n" + "="*90)
    print(f"{'Task':15s} {'Model':20s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} "
          f"{'F1':>7s} {'Spec':>7s} {'Total':>7s}")
    print("="*90)
    for r in metrics_rows:
        print(f"{r['task']:15s} {r['model']:20s} "
              f"{r['accuracy']:>7s} {r['precision']:>7s} {r['recall']:>7s} "
              f"{r['f1']:>7s} {r['specificity']:>7s} {r['total']:>7d}")
    print("="*90)


if __name__ == "__main__":
    main()
