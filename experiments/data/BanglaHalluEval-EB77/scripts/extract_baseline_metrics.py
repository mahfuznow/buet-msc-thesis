#!/usr/bin/env python3
"""
Extract baseline (before-CoT) BenHalluScore metrics for the 5 remaining models:
  qwen2_5_32b, deepseek_r1_14b, gemma2_27b, tigerllm_9b, gpt4_1_mini

For each model x task, reads the hallu file (expected = yes) and GT file (expected = no),
counts predictions, then computes:
  A-err  = FP / N_A * 100   (wrong on ground-truth instances)
  B-err  = FN / N_B * 100   (wrong on hallucinated instances)
  BHS    = 0.5 * (A-err + B-err)

Output: Evaluations/baseline_metrics.csv
"""

import csv
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent

def count_labels(path: Path) -> Counter:
    c = Counter()
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            v = row.get("is_hallucinated", "").strip().lower()
            c[v] += 1
    return c

def bhs(a_err: float, b_err: float) -> float:
    return round(0.5 * (a_err + b_err), 2)

def metrics(hallu_path, gt_path):
    """Return (A-err, B-err, BHS, note) or None if files missing."""
    notes = []

    # --- hallu file (expected = yes) ---
    if hallu_path is None or not (BASE / hallu_path).exists():
        notes.append(f"MISSING hallu: {hallu_path}")
        TP = FN = None
    else:
        c = count_labels(BASE / hallu_path)
        yes = c.get("yes", 0)
        no  = c.get("no",  0)
        TP  = yes
        FN  = no
        N_B = yes + no
        print(f"    hallu {Path(hallu_path).name}: yes={yes}, no={no}, total={N_B}")

    # --- gt file (expected = no) ---
    if gt_path is None or not (BASE / gt_path).exists():
        notes.append(f"MISSING gt: {gt_path}")
        FP = TN = None
    else:
        c = count_labels(BASE / gt_path)
        yes = c.get("yes", 0)
        no  = c.get("no",  0)
        FP  = yes
        TN  = no
        N_A = yes + no
        print(f"    gt    {Path(gt_path).name}: yes={yes}, no={no}, total={N_A}")

    if TP is None or FP is None:
        return None, None, None, "; ".join(notes) or "OK"

    N_B = TP + FN
    N_A = FP + TN

    a_err = round(FP / N_A * 100, 2) if N_A > 0 else 0.0
    b_err = round(FN / N_B * 100, 2) if N_B > 0 else 0.0
    score = bhs(a_err, b_err)
    return a_err, b_err, score, "; ".join(notes) or "OK"


# ---------------------------------------------------------------------------
# File registry: (hallu_rel_path, gt_rel_path)
#   hallu -> all rows expected = yes (B-track)
#   gt    -> all rows expected = no  (A-track)
# ---------------------------------------------------------------------------

REGISTRY = {
    # ── TigerLLM-9B ──────────────────────────────────────────────────────
    ("QA",            "tigerllm_9b"): (
        "tigerllm_qa_4000_eval_results.csv",
        "tigerllm_qa_1000_ground_truth_eval_results.csv",
    ),
    ("Summarization", "tigerllm_9b"): (
        "tigerllm_summarization_3000_eval_results.csv",
        "tigerllm_summarization_1000_ground_truth_eval_results.csv",
    ),
    ("Reasoning",     "tigerllm_9b"): (
        "tigerllm_reasoning_1000_eval_results.csv",
        "tigerllm_reasoning_1000_ground_truth_eval_results.csv",
    ),

    # ── GPT-4.1 mini ─────────────────────────────────────────────────────
    ("QA",            "gpt4_1_mini"): (
        "Evaluation/qa_4000_eval_gpt_4_1_mini.csv",
        "Evaluation/qa_1000_eval_gpt_4_1_mini.csv",
    ),
    ("Summarization", "gpt4_1_mini"): (
        "Results/summarization_3000_eval_gpt_4_1_mini.csv",
        "Results/summarization_dataset_1000_eval_gpt_4_1_mini.csv",
    ),
    ("Reasoning",     "gpt4_1_mini"): (
        "Results/reasoning_1000_eval_gpt_4_1_mini.csv",
        "Results/reasoning_main_1000_eval_gpt_4_1_mini.csv",
    ),

    # ── Qwen2.5-32B ──────────────────────────────────────────────────────
    ("QA",            "qwen2_5_32b"): (
        None,                                        # no QA-hallu baseline found
        "Results/hallucinated_1000_qwen_labels.csv", # GT correct-answer file
    ),
    ("Summarization", "qwen2_5_32b"): (
        "Summarization/Evaluation_Results/summarization_3000_corrected_labeled_qwen32b.csv",
        "Summarization/Evaluation_Results/summarization_1000_labeled_qwen2_5_32b.csv",
    ),
    ("Reasoning",     "qwen2_5_32b"): (
        "Reasoning/Results/reasoning_evaluation_scored.csv",          # hallu (generic scored)
        "Reasoning/Results/reasoning_evaluation_scored_groundtruth_qwen2_5_32b.csv",
    ),

    # ── DeepSeek-R1-14B ──────────────────────────────────────────────────
    ("QA",            "deepseek_r1_14b"): (
        None,   # no QA-hallu baseline found
        None,   # no QA-GT baseline found
    ),
    ("Summarization", "deepseek_r1_14b"): (
        "Summarization/Evaluation_Results/summ_3000_deepseek_hallu.csv",
        "Summarization/Evaluation_Results/summ_gt_1000_deepseek.csv",
    ),
    ("Reasoning",     "deepseek_r1_14b"): (
        "Reasoning/Results/reasoning_1000_candidates_deepseek.csv",
        "Reasoning/Results/reasoning_main_1000_deepseek.csv",
    ),

    # ── Gemma2-27B ───────────────────────────────────────────────────────
    ("QA",            "gemma2_27b"): (
        None,   # no QA-hallu baseline found
        None,   # no QA-GT baseline found
    ),
    ("Summarization", "gemma2_27b"): (
        "Summarization/Evaluation_Results/summarization_3000_corrected_labeled_gemma2_27b.csv",
        "Summarization/Evaluation_Results/summarization_1000_labeled_gemma2_27b.csv",
    ),
    ("Reasoning",     "gemma2_27b"): (
        "Reasoning/Results/reasoning_evaluation_scored_gemma2.csv",
        "Reasoning/Results/reasoning_evaluation_scored_groundtruth_gemma2_27b.csv",
    ),
}

TASKS  = ["QA", "Summarization", "Reasoning"]
MODELS = ["tigerllm_9b", "gpt4_1_mini", "qwen2_5_32b", "deepseek_r1_14b", "gemma2_27b"]


def main():
    rows = []
    for task in TASKS:
        for model in MODELS:
            key = (task, model)
            hallu_path, gt_path = REGISTRY.get(key, (None, None))
            print(f"\n[{task}] {model}")
            a_err, b_err, score, note = metrics(hallu_path, gt_path)
            rows.append({
                "task":    task,
                "model":   model,
                "a_err":   a_err if a_err is not None else "N/A",
                "b_err":   b_err if b_err is not None else "N/A",
                "bhs":     score if score is not None else "N/A",
                "note":    note,
            })
            if score is not None:
                print(f"  => A-err={a_err:.2f}%  B-err={b_err:.2f}%  BHS={score:.2f}%")
            else:
                print(f"  => N/A  ({note})")

    out = BASE / "Evaluations" / "baseline_metrics.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task","model","a_err","b_err","bhs","note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {out}")

    # Pretty print summary
    print("\n" + "="*70)
    print(f"{'Task':15s} {'Model':20s} {'A-err':>8s} {'B-err':>8s} {'BHS':>8s}")
    print("="*70)
    for r in rows:
        print(f"{r['task']:15s} {r['model']:20s} "
              f"{str(r['a_err']):>8s} {str(r['b_err']):>8s} {str(r['bhs']):>8s}")
    print("="*70)


if __name__ == "__main__":
    main()
