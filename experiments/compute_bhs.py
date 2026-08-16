#!/usr/bin/env python3
"""
Compute BenHalluScore (BHS) from SelfCheckGPT output CSVs.

Run after selfcheck_qa.py has produced Track A and Track B output files.
Prints a comparison table against BanglaHalluEval's published numbers.

Usage:
  python compute_bhs.py \\
    --track-a results/selfcheck_qa_1000_gt_qwen7b.csv \\
    --track-b results/selfcheck_qa_4000_qwen7b.csv \\
    --label-col selfcheck_label \\
    --threshold 0.75
"""

import argparse
import csv
from pathlib import Path


def load_labels(path: Path, label_col: str) -> list[str]:
    labels = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            label = row.get(label_col, "").strip().lower()
            if label in ("yes", "no"):
                labels.append(label)
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track-a", required=True,
                   help="Track A CSV (ground truth — correct answers, expect label=no)")
    p.add_argument("--track-b", required=True,
                   help="Track B CSV (hallucinated answers, expect label=yes)")
    p.add_argument("--label-col", default="selfcheck_label",
                   help="Column containing yes/no labels (default: selfcheck_label)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Threshold used (for display only)")
    args = p.parse_args()

    a_labels = load_labels(Path(args.track_a), args.label_col)
    b_labels = load_labels(Path(args.track_b), args.label_col)

    # Track A: correct answers labeled 'yes' = false positive (error)
    a_errors = sum(1 for l in a_labels if l == "yes")
    a_rate   = a_errors / len(a_labels) if a_labels else 0

    # Track B: hallucinated answers labeled 'no' = missed detection (error)
    b_errors = sum(1 for l in b_labels if l == "no")
    b_rate   = b_errors / len(b_labels) if b_labels else 0

    bhs = 0.5 * (a_rate + b_rate) * 100

    thresh_str = f"  (threshold={args.threshold})" if args.threshold else ""

    print("\n" + "=" * 58)
    print("  SelfCheckGPT BHS Results — QA Task")
    if thresh_str:
        print(f"  {thresh_str.strip()}")
    print("=" * 58)
    print(f"  Track A (ground truth):   n={len(a_labels):>5}  "
          f"error rate = {a_rate*100:5.2f}%")
    print(f"  Track B (hallucinated):   n={len(b_labels):>5}  "
          f"error rate = {b_rate*100:5.2f}%")
    print(f"  BenHalluScore (BHS):      {bhs:.2f}%")
    print("=" * 58)

    print("\n  Published BHS for comparison (GQA, zero-shot):")
    published = [
        ("GPT-4.1 mini",     "Multilingual",  15.56),
        ("TigerLLM-9B",      "Bangla-centric", 28.28),
        ("DeepSeek-R1-14B",  "Reasoning",     38.60),
        ("Qwen2.5-32B",      "Multilingual",  46.85),
        ("Gemma-2-27B",      "Multilingual",  47.45),
        ("LLaMA-3.1-8B",     "Multilingual",  47.71),
        ("Mistral-nemo-12B", "Multilingual",  53.84),
    ]
    print(f"  {'Model':<22} {'Category':<16} {'BHS':>6}")
    print(f"  {'-'*22} {'-'*16} {'-'*6}")
    for name, cat, score in published:
        print(f"  {name:<22} {cat:<16} {score:>5.2f}%")

    print(f"\n  Your SelfCheckGPT (qwen2.5:7b, n=5)    {bhs:>5.2f}%")
    print("=" * 58)

    # Interpretation
    print("\n  Interpretation:")
    if bhs < 20:
        note = "Excellent — better than all published baselines"
    elif bhs < 30:
        note = "Strong — better than most published models"
    elif bhs < 45:
        note = "Moderate — competitive with mid-range models"
    else:
        note = "Weak — worse than most published models"
    print(f"  {note}")

    print("\n  Track A error rate interpretation:")
    print(f"  {a_rate*100:.1f}% of correct answers were flagged as hallucinated")
    print(f"  (false positive rate)")

    print("\n  Track B error rate interpretation:")
    print(f"  {b_rate*100:.1f}% of actual hallucinations were missed")
    print(f"  (missed detection rate)\n")


if __name__ == "__main__":
    main()
