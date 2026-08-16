#!/usr/bin/env python3
"""Compute BenHalluScore for the 10% rebuttal runs.

Pairs each gt track with its hallu counterpart, per model, per mode
(baseline / cot), per task (gqa / summ / reason / codemix — codemix only
in baseline).

Metric definitions (identical to scripts/extract_baseline_metrics.py):
    A-err = FP / N_A * 100      wrong on ground-truth rows
    B-err = FN / N_B * 100      wrong on hallucinated rows
    BHS   = 0.5 * (A-err + B-err)      lower is better

Reads:
    T Sampled Evaluations/T_baseline_<model>/<task>_gt.csv    (expected=no)
    T Sampled Evaluations/T_baseline_<model>/<task>_hallu.csv (expected=yes)
    T Sampled Evaluations/T_cot_<model>/<task>_gt_cot.csv
    T Sampled Evaluations/T_cot_<model>/<task>_hallu_cot.csv

Writes:
    T Sampled Evaluations/T_bhs.csv
    also prints a compact table.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "T Sampled Evaluations" / "T_bhs.csv"

MODELS = ["banglallama", "titullm"]
BASELINE_TASKS = ["gqa", "summ", "reason", "codemix"]
COT_TASKS      = ["gqa", "summ", "reason"]


def count(path: Path) -> Counter:
    c = Counter()
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            c[(row.get("is_hallucinated") or "").strip().lower()] += 1
    return c


def bhs_row(gt_path: Path, hallu_path: Path):
    if not gt_path.exists() or not hallu_path.exists():
        missing = []
        if not gt_path.exists():    missing.append(gt_path.name)
        if not hallu_path.exists(): missing.append(hallu_path.name)
        return None, None, None, None, None, f"missing: {', '.join(missing)}"

    gt = count(gt_path)
    hallu = count(hallu_path)

    # ground-truth track: expected label = "no"
    FP = gt.get("yes", 0)
    TN = gt.get("no", 0)
    N_A = FP + TN

    # hallucinated track: expected label = "yes"
    TP = hallu.get("yes", 0)
    FN = hallu.get("no", 0)
    N_B = TP + FN

    a_err = round(FP / N_A * 100, 2) if N_A > 0 else 0.0
    b_err = round(FN / N_B * 100, 2) if N_B > 0 else 0.0
    score = round(0.5 * (a_err + b_err), 2)
    return a_err, b_err, score, N_A, N_B, "OK"


def main() -> None:
    rows = []
    for model in MODELS:
        for mode, tasks in (("baseline", BASELINE_TASKS), ("cot", COT_TASKS)):
            d = ROOT / "T Sampled Evaluations" / f"T_{mode}_{model}"
            suffix = "_cot" if mode == "cot" else ""
            for task in tasks:
                gt_path    = d / f"{task}_gt{suffix}.csv"
                hallu_path = d / f"{task}_hallu{suffix}.csv"
                a, b, s, na, nb, note = bhs_row(gt_path, hallu_path)
                rows.append({
                    "model": model, "mode": mode, "task": task,
                    "n_gt": na if na is not None else "",
                    "n_hallu": nb if nb is not None else "",
                    "a_err": a if a is not None else "",
                    "b_err": b if b is not None else "",
                    "bhs":   s if s is not None else "",
                    "note":  note,
                })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","mode","task","n_gt","n_hallu","a_err","b_err","bhs","note"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {OUT_FILE.relative_to(ROOT)}\n")

    print("=" * 88)
    print(f"{'Model':13s} {'Mode':8s} {'Task':10s} {'N_gt':>6s} {'N_hal':>6s} {'A-err':>7s} {'B-err':>7s} {'BHS':>7s}  Note")
    print("=" * 88)
    for r in rows:
        print(f"{r['model']:13s} {r['mode']:8s} {r['task']:10s} "
              f"{str(r['n_gt']):>6s} {str(r['n_hallu']):>6s} "
              f"{str(r['a_err']):>7s} {str(r['b_err']):>7s} {str(r['bhs']):>7s}  {r['note']}")
    print("=" * 88)


if __name__ == "__main__":
    main()
