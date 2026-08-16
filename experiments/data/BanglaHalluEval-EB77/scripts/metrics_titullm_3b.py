#!/usr/bin/env python3
"""Per-task A-err / B-err / BHS for TituLLM-3B baseline + CoT outputs.

Reads the labeled CSVs under scripts/results_titullm_3b/ and
scripts/results_titullm_3b_cot/, buckets rows by track (gt vs hallu),
and reports:

  A-err = fraction of GT rows judged "yes" (false alarms)      %
  B-err = fraction of hallu rows judged NOT "yes" (misses)     %
  BHS   = 0.5 * (A-err + B-err)                                %

Unknown/empty labels are counted separately (they contribute to
B-err on the hallu side and to a "coverage" figure that we print).

Usage:  python3 scripts/metrics_titullm_3b.py
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

# task_key -> (relative path pattern, track "A"=gt / "B"=hallu)
BASELINE_FILES = {
    ("gqa",     "A"): "scripts/results_titullm_3b/gqa_gt_labeled.csv",
    ("gqa",     "B"): "scripts/results_titullm_3b/gqa_hallu_labeled.csv",
    ("summ",    "A"): "scripts/results_titullm_3b/summ_gt_labeled.csv",
    ("summ",    "B"): "scripts/results_titullm_3b/summ_hallu_labeled.csv",
    ("reason",  "A"): "scripts/results_titullm_3b/reason_gt_labeled.csv",
    ("reason",  "B"): "scripts/results_titullm_3b/reason_hallu_labeled.csv",
    ("codemix", "A"): "scripts/results_titullm_3b/codemix_gt_labeled.csv",
    ("codemix", "B"): "scripts/results_titullm_3b/codemix_hallu_labeled.csv",
}

COT_FILES = {
    ("gqa",    "A"): "scripts/results_titullm_3b_cot/gqa_gt_cot.csv",
    ("gqa",    "B"): "scripts/results_titullm_3b_cot/gqa_hallu_cot.csv",
    ("summ",   "A"): "scripts/results_titullm_3b_cot/summ_gt_cot.csv",
    ("summ",   "B"): "scripts/results_titullm_3b_cot/summ_hallu_cot.csv",
    ("reason", "A"): "scripts/results_titullm_3b_cot/reason_gt_cot.csv",
    ("reason", "B"): "scripts/results_titullm_3b_cot/reason_hallu_cot.csv",
}


def bucket(path):
    """Return (yes, no, unknown, total) for a labeled CSV."""
    if not os.path.exists(path):
        return None
    y = n = u = 0
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("is_hallucinated") or "").strip().lower()
            if v == "yes":  y += 1
            elif v == "no": n += 1
            else:           u += 1
    return y, n, u, y + n + u


def a_err(yes, total_gt):
    """A-err: fraction of GT rows labeled 'yes' (false alarm)."""
    return (yes / total_gt * 100) if total_gt else float("nan")


def b_err(yes, total_hallu):
    """B-err: fraction of hallu rows NOT labeled 'yes' (miss).
    Unknown/empty count as misses (the model didn't correctly flag them).
    """
    return ((total_hallu - yes) / total_hallu * 100) if total_hallu else float("nan")


def report(name, files, tasks):
    print(f"\n{'='*84}")
    print(f"  {name}")
    print(f"{'='*84}")
    print(f"  {'Task':10s} {'A-err':>8s} {'B-err':>8s} {'BHS':>8s}   "
          f"{'GT y/n/u':>16s}   {'Hallu y/n/u':>16s}   {'Cov%':>6s}")
    print(f"  {'-'*80}")
    for task in tasks:
        gt = bucket(str(ROOT / files.get((task, "A"), "")))
        hl = bucket(str(ROOT / files.get((task, "B"), "")))
        if gt is None or hl is None:
            print(f"  {task:10s}   MISSING FILE(s) — skip")
            continue
        gy, gn, gu, gtot = gt
        hy, hn, hu, htot = hl
        aerr = a_err(gy, gtot)
        berr = b_err(hy, htot)
        bhs = 0.5 * (aerr + berr) if aerr == aerr and berr == berr else float("nan")
        cov = (gy + gn + hy + hn) / (gtot + htot) * 100 if (gtot + htot) else 0
        print(f"  {task:10s} {aerr:>7.2f}% {berr:>7.2f}% {bhs:>7.2f}%   "
              f"{gy:>4d}/{gn:>3d}/{gu:>4d}({gtot:4d})   "
              f"{hy:>4d}/{hn:>3d}/{hu:>4d}({htot:4d})   {cov:>5.1f}%")


def main() -> None:
    print(f"BenHalluEval — TituLLM-3B metrics")
    print(f"Model: hishab/titulm-llama-3.2-3b-v1.1\n")
    print("Legend:")
    print("  A-err = % of GT rows judged 'yes' (false alarms — lower better)")
    print("  B-err = % of hallu rows NOT judged 'yes' (misses — lower better)")
    print("          unknown/empty labels count as misses on the hallu track")
    print("  BHS   = 0.5 * (A-err + B-err)   (BenHalluScore — lower better)")
    print("  Cov%  = fraction of rows the parser could label yes/no")

    report("BASELINE (4 tasks x 2 tracks)", BASELINE_FILES,
           tasks=["gqa", "summ", "reason", "codemix"])
    report("CoT (3 tasks x 2 tracks, no codemix)", COT_FILES,
           tasks=["gqa", "summ", "reason"])


if __name__ == "__main__":
    main()
