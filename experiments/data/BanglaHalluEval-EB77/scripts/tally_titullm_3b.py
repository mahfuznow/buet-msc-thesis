#!/usr/bin/env python3
"""Read-only tally of yes/no verdicts across TituLLM-3B baseline + CoT outputs.

Safe to run at any time during an active run — it only reads the CSV
files, never touches the running process. Uses csv.DictReader so
multi-line Bengali cells (which break `awk`) are parsed correctly.

Usage:
    python3 scripts/tally_titullm_3b.py
"""

import csv
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tally(pattern: str, phase: str) -> None:
    print(f"\n=== {phase} ({pattern}) ===")
    files = sorted(glob.glob(str(ROOT / pattern)))
    if not files:
        print("  (no files yet — phase hasn't started or produced output)")
        return
    total_yes = total_no = total_unk = 0
    for path in files:
        yes = no = unk = 0
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                for r in csv.DictReader(f):
                    v = (r.get("is_hallucinated") or "").strip().lower()
                    if v == "yes":
                        yes += 1
                    elif v == "no":
                        no += 1
                    else:
                        unk += 1
        except Exception as exc:
            print(f"  {os.path.basename(path):32s}  ERROR reading: {exc}")
            continue
        tot = yes + no + unk
        pct = (yes / tot * 100) if tot else 0.0
        print(f"  {os.path.basename(path):32s}  "
              f"yes={yes:5d}  no={no:5d}  unk={unk:3d}  tot={tot:5d}  yes%={pct:5.1f}")
        total_yes += yes
        total_no += no
        total_unk += unk
    grand = total_yes + total_no + total_unk
    if grand:
        print(f"  {'TOTAL':32s}  "
              f"yes={total_yes:5d}  no={total_no:5d}  unk={total_unk:3d}  tot={grand:5d}  "
              f"yes%={total_yes / grand * 100:5.1f}")


def main() -> None:
    tally("scripts/results_titullm_3b/*.csv",     "Baseline (4 tasks x 2 tracks)")
    tally("scripts/results_titullm_3b_cot/*.csv", "CoT (3 tasks x 2 tracks, no codemix)")


if __name__ == "__main__":
    main()
