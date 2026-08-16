#!/usr/bin/env python3
"""Reconstruct a partial summ_3000_deepseek_hallu.csv from:
  1. The local 9-row output CSV (rows 0-8)
  2. The RunPod session log file (rows 118-929)

Rows 9-117 are NOT recoverable and will be re-processed when you --resume.

Usage:
    python scripts/reconstruct_partial_output.py \
        --log "Incomplete_deepseek-r1_candidate_3000.txt"
"""

import csv
import re
import argparse
from pathlib import Path

INPUT_CSV   = "Hallucination Generated Answers/summarization_3000_corrected.csv"
OUTPUT_CSV  = "Summarization/Evaluation_Results/summ_3000_deepseek_hallu.csv"


def parse_log(log_path: str) -> dict:
    """Extract {row_index: (doc_id, label)} from log lines like '  118: 7179::Non-factual -> yes'."""
    pattern = re.compile(r"^\s+(\d+):\s+(.+?)\s+->\s+(yes|no)\s*$")
    entries = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                idx  = int(m.group(1))
                doc_id = m.group(2).strip()
                label  = m.group(3).strip()
                entries[idx] = (doc_id, label)
    return entries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="Incomplete_deepseek-r1_candidate_3000.txt",
                   help="Path to the RunPod session log file")
    args = p.parse_args()

    print(f"Parsing log: {args.log}")
    log_entries = parse_log(args.log)
    print(f"  Extracted {len(log_entries)} labeled rows from log (rows {min(log_entries)} to {max(log_entries)})")

    # Read input CSV
    with open(INPUT_CSV, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        input_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if "is_hallucinated" not in fieldnames:
        fieldnames.append("is_hallucinated")
    print(f"Input CSV: {len(input_rows)} rows")

    # Read existing local output (may have 9 rows for rows 0-8)
    local_labels: dict[str, str] = {}
    out_path = Path(OUTPUT_CSV)
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(OUTPUT_CSV, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                sid = row.get("id") or row.get("source_id")
                lbl = row.get("is_hallucinated", "")
                if sid and lbl in ("yes", "no"):
                    local_labels[sid] = lbl
        print(f"Local output CSV: {len(local_labels)} labeled rows")

    # Merge: local labels take precedence; log fills gaps
    output_rows = []
    for i, row in enumerate(input_rows):
        sid = row.get("id") or row.get("source_id") or str(i)
        if sid in local_labels:
            r_out = dict(row)
            r_out["is_hallucinated"] = local_labels[sid]
            output_rows.append(r_out)
        elif i in log_entries:
            log_id, label = log_entries[i]
            r_out = dict(row)
            r_out["is_hallucinated"] = label
            output_rows.append(r_out)
        else:
            break  # stop at first gap — script resumes from here

    print(f"\nReconstructed {len(output_rows)} labeled rows.")
    recovered_from_log = sum(1 for i, r in enumerate(input_rows[:len(output_rows)])
                             if (r.get("id") or r.get("source_id") or str(i)) not in local_labels
                             and i in log_entries)
    missing = len(output_rows) - len(local_labels) - recovered_from_log
    print(f"  From local CSV : {len(local_labels)}")
    print(f"  From log file  : {recovered_from_log}")
    print(f"  Still missing (rows 9-117, will be re-processed): ~109")
    print(f"  Still to label (rows {len(output_rows)+1}-2999): {3000 - len(output_rows) - 1} rows")

    # Overwrite the output CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nSaved reconstructed CSV to: {OUTPUT_CSV}")
    print("Next step: upload this file to your RunPod pod and run:")
    print("  python scripts/evaluate_deepseek_ollama.py --task summ_hallu")


if __name__ == "__main__":
    main()
