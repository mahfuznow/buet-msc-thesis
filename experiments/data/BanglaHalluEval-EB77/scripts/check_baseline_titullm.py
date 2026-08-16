#!/usr/bin/env python3
"""Comprehensive inspection of TituLLM-3B baseline output CSVs.

Prints per-file: size, row count, column layout, yes/no/unk distribution
plus 2 sample rows. Ends with grand totals across all files.

Read-only. Safe to run while other jobs are active.
"""
import csv
import glob
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
BASELINE_GLOB = str(ROOT / "scripts" / "results_titullm_3b" / "*.csv")

print("=" * 92)
print(f"BASELINE OUTPUTS SUMMARY  ({BASELINE_GLOB})")
print("=" * 92)

files = sorted(glob.glob(BASELINE_GLOB))
if not files:
    print("  (no files found — is your working directory the repo root?)")
    raise SystemExit(0)

for path in files:
    fname = os.path.basename(path)
    size_kb = os.path.getsize(path) / 1024
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"\n[{fname}]  EMPTY")
        continue
    cols = list(rows[0].keys())
    y = n = u = 0
    blank_ids = 0
    for r in rows:
        v = (r.get("is_hallucinated") or "").strip().lower()
        if v == "yes":
            y += 1
        elif v == "no":
            n += 1
        else:
            u += 1
        sid = r.get("id") or r.get("source_id") or r.get("question_id") or ""
        if not str(sid).strip():
            blank_ids += 1
    cov = (y + n) / len(rows) * 100
    print(f"\n[{fname}]  size={size_kb:6.1f} KB   rows={len(rows):5d}   "
          f"yes={y:5d}  no={n:5d}  unk={u:4d}   yes%={y/len(rows)*100:5.1f}   cov%={cov:5.1f}")
    print(f"   columns ({len(cols)}): {cols}")
    if blank_ids:
        print(f"   WARNING: {blank_ids} rows have blank id/source_id/question_id")
    # samples: 1 yes + 1 no if both exist; else first 2
    samples = []
    for want in ("yes", "no"):
        for r in rows:
            if (r.get("is_hallucinated") or "").strip().lower() == want:
                samples.append(r)
                break
    if len(samples) < 2:
        samples = rows[:2]
    for i, r in enumerate(samples[:2], 1):
        cand_field = None
        for k in ("correct_answer", "hallucinated_answer", "summary",
                  "hallucinated_summary", "answer", "hallucinated_chain",
                  "codemix_answer"):
            if k in r:
                cand_field = k
                break
        cand = (r.get(cand_field, "") or "")[:60] if cand_field else ""
        print(f"   sample {i}: is_hallucinated={r.get('is_hallucinated', '')!r}   "
              f"{cand_field or 'candidate'}={cand!r}")

# Grand totals
print("\n" + "=" * 92)
print("GRAND TOTALS")
print("=" * 92)
gy = gn = gu = gtot = 0
for path in files:
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("is_hallucinated") or "").strip().lower()
            if v == "yes":
                gy += 1
            elif v == "no":
                gn += 1
            else:
                gu += 1
            gtot += 1
print(f"  total rows across {len(files)} files: {gtot}")
if gtot:
    print(f"  yes: {gy:6d}   ({gy/gtot*100:5.1f}%)")
    print(f"  no:  {gn:6d}   ({gn/gtot*100:5.1f}%)")
    print(f"  unk: {gu:6d}   ({gu/gtot*100:5.1f}%)")
