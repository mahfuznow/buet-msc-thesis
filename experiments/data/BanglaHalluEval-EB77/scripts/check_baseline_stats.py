#!/usr/bin/env python3
"""Quick script to check baseline stats for TituLLM-3B (unknowns, yes/no counts)."""
import csv, glob

BASELINE_GLOB = "scripts/results_titullm_3b/*_labeled.csv"

# Per-file stats
for path in sorted(glob.glob(BASELINE_GLOB)):
    y = n = u = tot = 0
    unknowns = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for idx, r in enumerate(csv.DictReader(f)):
            v = (r.get("is_hallucinated") or "").strip().lower()
            if v == "yes":  y += 1
            elif v == "no": n += 1
            else:
                u += 1
                unknowns.append((idx, v))
            tot += 1
    print(f"{path}")
    print(f"  total={tot}  yes={y} ({y/tot*100:.1f}%)  no={n} ({n/tot*100:.1f}%)  unk={u} ({u/tot*100:.1f}%)")
    if unknowns:
        for idx, v in unknowns[:5]:
            print(f"    unknown row {idx}: '{v}'")
        if len(unknowns) > 5:
            print(f"    ... and {len(unknowns)-5} more unknowns")
    print()

# Grand totals
print("=" * 90)
print("GRAND TOTALS")
print("=" * 90)
gy = gn = gu = gtot = 0
for path in sorted(glob.glob(BASELINE_GLOB)):
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("is_hallucinated") or "").strip().lower()
            if v == "yes":  gy += 1
            elif v == "no": gn += 1
            else:           gu += 1
            gtot += 1
print(f"  total rows across 8 files: {gtot}")
print(f"  yes: {gy:6d}   ({gy/gtot*100:5.1f}%)")
print(f"  no:  {gn:6d}   ({gn/gtot*100:5.1f}%)")
print(f"  unk: {gu:6d}   ({gu/gtot*100:5.1f}%)")
