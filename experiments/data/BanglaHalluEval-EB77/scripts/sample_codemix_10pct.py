#!/usr/bin/env python3
"""Deterministic 10% samples of the codemix task, shared across all 9 models.

Writes:
  Sampling/10pct_codemix/gt.csv     (100 rows from Codemix/Main dataset/codemix_1000.csv)
  Sampling/10pct_codemix/hallu.csv  (400 rows from Hallucination Generated Answers/codemix_4000.csv)

Seed 42 so every model evaluates the *same* rows.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "Sampling" / "10pct_codemix"
SEED = 42

SOURCES = {
    "gt":    ("Codemix/Main dataset/codemix_1000.csv",                        100),
    "hallu": ("Hallucination Generated Answers/codemix_4000.csv",             400),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, (rel, n) in SOURCES.items():
        out = OUT_DIR / f"{key}.csv"
        if out.exists() and not args.force:
            print(f"[sample] {key}: {out} exists, skip")
            continue
        df = pd.read_csv(ROOT / rel, encoding="utf-8", on_bad_lines="skip")
        n = min(n, len(df))
        sub = df.sample(n=n, random_state=SEED).reset_index(drop=True)
        sub.to_csv(out, index=False, encoding="utf-8")
        print(f"[sample] {key}: wrote {len(sub)} rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
