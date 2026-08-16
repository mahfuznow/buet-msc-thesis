#!/usr/bin/env python3
"""Materialize the deterministic 10% subsets used by the rebuttal eval.

Writes:
  Sampling/10pct/qa.csv           (400 rows, seed 42)
  Sampling/10pct/summ_3000.csv    (300 rows)
  Sampling/10pct/summ_1000.csv    (100 rows)
  Sampling/10pct/reason_hallu.csv (100 rows)
  Sampling/10pct/reason_gt.csv    (100 rows)
  Sampling/10pct/codemix.csv      (400 rows)

Usage:
    python scripts/sample_10pct.py            # skip already-existing
    python scripts/sample_10pct.py --force    # regenerate
"""

import argparse
from eval10_common import build_samples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Overwrite existing sample files")
    args = p.parse_args()
    build_samples(force=args.force)


if __name__ == "__main__":
    main()
