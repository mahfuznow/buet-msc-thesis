#!/usr/bin/env python3
"""E-CoT (Variant C) evaluation for Mistral-Nemo-12B on the code-mixed task.

Runs only the 10% subsets built by scripts/sample_codemix_10pct.py:
    Sampling/10pct_codemix/gt.csv    (100 rows,  expected verdict = "no")
    Sampling/10pct_codemix/hallu.csv (400 rows,  expected verdict = "yes")

Reuses the existing E-CoT JSON parser and verdict aggregator from
_ecot_core.py so downstream metrics tools work unchanged.

Outputs:
    full_ecot_run/results/mistral_nemo/codemix_gt_ecot.csv
    full_ecot_run/results/mistral_nemo/codemix_hallu_ecot.csv

Usage:
    OLLAMA_BASE_URL=http://localhost:11434 \
        python full_ecot_run/scripts/02_run_mistral_nemo_codemix.py --track both
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ecot_core import (
    OUTPUT_FIELDS,
    MISSING_THRESHOLD,
    VERDICT_SOURCE,
    parse_json,
    make_output_row,
)
from _backend_ollama import make_call_fn

# ── Config ──────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "mistral-nemo:latest"
SLUG    = "mistral_nemo"
DISPLAY = "Mistral-Nemo-12B"
TASK    = "codemix"   # only used for MISSING_THRESHOLD / VERDICT_SOURCE lookup

ROOT = Path(__file__).resolve().parent.parent.parent
RUN_DIR = ROOT / "full_ecot_run"
PROMPT_PATH = RUN_DIR / "prompts" / "ecot_codemix.txt"
RESULTS_DIR = RUN_DIR / "results" / SLUG

TRACKS = {
    "A": {  # ground-truth
        "input":         ROOT / "Sampling" / "10pct_codemix" / "gt.csv",
        "context_col":   "codemix_context",
        "question_col":  "codemix_question",
        "candidate_col": "codemix_answer",
        "id_col":        "id",
        "out":           RESULTS_DIR / "codemix_gt_ecot.csv",
    },
    "B": {  # hallucinated
        "input":         ROOT / "Sampling" / "10pct_codemix" / "hallu.csv",
        "context_col":   "codemix_context",
        "question_col":  "codemix_question",
        "candidate_col": "hallucinated_answer",
        "id_col":        "id",
        "out":           RESULTS_DIR / "codemix_hallu_ecot.csv",
    },
}

# Codemix isn't in the frozen policy dict, so we mirror QA (0.30, agg verdict).
MISSING_THRESHOLD["codemix"] = 0.30
VERDICT_SOURCE["codemix"]    = "agg"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def load_done_keys(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            k = r.get("id_key")
            if k:
                done.add(k)
    return done


def open_writer(out_path: Path, original_fields: list):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = (not out_path.exists()) or out_path.stat().st_size == 0
    fieldnames = list(original_fields) + [f for f in OUTPUT_FIELDS if f not in original_fields]
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if fresh:
        w.writeheader(); f.flush()
    return f, w, fieldnames


def num_predict_for(task: str) -> int:
    # E-CoT JSON output — codemix answers are typically short; 1024 is plenty.
    return 1024


def num_ctx_for(task: str) -> int:
    return 4096


def run_track(call_fn: Callable[[str, str], str], track: str, resume: bool) -> None:
    cfg = TRACKS[track]
    template = load_prompt()
    df = pd.read_csv(cfg["input"], on_bad_lines="skip")
    id_col = cfg["id_col"]
    df[id_col] = df[id_col].astype(str).str.strip()

    out_path = cfg["out"]
    done = load_done_keys(out_path) if resume else set()
    original_fields = list(df.columns)
    f, writer, _ = open_writer(out_path, original_fields)

    pending = df[~df[id_col].isin(done)]
    print(f"\n[{SLUG}] codemix / track {track}: total {len(df)}  done {len(done)}  pending {len(pending)}")
    print(f"  writing -> {out_path.relative_to(ROOT)}")
    t0 = time.time()

    for i, (_, row) in enumerate(pending.iterrows(), 1):
        prompt = template.format(
            context=str(row[cfg["context_col"]]).strip(),
            question=str(row[cfg["question_col"]]).strip(),
            candidate=str(row[cfg["candidate_col"]]).strip(),
        )
        raw = call_fn(prompt, TASK)
        parsed, err = parse_json(raw)
        out_row = make_output_row(TASK, raw, parsed, err, id_key=str(row[id_col]))
        merged = {**{c: row[c] for c in original_fields}, **out_row}
        writer.writerow(merged)
        f.flush()

        if i % 10 == 0 or i == len(pending):
            rate = i / max(time.time() - t0, 1e-6)
            eta = (len(pending) - i) / max(rate, 1e-6) / 60
            print(f"  {i}/{len(pending)}  rate={rate:.2f}/s  eta={eta:.1f}min  "
                  f"last_verdict={out_row['is_hallucinated']}  parse_err={out_row['parse_error'] or '-'}")

    f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["A", "B", "both"], default="both")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    call_fn = make_call_fn(OLLAMA_MODEL, num_predict_for, num_ctx_for)
    tracks = ("A", "B") if args.track == "both" else (args.track,)
    for tr in tracks:
        run_track(call_fn, tr, resume=not args.no_resume)


if __name__ == "__main__":
    main()
