#!/usr/bin/env python3
"""E-CoT Variant C: GPT-4.1 mini — pilot-verify then full-benchmark pipeline.

Reads OPENAI_API_KEY from the .env file at the repo root.

Flow
----
Step 1  Pilot check
        If pilot_50_samples/results/pilot_metrics.csv already exists the
        results are displayed without spending any API calls.
        If it is missing the pilot script is invoked automatically.
        The full run is aborted if mean pilot E-CoT BHS > PILOT_BHS_LIMIT.

Step 2  Full benchmark (resumable)
        Runs QA / Summarization / Reasoning, tracks A and B.
        Writes to full_ecot_run/results/gpt4_1_mini/.
        Re-running the script skips rows whose id_key is already in the CSV.

Usage
-----
    python full_ecot_run/scripts/run_gpt4_1_mini_ecot.py
    python full_ecot_run/scripts/run_gpt4_1_mini_ecot.py --pilot-only
    python full_ecot_run/scripts/run_gpt4_1_mini_ecot.py --skip-pilot
    python full_ecot_run/scripts/run_gpt4_1_mini_ecot.py --task qa --track A
    python full_ecot_run/scripts/run_gpt4_1_mini_ecot.py --no-resume
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

from _ecot_core import ModelConfig, run_one
from _backend_openai import make_call_fn

ROOT      = Path(__file__).resolve().parent.parent.parent
PILOT_DIR = ROOT / "pilot_50_samples"
SLUG      = "gpt4_1_mini"

PILOT_BHS_LIMIT = 25.0   # abort full run if mean pilot E-CoT BHS exceeds this


# ── Pilot helpers ─────────────────────────────────────────────────────────────

def _read_pilot_metrics() -> list[dict] | None:
    path = PILOT_DIR / "results" / "pilot_metrics.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_pilot_script() -> None:
    script = PILOT_DIR / "scripts" / "02_run_ecot.py"
    print(f"[pilot] Running {script.relative_to(ROOT)} ...")
    result = subprocess.run(
        [sys.executable, str(script), "--task", "all", "--track", "both"],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        raise SystemExit("[pilot] Pilot script failed — fix errors above before proceeding.")


def pilot_phase() -> float:
    """Ensure pilot results exist; print table; return mean E-CoT BHS."""
    metrics = _read_pilot_metrics()
    if metrics is None:
        _run_pilot_script()
        metrics = _read_pilot_metrics()
    if not metrics:
        raise SystemExit("[pilot] pilot_metrics.csv not found even after running pilot.")

    print("\n" + "=" * 72)
    print("  PILOT RESULTS  (50 + 50 samples per task, GPT-4.1 mini)")
    print("=" * 72)
    print(f"  {'Task':15s} {'Baseline':>10s}  {'CoT':>8s}  {'E-CoT':>8s}  {'Δ vs Base':>11s}")
    print("  " + "-" * 68)
    ecot_scores: list[float] = []
    for r in metrics:
        task  = r["task"]
        base  = float(r["baseline_bhs"])
        cot   = float(r["cot_bhs"])
        ecot  = float(r["ecot_bhs"])
        delta = float(r["delta_bhs_vs_baseline"])
        ecot_scores.append(ecot)
        sign = "+" if delta >= 0 else ""
        print(f"  {task:15s} {base:>9.1f}%  {cot:>7.1f}%  {ecot:>7.1f}%  {sign}{delta:.1f} pp")

    mean_ecot = sum(ecot_scores) / len(ecot_scores)
    print("  " + "-" * 68)
    print(f"  {'Mean':15s} {'':>10s}  {'':>8s}  {mean_ecot:>7.1f}%")
    print("=" * 72 + "\n")
    return mean_ecot


# ── Full-benchmark runner ─────────────────────────────────────────────────────

def full_phase(tasks: tuple[str, ...], tracks: tuple[str, ...], resume: bool) -> None:
    call_fn = make_call_fn(model="gpt-4.1-mini")
    model   = ModelConfig(slug=SLUG, display_name="GPT-4.1 mini", call_fn=call_fn)
    for t in tasks:
        for tr in tracks:
            run_one(model, t, tr, resume=resume)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="E-CoT Variant C full-benchmark runner for GPT-4.1 mini."
    )
    ap.add_argument("--pilot-only",  action="store_true",
                    help="Display pilot results and exit without running the full benchmark.")
    ap.add_argument("--skip-pilot",  action="store_true",
                    help="Skip the pilot check and go straight to the full run.")
    ap.add_argument("--task",  choices=["qa", "summarization", "reasoning", "all"], default="all")
    ap.add_argument("--track", choices=["A", "B", "both"], default="both")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-evaluate all rows even if output CSV already exists.")
    args = ap.parse_args()

    # Load API key from .env
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not found.\n"
            f"Add it to {ROOT / '.env'} as:  OPENAI_API_KEY=sk-..."
        )

    # ── Step 1: Pilot ─────────────────────────────────────────────────────────
    if not args.skip_pilot:
        mean_bhs = pilot_phase()
        if mean_bhs > PILOT_BHS_LIMIT:
            raise SystemExit(
                f"[ABORT] Pilot mean E-CoT BHS = {mean_bhs:.1f}% exceeds limit "
                f"{PILOT_BHS_LIMIT}%. Review pilot results before scaling up."
            )
        print(
            f"[pilot] PASS  mean E-CoT BHS = {mean_bhs:.1f}%  "
            f"(limit = {PILOT_BHS_LIMIT}%)."
        )

    if args.pilot_only:
        print("[done] Pilot-only mode — exiting.")
        return

    # ── Step 2: Full benchmark ────────────────────────────────────────────────
    tasks  = ("qa", "summarization", "reasoning") if args.task == "all" else (args.task,)
    tracks = ("A", "B") if args.track == "both" else (args.track,)

    print(
        f"\n[full] Starting E-CoT full benchmark\n"
        f"       model  = gpt-4.1-mini\n"
        f"       tasks  = {tasks}\n"
        f"       tracks = {tracks}\n"
        f"       resume = {not args.no_resume}\n"
        f"       output = full_ecot_run/results/{SLUG}/\n"
    )
    full_phase(tasks, tracks, resume=not args.no_resume)
    print(f"\n[done] Full benchmark complete. Results in full_ecot_run/results/{SLUG}/")


if __name__ == "__main__":
    main()
