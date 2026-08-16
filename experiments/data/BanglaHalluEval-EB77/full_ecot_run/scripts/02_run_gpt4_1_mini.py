#!/usr/bin/env python3
"""Full-benchmark E-CoT run on GPT-4.1 mini (OpenAI API).

Usage:
    python full_ecot_run/scripts/02_run_gpt4_1_mini.py --task all --track both [--no-resume]

Outputs CSVs in full_ecot_run/results/gpt4_1_mini/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ecot_core import ModelConfig, run_one
from _backend_openai import make_call_fn

SLUG = "gpt4_1_mini"
DISPLAY = "GPT-4.1 mini"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa", "summarization", "reasoning", "all"], default="all")
    ap.add_argument("--track", choices=["A", "B", "both"], default="both")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    model = ModelConfig(slug=SLUG, display_name=DISPLAY, call_fn=make_call_fn(model="gpt-4.1-mini"))
    tasks  = ("qa", "summarization", "reasoning") if args.task == "all" else (args.task,)
    tracks = ("A", "B") if args.track == "both" else (args.track,)
    for t in tasks:
        for tr in tracks:
            run_one(model, t, tr, resume=not args.no_resume)


if __name__ == "__main__":
    main()
