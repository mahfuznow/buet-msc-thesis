#!/usr/bin/env python3
"""Full-benchmark E-CoT run on LLaMA-3.1-8B-Instruct (Ollama).

This is the highest-leverage judge to scale up: in the original benchmark,
plain CoT regressed LLaMA-3.1 on QA (BHS 45.93 -> 48.10) and Summarization
(28.79 -> 43.35). The pilot showed E-CoT recovers both regressions.

Usage:
    OLLAMA_BASE_URL=http://localhost:11434 python full_ecot_run/scripts/02_run_llama3_1_8b.py --task all --track both

Outputs CSVs in full_ecot_run/results/llama3_1_8b/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ecot_core import ModelConfig, run_one
from _backend_ollama import make_call_fn

OLLAMA_MODEL = "llama3.1:8b"
SLUG = "llama3_1_8b"
DISPLAY = "LLaMA-3.1-8B-Instruct"


def num_predict_for(task: str) -> int:
    return {"qa": 1024, "summarization": 1536, "reasoning": 1536}[task]


def num_ctx_for(task: str) -> int:
    return {"qa": 4096, "summarization": 4096, "reasoning": 6144}[task]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa", "summarization", "reasoning", "all"], default="all")
    ap.add_argument("--track", choices=["A", "B", "both"], default="both")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    call_fn = make_call_fn(OLLAMA_MODEL, num_predict_for, num_ctx_for)
    model = ModelConfig(slug=SLUG, display_name=DISPLAY, call_fn=call_fn)
    tasks  = ("qa", "summarization", "reasoning") if args.task == "all" else (args.task,)
    tracks = ("A", "B") if args.track == "both" else (args.track,)
    for t in tasks:
        for tr in tracks:
            run_one(model, t, tr, resume=not args.no_resume)


if __name__ == "__main__":
    main()
