#!/usr/bin/env python3
"""E-CoT Variant C evaluator for the pilot.

For a given task (qa | summarization | reasoning) read the 50-sample pilot CSV,
build the evidence-augmented Citation-Forcing prompt, query GPT-4.1 mini in
JSON-object mode, and save the per-row verdict + claims trace.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("Missing dependency: openai. `pip install openai`.") from exc

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


ROOT = Path(__file__).resolve().parent.parent.parent
PILOT_DIR = ROOT / "pilot_50_samples"
PROMPTS_DIR = PILOT_DIR / "prompts"
DATA_DIR = PILOT_DIR / "data"
RESULTS_DIR = PILOT_DIR / "results"

MODEL = "gpt-4.1-mini"
MAX_OUTPUT_TOKENS = 2000
TEMPERATURE = 0
MAX_RETRIES = 3

# Per-task aggregation threshold for the "missing" rule.
# Summarization paraphrases heavily, so missing claims are expected; use 0.5.
# QA / Reasoning are extractive/precise, so 0.3 is the right cutoff.
MISSING_THRESHOLD = {
    "qa": 0.30,
    "summarization": 0.30,
    "reasoning": 0.30,
}

# Task-aware verdict source: model self-report or deterministic aggregation.
# Reasoning + QA: model under-flags contradictions, so aggregation wins.
# Summarization: model is well-calibrated; aggregation over-flags via "missing".
VERDICT_SOURCE = {
    "qa":            "agg",
    "summarization": "model",
    "reasoning":     "agg",
}

TASK_CFG = {
    # ── Track A (ground-truth correct candidates; expected verdict = "no") ──
    "A": {
        "qa": {
            "data":   "qa_gt_50.csv",
            "out":    "qa_gt_50_ecot.csv",
            "prompt": "ecot_qa.txt",
        },
        "summarization": {
            "data":   "summarization_gt_50.csv",
            "out":    "summarization_gt_50_ecot.csv",
            "prompt": "ecot_summarization.txt",
        },
        "reasoning": {
            "data":   "reasoning_gt_50.csv",
            "out":    "reasoning_gt_50_ecot.csv",
            "prompt": "ecot_reasoning.txt",
        },
    },
    # ── Track B (hallucinated candidates; expected verdict = "yes") ────────
    "B": {
        "qa": {
            "data":   "qa_hallu_50.csv",
            "out":    "qa_hallu_50_ecot.csv",
            "prompt": "ecot_qa.txt",
        },
        "summarization": {
            "data":   "summarization_hallu_50.csv",
            "out":    "summarization_hallu_50_ecot.csv",
            "prompt": "ecot_summarization.txt",
        },
        "reasoning": {
            "data":   "reasoning_hallu_50.csv",
            "out":    "reasoning_hallu_50_ecot.csv",
            "prompt": "ecot_reasoning.txt",
        },
    },
}


def build_prompt(task: str, track: str, row: pd.Series, template: str) -> str:
    if task == "qa":
        candidate = row["correct_answer"] if track == "A" else row["hallucinated_answer"]
        return template.format(
            context=str(row["context"]).strip(),
            question=str(row["question"]).strip(),
            candidate=str(candidate).strip(),
        )
    if task == "summarization":
        if track == "A":
            source = row["question"]
            candidate = row["summary"]
        else:
            source = row["document"]
            candidate = row["hallucinated_summary"]
        return template.format(
            source=str(source).strip(),
            candidate=str(candidate).strip(),
        )
    if task == "reasoning":
        if track == "A":
            reference = str(row["answer"]).strip()
            candidate = reference  # GT: candidate equals reference
        else:
            reference = str(row["answer"]).strip()           # gold CoT
            candidate = str(row["hallucinated_chain"]).strip()  # hallucinated CoT
        return template.format(
            question=str(row["question"]).strip(),
            reference=reference,
            candidate=candidate,
        )
    raise ValueError(f"Unknown task: {task}")


def call_gpt(client: OpenAI, prompt: str) -> Optional[str]:
    """Return raw text response or None on hard failure."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        print(f"  ! OpenAI error: {exc}", file=sys.stderr)
        return None


def parse_json(raw: str) -> Tuple[Optional[dict], Optional[str]]:
    """Return (parsed_dict, error_message)."""
    if not raw:
        return None, "empty"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"json_decode: {exc}"
    if not isinstance(obj, dict):
        return None, "not_dict"
    return obj, None


def aggregate_verdict(claims: list, missing_threshold: float = 0.30) -> str:
    if not claims:
        return "unknown"
    statuses = [str(c.get("status", "")).lower().strip() for c in claims]
    if any(s == "contradicted" for s in statuses):
        return "yes"
    n_missing = sum(1 for s in statuses if s == "missing")
    if len(statuses) > 0 and (n_missing / len(statuses)) > missing_threshold:
        return "yes"
    return "no"


def normalize_verdict(v) -> str:
    if v is None:
        return "unknown"
    s = str(v).strip().lower()
    if s.startswith("y"):
        return "yes"
    if s.startswith("n"):
        return "no"
    return "unknown"


def run_task(task: str, track: str, client: OpenAI) -> None:
    cfg = TASK_CFG[track][task]
    template = (PROMPTS_DIR / cfg["prompt"]).read_text(encoding="utf-8")
    df_in = pd.read_csv(DATA_DIR / cfg["data"])
    out_path = RESULTS_DIR / cfg["out"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for i, row in df_in.iterrows():
        prompt = build_prompt(task, track, row, template)

        parsed = None
        raw = None
        err = None
        for attempt in range(MAX_RETRIES):
            raw = call_gpt(client, prompt)
            parsed, err = parse_json(raw or "")
            if parsed is not None:
                break
            time.sleep(0.5 + attempt * 0.5)

        if parsed is None:
            verdict_model = "unknown"
            verdict_agg = "unknown"
            claims = []
        else:
            claims = parsed.get("claims") or []
            verdict_model = normalize_verdict(parsed.get("verdict"))
            verdict_agg = aggregate_verdict(claims, missing_threshold=MISSING_THRESHOLD[task])

        n_claims = len(claims)
        statuses = [str(c.get("status", "")).lower().strip() for c in claims]
        n_sup = sum(1 for s in statuses if s == "supported")
        n_con = sum(1 for s in statuses if s == "contradicted")
        n_mis = sum(1 for s in statuses if s == "missing")
        divergence = (verdict_model != verdict_agg) and verdict_model != "unknown"

        chosen_source = VERDICT_SOURCE[task]
        is_hallu = verdict_model if chosen_source == "model" else verdict_agg

        out_row = {k: row[k] for k in df_in.columns}
        out_row.update({
            "raw_json": raw or "",
            "claims_json": json.dumps(claims, ensure_ascii=False),
            "verdict_model": verdict_model,
            "verdict_agg": verdict_agg,
            "num_claims": n_claims,
            "num_supported": n_sup,
            "num_contradicted": n_con,
            "num_missing": n_mis,
            "divergence": int(divergence),
            "parse_error": err or "",
            "verdict_source": chosen_source,
            "is_hallucinated": is_hallu,   # task-aware: agg for QA+Reas, model for Summ
        })
        rows_out.append(out_row)

        flag = " *DIV*" if divergence else ""
        print(f"  [{task}/{track}] {i+1:2d}/{len(df_in)} -> verdict={verdict_model}/{verdict_agg} claims={n_claims} sup={n_sup} con={n_con} mis={n_mis}{flag}")

    fieldnames = list(rows_out[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"  -> wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["qa", "summarization", "reasoning", "all"], default="all")
    ap.add_argument("--track", choices=["A", "B", "both"], default="A",
                    help="A = ground-truth pilot, B = hallucinated pilot, both = run both")
    args = ap.parse_args()

    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not found in environment / .env.")
    client = OpenAI(api_key=api_key)

    tasks = ["qa", "summarization", "reasoning"] if args.task == "all" else [args.task]
    tracks = ["A", "B"] if args.track == "both" else [args.track]
    for tr in tracks:
        for t in tasks:
            print(f"\n== Running E-CoT (Variant C) on task={t} track={tr} ==")
            run_task(t, tr, client)


if __name__ == "__main__":
    main()
