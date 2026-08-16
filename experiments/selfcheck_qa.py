#!/usr/bin/env python3
"""
SelfCheckGPT-style consistency detection on BanglaHalluEval QA data.

Instead of asking a judge model "is this hallucinated?", this script:
  1. Takes a question from the dataset
  2. Asks a local Ollama model to answer it N times (high temperature = stochastic)
  3. Measures how semantically consistent those N answers are with each other
  4. Low consistency = model is uncertain = likely hallucinating (SelfCheckGPT logic)

Produces a CSV with the same structure as BanglaHalluEval's labeling scripts,
adding three new columns:
  - consistency_score  : mean cosine similarity across the N samples (0.0 to 1.0)
  - selfcheck_label    : yes / no  (yes = hallucinated, based on threshold)
  - samples_json       : the N generated samples (for inspection)

This allows direct BHS comparison with BanglaHalluEval's judge-model results.

Usage:

  # Track B — hallucinated candidates (4000 rows)
  python selfcheck_qa.py \\
    --input  "data/BanglaHalluEval-EB77/Hallucination Generated Answers/qa_4000.csv" \\
    --output "results/selfcheck_qa_4000_qwen7b.csv" \\
    --answer-col hallucinated_answer \\
    --model qwen2.5:7b \\
    --n-samples 5 \\
    --threshold 0.75

  # Track A — ground truth (1000 rows)
  python selfcheck_qa.py \\
    --input  "data/BanglaHalluEval-EB77/BanglaHalluEval Datasets/banglahallueval_qa_1000.csv" \\
    --output "results/selfcheck_qa_1000_gt_qwen7b.csv" \\
    --answer-col correct_answer \\
    --model qwen2.5:7b \\
    --n-samples 5 \\
    --threshold 0.75

  # Quick pilot on first 50 rows only
  python selfcheck_qa.py \\
    --input  "data/BanglaHalluEval-EB77/Hallucination Generated Answers/qa_4000.csv" \\
    --output "results/selfcheck_pilot_50.csv" \\
    --answer-col hallucinated_answer \\
    --start 0 --end 50 \\
    --model qwen2.5:7b
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ── Model loaded once globally ──────────────────────────────────────────────
print("Loading multilingual sentence embedding model...")
EMBEDDER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Embedding model ready.\n")

OLLAMA_URL = "http://localhost:11434/api/generate"

GENERATION_PROMPT = (
    "You are a helpful assistant. Answer the following question in Bengali.\n"
    "Be concise — one or two sentences at most.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ── Ollama helpers ───────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str, temperature: float = 1.0,
                max_tokens: int = 128, timeout: int = 60) -> Optional[str]:
    """Call local Ollama HTTP API. Returns response text or None on failure."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        print(f"  [ollama error] {e}")
        return None


def generate_samples(question: str, model: str, n: int,
                     temperature: float = 1.0) -> list[str]:
    """Generate N stochastic answers for a given question."""
    samples = []
    prompt = GENERATION_PROMPT.format(question=question)
    for i in range(n):
        for attempt in range(3):
            resp = call_ollama(prompt, model=model, temperature=temperature)
            if resp:
                samples.append(resp)
                break
            time.sleep(0.5 + attempt * 0.5)
        else:
            samples.append("")   # empty string if all retries fail
        time.sleep(0.1)
    return samples


# ── Consistency scoring ──────────────────────────────────────────────────────

def consistency_score(samples: list[str]) -> float:
    """
    Mean pairwise cosine similarity across all N samples.
    High score = samples agree = model is consistent = probably not hallucinating.
    Low score  = samples diverge = model is uncertain = likely hallucinating.
    """
    valid = [s for s in samples if s.strip()]
    if len(valid) < 2:
        return 0.0   # can't compute — treat as uncertain
    embs = EMBEDDER.encode(valid, normalize_embeddings=True)
    # Upper triangle of cosine similarity matrix (no self-similarity)
    sims = embs @ embs.T
    n = len(valid)
    pairs = [(sims[i][j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(pairs))


# ── BHS computation ──────────────────────────────────────────────────────────

def compute_bhs(track_a_labels: list[str], track_b_labels: list[str]) -> dict:
    """
    Compute BenHalluScore exactly as the paper does.

    Track A (ground truth): correct answers → expected label = 'no'
      Track A error = labeling a correct answer as 'yes' (false positive)

    Track B (hallucinated): hallucinated answers → expected label = 'yes'
      Track B error = labeling a hallucination as 'no' (missed detection)

    BHS = 0.5 × (Track A error rate + Track B error rate) × 100
    Lower BHS = better calibration.
    """
    if track_a_labels:
        track_a_errors = sum(1 for l in track_a_labels if l == "yes")
        track_a_rate = track_a_errors / len(track_a_labels)
    else:
        track_a_rate = None

    if track_b_labels:
        track_b_errors = sum(1 for l in track_b_labels if l == "no")
        track_b_rate = track_b_errors / len(track_b_labels)
    else:
        track_b_rate = None

    if track_a_rate is not None and track_b_rate is not None:
        bhs = 0.5 * (track_a_rate + track_b_rate) * 100
    else:
        bhs = None

    return {
        "track_a_n": len(track_a_labels),
        "track_a_error_rate": round(track_a_rate * 100, 2) if track_a_rate is not None else None,
        "track_b_n": len(track_b_labels),
        "track_b_error_rate": round(track_b_rate * 100, 2) if track_b_rate is not None else None,
        "bhs": round(bhs, 2) if bhs is not None else None,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="SelfCheckGPT-style consistency detection on BanglaHalluEval QA data."
    )
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument(
        "--answer-col", default="hallucinated_answer",
        help="Column containing the answer to score. "
             "Use 'hallucinated_answer' for Track B, 'correct_answer' for Track A."
    )
    p.add_argument("--model", default="qwen2.5:7b", help="Ollama model name")
    p.add_argument("--n-samples", type=int, default=5,
                   help="Number of stochastic samples to generate per question")
    p.add_argument("--sample-temperature", type=float, default=1.0,
                   help="Generation temperature for samples (default: 1.0)")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="Consistency score below this → label as 'yes' (hallucinated). "
                        "Default: 0.75. Tune based on pilot results.")
    p.add_argument("--start", type=int, default=0, help="Start row (0-indexed)")
    p.add_argument("--end", type=int, default=None, help="End row (exclusive)")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows that already have a consistency_score in the output file")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load input
    with inp.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        original_fields = reader.fieldnames or []

    end = args.end if args.end is not None else len(rows)
    selected = rows[args.start:end]

    # Load already-done rows if resuming
    done_ids: set[str] = set()
    if args.resume and out.exists():
        with out.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("consistency_score", "") not in ("", "error"):
                    done_ids.add(r.get("id", ""))
        print(f"Resuming: {len(done_ids)} rows already completed.\n")

    # Output fieldnames — original + 3 new columns
    new_fields = ["consistency_score", "selfcheck_label", "samples_json"]
    out_fields = original_fields + [f for f in new_fields if f not in original_fields]

    # Open output (append if resuming, write if fresh)
    mode = "a" if args.resume and out.exists() else "w"
    with out.open(mode, newline="", encoding="utf-8") as ofh:
        writer = csv.DictWriter(ofh, fieldnames=out_fields)
        if mode == "w":
            writer.writeheader()

        all_labels = []

        for i, row in enumerate(selected, start=args.start):
            sample_id = row.get("id", str(i))

            if args.resume and sample_id in done_ids:
                print(f"{i}: {sample_id} — skipped (already done)")
                continue

            question   = row.get("question", "")
            answer_col = args.answer_col
            answer     = row.get(answer_col, "")

            if not question.strip():
                print(f"{i}: {sample_id} — skipped (empty question)")
                continue

            print(f"{i}: {sample_id}", end="", flush=True)

            # Generate N samples
            samples = generate_samples(
                question, model=args.model,
                n=args.n_samples, temperature=args.sample_temperature
            )

            # Compute consistency
            score = consistency_score(samples)

            # Apply threshold → label
            label = "yes" if score < args.threshold else "no"
            all_labels.append(label)

            print(f"  score={score:.4f}  label={label}")

            # Write row
            out_row = dict(row)
            out_row["consistency_score"] = round(score, 6)
            out_row["selfcheck_label"]   = label
            out_row["samples_json"]      = json.dumps(samples, ensure_ascii=False)
            writer.writerow(out_row)
            ofh.flush()   # write immediately so --resume works on crash

    print(f"\nDone. Wrote {len(all_labels)} rows to {out}")

    # Print summary
    if all_labels:
        n_yes = all_labels.count("yes")
        n_no  = all_labels.count("no")
        print(f"\nSummary (threshold={args.threshold}):")
        print(f"  Labeled 'yes' (hallucinated) : {n_yes} / {len(all_labels)} "
              f"({100*n_yes/len(all_labels):.1f}%)")
        print(f"  Labeled 'no'  (not halluc.)  : {n_no} / {len(all_labels)} "
              f"({100*n_no/len(all_labels):.1f}%)")
        print(f"\nTo compute BHS, run both Track A and Track B, then:")
        print(f"  python compute_bhs.py \\")
        print(f"    --track-a results/selfcheck_qa_1000_gt_qwen7b.csv \\")
        print(f"    --track-b results/selfcheck_qa_4000_qwen7b.csv")


if __name__ == "__main__":
    main()
