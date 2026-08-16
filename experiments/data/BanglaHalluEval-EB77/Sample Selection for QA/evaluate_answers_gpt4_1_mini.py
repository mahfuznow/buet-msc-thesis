"""Evaluate deepseek/gemma/qwen zero-shot QA answers using GPT-4.1-mini as judge.

Mirrors evaluate_answers.py exactly (same merge logic, same scoring prompt),
but:
  - Uses the 1000-item ground-truth set (BanglaHalluEval Datasets/banglahallueval_qa_1000.csv) instead of
    Datasets/tydiqa_goldp_bengali.csv.
  - Judges a reproducible 500-item random sample (seed=42) of the
    deepseek/gemma/qwen intersection, instead of the full set.
  - Calls OpenAI's gpt-4.1-mini via the API instead of a local qwen2.5:32b
    Ollama judge, to avoid a same-family (Qwen-judging-Qwen) self-preference
    bias in the comparison.

API key is read from the .env file in the project root (OPENAI_API_KEY).

Resumable: every row is flushed to disk immediately after being scored. If
the script is stopped and re-run, it skips ids already present in the output
file, so no work is duplicated and no API budget is wasted on redo.
"""

import csv
import json
import os
import random
import time
from pathlib import Path

from tqdm import tqdm

try:
    from openai import OpenAI, RateLimitError, APIError, APIConnectionError
except ImportError as exc:
    raise SystemExit("Missing dependency: openai.  Run: pip install openai") from exc

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gpt-4.1-mini"
SAMPLE_SIZE = 500
SEED = 42

GT_FILE = ROOT / "QA" / "qa_gt_1000.csv"
DEEPSEEK_FILE = ROOT / "Sample Selection for QA" / "Results" / "deepseek_answers_bengali.csv"
GEMMA_FILE = ROOT / "Sample Selection for QA" / "Results" / "gemma_answers_bengali.csv"
QWEN_FILE = ROOT / "Sample Selection for QA" / "Results" / "qwen_answers_bengali.csv"

UNSCORED_OUT = ROOT / "Sample Selection for QA" / "Results" / "combined_evaluation_unscored_500_gpt4_1_mini.csv"
SCORED_OUT = ROOT / "Sample Selection for QA" / "Results" / "combined_evaluation_scored_500_gpt4_1_mini.csv"

FIELDNAMES = [
    "id", "question", "deepseek_answer", "gemma_answer", "qwen_answer", "correct_answer",
    "deepseek_score", "gemma_score", "qwen_score",
]


def load_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def prepare_combined_dataset():
    print("Loading data...")
    gt = load_csv(GT_FILE)
    deepseek_res = load_csv(DEEPSEEK_FILE)
    gemma_res = load_csv(GEMMA_FILE)
    qwen_res = load_csv(QWEN_FILE)

    d_map = {r["serial_id"]: r["model_answer"] for r in deepseek_res}
    g_map = {r["serial_id"]: r["model_answer"] for r in gemma_res}
    q_map = {r["serial_id"]: r["model_answer"] for r in qwen_res}

    merged = []
    seen_ids = set()
    for r in gt:
        rid = r["id"]
        if rid in seen_ids:
            continue  # qa_gt_1000.csv has 1000 rows but only 997 unique ids
        if rid in d_map and rid in g_map and rid in q_map:
            seen_ids.add(rid)
            merged.append({
                "id": rid,
                "question": r["question"],
                "deepseek_answer": d_map[rid],
                "gemma_answer": g_map[rid],
                "qwen_answer": q_map[rid],
                "correct_answer": r["correct_answer"],
                "deepseek_score": 0,
                "gemma_score": 0,
                "qwen_score": 0,
            })

    print(f"  Ground truth (selected 1000): {len(gt)} rows")
    print(f"  Intersection with all three model-answer files: {len(merged)} rows")

    sample = random.Random(SEED).sample(merged, SAMPLE_SIZE)
    print(f"  Sampled {len(sample)} rows (seed={SEED}) for evaluation.")
    return sample


def build_prompt(row: dict) -> str:
    # Identical prompt to evaluate_answers.py's evaluate_with_qwen().
    return f"""You are an expert evaluator for Bengali Question Answering tasks.
Your task is to evaluate and compare three model answers against a correct ground truth answer.

Question: {row['question']}
Correct Answer: {row['correct_answer']}

Model Answers:
DeepSeek Answer: {row['deepseek_answer']}
Gemma Answer: {row['gemma_answer']}
Qwen Answer: {row['qwen_answer']}

Please score each model answer as 1 if it is correct, or 0 if it is incorrect.
Keep in mind these edge cases:
- Answers might have spelling mistakes or typos but are essentially correct.
- An answer may be provided in English instead of Bengali but conveys the exact same meaning.
- An answer might be partial but captures the core correct information.
If any of the above conditions are met and the core meaning matches the correct answer, score it as 1.

Respond ONLY with a JSON object containing the scores for each model, like this:
{{"deepseek_score": 1, "gemma_score": 0, "qwen_score": 1}}
Do not explain your reasoning or output anything else.
"""


def call_openai(prompt: str, client: OpenAI, max_tokens: int = 200) -> str:
    backoff = 5
    attempts = 0
    while True:
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=max_tokens,
                temperature=0,
            )
            return (resp.output_text or "").strip()
        except RateLimitError as e:
            wait = backoff * (2 ** min(attempts, 4))
            print(f"  [RATE LIMIT] {e} - sleeping {wait}s ...")
            time.sleep(wait)
            attempts += 1
        except (APIConnectionError, APIError) as e:
            attempts += 1
            wait = backoff * min(attempts, 6)
            print(f"  [API ERROR] {e} (attempt {attempts}) - retrying in {wait}s ...")
            if attempts >= 10:
                print("  [GIVE UP] Too many failures, returning empty string.")
                return ""
            time.sleep(wait)
        except Exception as e:
            attempts += 1
            print(f"  [ERROR] Unexpected: {e} (attempt {attempts})")
            if attempts >= 5:
                return ""
            time.sleep(backoff)


def parse_scores(raw: str) -> dict:
    """Extract the scores JSON object from the model response (robust to stray text)."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    import re
    m = re.search(r"\{[^{}]*deepseek_score[^{}]*\}", raw, re.IGNORECASE | re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def evaluate_with_gpt(rows: list, client: OpenAI) -> list:
    # Resume: skip ids already scored in the output file.
    done_ids = set()
    if SCORED_OUT.exists():
        for r in load_csv(SCORED_OUT):
            if r.get("id"):
                done_ids.add(r["id"])
        print(f"  Resuming — {len(done_ids)} rows already scored.")

    write_header = not SCORED_OUT.exists() or SCORED_OUT.stat().st_size == 0
    SCORED_OUT.parent.mkdir(parents=True, exist_ok=True)

    with SCORED_OUT.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            f.flush()

        for row in tqdm(rows, total=len(rows)):
            if row["id"] in done_ids:
                continue
            prompt = build_prompt(row)
            raw = call_openai(prompt, client)
            scores = parse_scores(raw)

            out_row = dict(row)
            out_row["deepseek_score"] = scores.get("deepseek_score", 0)
            out_row["gemma_score"] = scores.get("gemma_score", 0)
            out_row["qwen_score"] = scores.get("qwen_score", 0)

            writer.writerow(out_row)
            f.flush()

    return load_csv(SCORED_OUT)


def main():
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Add it to the .env file in the project root or export it."
        )
    client = OpenAI(api_key=api_key, timeout=30.0)

    combined = prepare_combined_dataset()

    UNSCORED_OUT.parent.mkdir(parents=True, exist_ok=True)
    with UNSCORED_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(combined)
    print(f"Prepared dataset with {len(combined)} samples -> {UNSCORED_OUT}")

    print(f"Evaluating answers using {MODEL} (judge)...")
    evaluate_with_gpt(combined, client)
    print(f"Evaluation complete! Results saved to {SCORED_OUT}")


if __name__ == "__main__":
    main()
