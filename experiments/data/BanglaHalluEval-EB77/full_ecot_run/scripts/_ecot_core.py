"""Shared logic for E-CoT (Variant C) full-benchmark evaluation.

Imports cleanly into every `02_run_<model>.py` wrapper so prompt-building,
JSON parsing, aggregation, and the task-aware verdict policy live in ONE
place. If you change them here they take effect for all 7 judges.

Policies frozen in the pilot (see pilot_50_samples/ECOT_PILOT_REPORT.md):
  - MISSING_THRESHOLD = 0.30 uniformly across tasks
  - VERDICT_SOURCE    = "agg" for QA + Reasoning, "model" for Summarization
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Repo roots ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
RUN_DIR = ROOT / "full_ecot_run"
PROMPTS_DIR = RUN_DIR / "prompts"
RESULTS_DIR = RUN_DIR / "results"

# ── Policy constants (frozen by pilot) ──────────────────────────────────────
MISSING_THRESHOLD = {"qa": 0.30, "summarization": 0.30, "reasoning": 0.30}
VERDICT_SOURCE    = {"qa": "agg", "summarization": "model", "reasoning": "agg"}

# ── Task / track / input file registry ──────────────────────────────────────
# Each entry: input CSV, prompt template, evidence + candidate field names,
# and the join-key column used for resume + downstream metric joins.
TASKS = {
    "qa": {
        "prompt_file": "ecot_qa.txt",
        "A": {
            "input":     ROOT / "BanglaHalluEval Datasets" / "banglahallueval_qa_1000.csv",
            "evidence_col":  "context",
            "candidate_col": "correct_answer",
            "id_col":        "id",
        },
        "B": {
            "input":     ROOT / "Hallucination Generated Answers" / "qa_4000.csv",
            "evidence_col":  "context",
            "candidate_col": "hallucinated_answer",
            "id_col":        "id",
        },
    },
    "summarization": {
        "prompt_file": "ecot_summarization.txt",
        "A": {
            "input":     ROOT / "Summarization" / "1000 Selected Samples" / "banglahallueval_summarization_dataset_1000.csv",
            "evidence_col":  "question",
            "candidate_col": "summary",
            "id_col":        "id",
        },
        "B": {
            "input":     ROOT / "Hallucination Generated Answers" / "summarization_3000_corrected.csv",
            "evidence_col":  "document",
            "candidate_col": "hallucinated_summary",
            "id_col":        "id",
        },
    },
    "reasoning": {
        "prompt_file": "ecot_reasoning.txt",
        "A": {
            "input":     ROOT / "Reasoning" / "1000 Selected Samples" / "somadhan_1000_main_ordered.csv",
            "evidence_col":  "answer",     # gold CoT
            "candidate_col": "answer",     # candidate equals reference on Track A
            "id_col":        "question",   # GT file has no id; join on question text
        },
        "B": {
            "input":     ROOT / "Hallucination Generated Answers" / "reasoning_1000.csv",
            "evidence_col":  "answer",            # gold CoT from SOMADHAN
            "candidate_col": "hallucinated_chain",
            "id_col":        "id",
        },
    },
}

OUTPUT_FIELDS = [
    "id_key", "raw_response", "claims_json",
    "verdict_model", "verdict_agg",
    "num_claims", "num_supported", "num_contradicted", "num_missing",
    "divergence", "parse_error",
    "verdict_source", "is_hallucinated",
]


# ── Prompt builders ─────────────────────────────────────────────────────────
def load_prompt(task: str) -> str:
    return (PROMPTS_DIR / TASKS[task]["prompt_file"]).read_text(encoding="utf-8")


def build_prompt(task: str, track: str, row: pd.Series, template: str) -> str:
    cfg = TASKS[task][track]
    evidence  = str(row[cfg["evidence_col"]]).strip()
    candidate = str(row[cfg["candidate_col"]]).strip()
    if task == "qa":
        return template.format(
            context=evidence,
            question=str(row["question"]).strip(),
            candidate=candidate,
        )
    if task == "summarization":
        return template.format(source=evidence, candidate=candidate)
    if task == "reasoning":
        return template.format(
            question=str(row["question"]).strip(),
            reference=evidence,
            candidate=candidate,
        )
    raise ValueError(f"Unknown task: {task}")


# ── JSON parsing + aggregation ──────────────────────────────────────────────
def parse_json(raw: str) -> tuple[Optional[dict], Optional[str]]:
    if not raw:
        return None, "empty"
    # Some open-weight judges wrap output in ```json fences; strip them.
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as exc:
        # Fallback: try to grab the first {...} substring
        start = s.find("{")
        end   = s.rfind("}")
        if 0 <= start < end:
            try:
                obj = json.loads(s[start:end + 1])
            except json.JSONDecodeError as exc2:
                return None, f"json_decode_fallback: {exc2}"
        else:
            return None, f"json_decode: {exc}"
    if not isinstance(obj, dict):
        return None, "not_dict"
    return obj, None


def _sanitize_claims(raw_claims) -> list:
    """Coerce whatever the judge produced under `claims` into a list of dicts.

    Small open-weight judges occasionally emit `"claims": ["some text", ...]`
    (list of strings) or `"claims": "some text"` (bare string) instead of
    the expected list of dicts. We keep only dict entries so downstream
    aggregation never blows up on `.get()` on a str.
    """
    if isinstance(raw_claims, dict):
        raw_claims = [raw_claims]
    if not isinstance(raw_claims, list):
        return []
    return [c for c in raw_claims if isinstance(c, dict)]


def aggregate_verdict(claims: list, missing_threshold: float = 0.30) -> str:
    claims = _sanitize_claims(claims)
    if not claims:
        return "unknown"
    statuses = [str(c.get("status", "")).lower().strip() for c in claims]
    if any(s == "contradicted" for s in statuses):
        return "yes"
    n_missing = sum(1 for s in statuses if s == "missing")
    if (n_missing / len(statuses)) > missing_threshold:
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


def make_output_row(task: str, raw: Optional[str], parsed: Optional[dict],
                    parse_err: Optional[str], id_key: str) -> dict:
    if parsed is None:
        claims = []
        verdict_model = "unknown"
        verdict_agg = "unknown"
    else:
        claims = _sanitize_claims(parsed.get("claims"))
        verdict_model = normalize_verdict(parsed.get("verdict"))
        verdict_agg   = aggregate_verdict(claims, MISSING_THRESHOLD[task])
    statuses = [str(c.get("status", "")).lower().strip() for c in claims]
    n_sup = sum(1 for s in statuses if s == "supported")
    n_con = sum(1 for s in statuses if s == "contradicted")
    n_mis = sum(1 for s in statuses if s == "missing")
    source = VERDICT_SOURCE[task]
    is_hallu = verdict_model if source == "model" else verdict_agg
    divergence = int(verdict_model != verdict_agg and verdict_model != "unknown")
    return {
        "id_key":           id_key,
        "raw_response":     raw or "",
        "claims_json":      json.dumps(claims, ensure_ascii=False),
        "verdict_model":    verdict_model,
        "verdict_agg":      verdict_agg,
        "num_claims":       len(claims),
        "num_supported":    n_sup,
        "num_contradicted": n_con,
        "num_missing":      n_mis,
        "divergence":       divergence,
        "parse_error":      parse_err or "",
        "verdict_source":   source,
        "is_hallucinated":  is_hallu,
    }


# ── Resume + streaming append-write ────────────────────────────────────────
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


def open_writer(out_path: Path, original_fields: list[str]):
    """Returns (file handle, csv.DictWriter, fieldnames). Writes header if needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fresh = (not out_path.exists()) or out_path.stat().st_size == 0
    fieldnames = list(original_fields) + [f for f in OUTPUT_FIELDS if f not in original_fields]
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if fresh:
        w.writeheader()
        f.flush()
    return f, w, fieldnames


# ── The driver loop (used by every per-model wrapper) ──────────────────────
@dataclass
class ModelConfig:
    slug: str                              # filesystem slug
    display_name: str                      # human label
    call_fn: Callable[[str, str], str]     # call_fn(prompt, task) -> raw response text
    task_token_budget: dict = field(default_factory=dict)
    # Optional: per-task max output tokens, for backends that accept it.


def out_path_for(slug: str, task: str, track: str) -> Path:
    suffix = {"A": "gt", "B": "hallu"}[track]
    return RESULTS_DIR / slug / f"{task}_{suffix}_ecot.csv"


def run_one(model: ModelConfig, task: str, track: str,
            resume: bool = True, max_retries: int = 3,
            checkpoint_every: int = 1) -> None:
    cfg = TASKS[task][track]
    template = load_prompt(task)
    df_in = pd.read_csv(cfg["input"], on_bad_lines="skip")
    id_col = cfg["id_col"]
    df_in[id_col] = df_in[id_col].astype(str).str.strip()

    out_path = out_path_for(model.slug, task, track)
    done = load_done_keys(out_path) if resume else set()
    fh, writer, fieldnames = open_writer(out_path, list(df_in.columns))

    total = len(df_in)
    todo  = total - len(done)
    print(f"[{model.slug}] task={task} track={track}  total={total}  done={len(done)}  todo={todo}")
    if todo == 0:
        fh.close()
        return

    written = 0
    t0 = time.time()
    try:
        for i, row in df_in.iterrows():
            id_key = str(row[id_col])
            if id_key in done:
                continue

            prompt = build_prompt(task, track, row, template)

            raw, parsed, err = None, None, None
            for attempt in range(max_retries):
                raw = model.call_fn(prompt, task)
                parsed, err = parse_json(raw or "")
                if parsed is not None:
                    break
                time.sleep(0.4 + 0.5 * attempt)

            # Defensive: any unexpected shape in the parsed JSON should NOT kill
            # the whole run. Fall back to an "unknown" row so --resume moves on.
            try:
                row_out = {k: row[k] for k in df_in.columns}
                row_out.update(make_output_row(task, raw, parsed, err, id_key))
            except Exception as exc:
                row_out = {k: row[k] for k in df_in.columns}
                row_out.update(make_output_row(task, raw, None,
                                               f"post_parse_error: {exc}", id_key))

            writer.writerow(row_out)
            fh.flush()
            try:
                os.fsync(fh.fileno())        # persist to disk, not just OS buffer
            except (AttributeError, OSError):
                pass
            written += 1

            # Progress log (still batched so stdout doesn't get flooded).
            if written % 20 == 0 or written == todo:
                elapsed = time.time() - t0
                rate = written / max(elapsed, 1e-6)
                eta  = (todo - written) / max(rate, 1e-6)
                print(f"  [{model.slug}] {task}/{track}  {written}/{todo}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f}min  "
                      f"last: verdict_model={row_out['verdict_model']}  agg={row_out['verdict_agg']}  "
                      f"sup/con/mis={row_out['num_supported']}/{row_out['num_contradicted']}/{row_out['num_missing']}")
    finally:
        fh.flush()
        fh.close()
    print(f"  [{model.slug}] {task}/{track} -> {out_path.relative_to(ROOT)}")


def run_all(model: ModelConfig,
            tasks: Iterable[str] = ("qa", "summarization", "reasoning"),
            tracks: Iterable[str] = ("A", "B"),
            resume: bool = True) -> None:
    for task in tasks:
        for tr in tracks:
            run_one(model, task, tr, resume=resume)
