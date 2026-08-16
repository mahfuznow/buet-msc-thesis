#!/usr/bin/env python3
"""Unified codemix CoT evaluator — dispatches per-model to Ollama, HF, or OpenAI.

Runs both tracks (gt from codemix_1000, hallu from codemix_4000) on the
deterministic 10% subsets produced by sample_codemix_10pct.py.

Outputs (resumable, incremental writes):
    T Sampled Evaluations/T_codemix_cot/codemix_cot_gt_<slug>.csv
    T Sampled Evaluations/T_codemix_cot/codemix_cot_hallu_<slug>.csv

Usage examples:
    python scripts/eval_codemix_cot.py --model llama3_1_8b
    python scripts/eval_codemix_cot.py --model banglallama_13b
    python scripts/eval_codemix_cot.py --all-ollama    # loops the 5 Ollama models
    python scripts/eval_codemix_cot.py --all-hf        # loops the 3 HF models
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "Sampling" / "10pct_codemix"
OUT_DIR    = ROOT / "T Sampled Evaluations" / "T_codemix_cot"

MODELS: Dict[str, dict] = {
    # Ollama-served (5)
    "llama3_1_8b":     {"backend": "ollama", "id": "llama3.1:8b",             "num_ctx": 4096, "num_predict": 800},
    "mistral_nemo":    {"backend": "ollama", "id": "mistral-nemo:latest",     "num_ctx": 4096, "num_predict": 800},
    "deepseek_r1_14b": {"backend": "ollama", "id": "deepseek-r1:14b",         "num_ctx": 6144, "num_predict": 1400},
    "gemma2_27b":      {"backend": "ollama", "id": "gemma2:27b",              "num_ctx": 4096, "num_predict": 800},
    "qwen2_5_32b":     {"backend": "ollama", "id": "qwen2.5:32b-instruct",    "num_ctx": 4096, "num_predict": 800},
    # HuggingFace (3)
    "tigerllm_9b":     {"backend": "hf",     "id": "md-nishat-008/TigerLLM-9B-it",              "quant": False, "batch": 4},
    "titullm_3b":      {"backend": "hf",     "id": "hishab/titulm-llama-3.2-3b-v1.1",           "quant": False, "batch": 8},
    "banglallama_13b": {"backend": "hf",     "id": "BanglaLLM/bangla-llama-13b-instruct-v0.1",  "quant": True,  "batch": 4},
    # OpenAI (1)
    "gpt4_1_mini":     {"backend": "openai", "id": "gpt-4.1-mini"},
}

OLLAMA_MODELS = [k for k, v in MODELS.items() if v["backend"] == "ollama"]
HF_MODELS     = [k for k, v in MODELS.items() if v["backend"] == "hf"]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt + parsing
# ─────────────────────────────────────────────────────────────────────────────

def _s(x) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


def build_cot_prompt(row: dict, answer_field: str) -> str:
    return (
        "You are an evaluator. Decide whether the model answer is hallucinated "
        "relative to the code-mixed (Bangla-English) context and question.\n\n"
        f"Context: {_s(row.get('codemix_context'))}\n"
        f"Question: {_s(row.get('codemix_question'))}\n"
        f"Model Answer: {_s(row.get(answer_field))}\n\n"
        "Analyze step by step:\n"
        "Step 1: Identify each factual claim made by the Model Answer.\n"
        "Step 2: For each claim, check whether it is supported by the Context.\n"
        "Step 3: Based on steps 1-2, decide the final verdict.\n\n"
        "Write your reasoning, then on the LAST line write exactly one word: yes or no"
    )


_YES_RE = re.compile(r"\byes\b", re.I)
_NO_RE  = re.compile(r"\bno\b",  re.I)


def parse_yesno(raw: str) -> str:
    if not raw:
        return "unknown"
    text = raw.strip()
    # last-line first
    for ln in reversed([x.strip(" .,!?:;\"'`*[](){}<>") for x in text.splitlines() if x.strip()]):
        low = ln.lower()
        if low in ("yes", "y"): return "yes"
        if low in ("no",  "n"): return "no"
    # JSON body: {"is_hallucinated": "yes"}
    try:
        obj = json.loads(text)
        v = str(obj.get("is_hallucinated") or obj.get("answer") or "").strip().lower()
        if v.startswith("y"): return "yes"
        if v.startswith("n"): return "no"
    except Exception:
        pass
    m = re.search(r'"(?:is_hallucinated|answer|verdict)"\s*:\s*"(yes|no)"', text, re.I)
    if m: return m.group(1).lower()
    # bengali fallback
    if "হ্যাঁ" in text or "হ্যা" in text: return "yes"
    if "না" in text and "নয়" not in text: return "no"
    # last occurrence wins
    y = list(_YES_RE.finditer(text)); n = list(_NO_RE.finditer(text))
    if y and (not n or y[-1].start() > n[-1].start()): return "yes"
    if n: return "no"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# CSV I/O
# ─────────────────────────────────────────────────────────────────────────────

def row_id(row: dict) -> str:
    for k in ("id", "question_id", "source_id"):
        v = row.get(k)
        if v not in (None, ""):
            return str(v).strip()
    return ""


def open_writer(out_path: Path, fieldnames: List[str]) -> Tuple[csv.DictWriter, set]:
    for extra in ("raw_response", "is_hallucinated"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("is_hallucinated") or "").strip().lower() in ("yes", "no"):
                    rid = row_id(r)
                    if rid: done.add(rid)
        f_out = open(out_path, "a", encoding="utf-8", newline="")
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
    else:
        f_out = open(out_path, "w", encoding="utf-8", newline="")
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
    w._f = f_out
    return w, done


# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────

def make_ollama_call(cfg: dict) -> Callable[[str], str]:
    import requests
    url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
    def call(prompt: str) -> str:
        payload = {
            "model": cfg["id"], "prompt": prompt, "stream": False,
            "options": {"num_predict": cfg["num_predict"], "num_ctx": cfg["num_ctx"], "temperature": 0.0},
        }
        waited = False
        while True:
            try:
                r = requests.post(url, json=payload, timeout=900)
                r.raise_for_status()
                if waited: print("  [ollama] reachable again")
                return (r.json().get("response") or "").strip()
            except requests.exceptions.ConnectionError:
                if not waited:
                    print(f"  [!] ollama unreachable, backing off"); waited = True
                time.sleep(15)
            except requests.exceptions.RequestException as e:
                print(f"  [!] ollama request error: {e}", file=sys.stderr)
                return ""
    return call


def make_hf_call(cfg: dict) -> Callable[[List[str]], List[str]]:
    """Returns a batched call. HF-specific batching for throughput."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs: dict = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if cfg.get("quant"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("torch_dtype", None)
    print(f"[hf] loading {cfg['id']} ...")
    tok = AutoTokenizer.from_pretrained(cfg["id"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["id"], **kwargs)
    model.eval()
    print(f"[hf] ready on {next(model.parameters()).device}")

    def apply_chat(prompt: str) -> str:
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
        except Exception:
            return prompt

    @torch.no_grad()
    def call(prompts: List[str]) -> List[str]:
        texts = [apply_chat(p) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=3072).to(model.device)
        input_len = enc.input_ids.shape[1]
        out = model.generate(
            **enc,
            max_new_tokens=500,
            do_sample=True, temperature=0.5, top_p=0.9,
            pad_token_id=tok.eos_token_id, repetition_penalty=1.05,
        )
        return [tok.decode(o[input_len:], skip_special_tokens=True).strip() for o in out]

    call._teardown = lambda: (_teardown_hf(model, tok))  # attached for later cleanup
    return call


def _teardown_hf(model, tok):
    import torch
    try:
        del model, tok
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_openai_call(cfg: dict) -> Callable[[str], str]:
    # Load .env so OPENAI_API_KEY at project root is picked up (matches
    # the pattern in scripts/evaluate_cot_gpt4_1_mini.py).
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not set. Put it in .env at repo root "
            "(OPENAI_API_KEY=sk-...) or export it."
        )
    from openai import OpenAI
    client = OpenAI()
    model_id = cfg["id"]
    def call(prompt: str) -> str:
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == 3:
                    print(f"  [!] openai error (giving up): {e}", file=sys.stderr)
                    return ""
                time.sleep(2 ** attempt)
        return ""
    return call


# ─────────────────────────────────────────────────────────────────────────────
# Track runner
# ─────────────────────────────────────────────────────────────────────────────

TRACKS = [
    ("gt",    "gt.csv",    "codemix_answer"),      # expected: no
    ("hallu", "hallu.csv", "hallucinated_answer"), # expected: yes
]


def run_model(slug: str) -> None:
    if slug not in MODELS:
        raise SystemExit(f"Unknown model slug: {slug}")
    cfg = MODELS[slug]
    print(f"\n{'=' * 70}\n[start] {slug}  ({cfg['backend']}: {cfg['id']})\n{'=' * 70}")

    # backend init
    if cfg["backend"] == "ollama":
        call_one = make_ollama_call(cfg)
        batched = False
        batch_size = 1
    elif cfg["backend"] == "hf":
        call_batch = make_hf_call(cfg)
        batched = True
        batch_size = cfg.get("batch", 4)
    elif cfg["backend"] == "openai":
        call_one = make_openai_call(cfg)
        batched = False
        batch_size = 1
    else:
        raise SystemExit(f"bad backend {cfg['backend']}")

    for track_name, sample_file, answer_field in TRACKS:
        src = SAMPLE_DIR / sample_file
        if not src.exists():
            print(f"[skip] {track_name}: sample {src} missing")
            continue
        df = pd.read_csv(src, encoding="utf-8", on_bad_lines="skip")
        out_path = OUT_DIR / f"codemix_cot_{track_name}_{slug}.csv"
        writer, done = open_writer(out_path, list(df.columns))
        rows_all = [dict(r) for _, r in df.iterrows()]
        pending = [r for r in rows_all if row_id(r) not in done]
        print(f"\n-- track {track_name} -- total {len(df)}  done {len(done)}  pending {len(pending)}")
        print(f"   -> {out_path.relative_to(ROOT)}")
        t0 = time.time()

        if batched:
            for i in range(0, len(pending), batch_size):
                chunk = pending[i:i + batch_size]
                prompts = [build_cot_prompt(r, answer_field) for r in chunk]
                raws = call_batch(prompts)
                for r, raw in zip(chunk, raws):
                    label = parse_yesno(raw)
                    r_out = dict(r)
                    r_out["raw_response"] = raw
                    r_out["is_hallucinated"] = label
                    writer.writerow(r_out)
                writer._f.flush()
                n = i + len(chunk)
                rate = n / max(time.time() - t0, 1e-6)
                print(f"   {n}/{len(pending)}  ({rate:.2f}/s)")
        else:
            for i, r in enumerate(pending, 1):
                prompt = build_cot_prompt(r, answer_field)
                raw = call_one(prompt)
                label = parse_yesno(raw)
                r_out = dict(r)
                r_out["raw_response"] = raw
                r_out["is_hallucinated"] = label
                writer.writerow(r_out)
                writer._f.flush()
                if i % 10 == 0 or i == len(pending):
                    rate = i / max(time.time() - t0, 1e-6)
                    eta = (len(pending) - i) / max(rate, 1e-6) / 60
                    print(f"   {i}/{len(pending)}  ({rate:.2f}/s)  eta={eta:.1f}min  last={label}")

        writer._f.close()

    # HF teardown so next model gets clean VRAM
    if cfg["backend"] == "hf":
        try:
            call_batch._teardown()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, help="Slug from MODELS dict")
    ap.add_argument("--all-ollama", action="store_true", help="Run all 5 Ollama models sequentially")
    ap.add_argument("--all-hf", action="store_true", help="Run all 3 HF models sequentially (small->large)")
    args = ap.parse_args()

    if args.all_ollama:
        for slug in OLLAMA_MODELS:
            run_model(slug)
    elif args.all_hf:
        # small -> large so a crash still gives us the small ones
        for slug in ("titullm_3b", "tigerllm_9b", "banglallama_13b"):
            run_model(slug)
    elif args.model:
        run_model(args.model)
    else:
        ap.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
