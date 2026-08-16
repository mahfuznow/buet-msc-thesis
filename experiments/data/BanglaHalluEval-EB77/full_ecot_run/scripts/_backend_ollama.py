"""Ollama backend. Returns raw response text given a prompt.

Used by all five Ollama-served judges: qwen2.5:32b-instruct, gemma2:27b,
deepseek-r1:14b, mistral-nemo:latest, llama3.1:8b.

Behavior mirrors scripts/evaluate_cot_ollama.py — block-and-retry on
connection drops so a transient outage doesn't write bogus rows.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def make_call_fn(model: str,
                 num_predict_for: Callable[[str], int],
                 num_ctx_for: Callable[[str], int]) -> Callable[[str, str], str]:
    """Return a (prompt, task) -> raw_response callable.

    `num_predict_for(task)` and `num_ctx_for(task)` are picked at call time
    so deepseek-r1 (which emits long <think> blocks) can get a bigger budget
    than the other Ollama judges.
    """
    url = f"{BASE_URL.rstrip('/')}/api/generate"

    def call(prompt: str, task: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",            # Ollama JSON mode
            "options": {
                "num_predict": num_predict_for(task),
                "num_ctx":     num_ctx_for(task),
                "temperature": 0,
            },
        }
        transient_left = 5
        waited = False
        while True:
            try:
                resp = requests.post(url, json=payload, timeout=600)
                resp.raise_for_status()
                if waited:
                    print("  [ok] Ollama reachable again, resuming.")
                return resp.json().get("response", "").strip()
            except requests.exceptions.ConnectionError:
                if not waited:
                    print(f"  [!] Ollama unreachable at {url}; backing off until it returns.")
                waited = True
                time.sleep(15)
            except requests.exceptions.RequestException as exc:
                transient_left -= 1
                if transient_left <= 0:
                    print(f"  [!] giving up on row after persistent errors: {exc}", file=sys.stderr)
                    return ""
                time.sleep(2 + (5 - transient_left) * 2)

    return call
