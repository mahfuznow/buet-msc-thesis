"""OpenAI backend. Returns raw response text given a prompt.

Used by GPT-4.1 mini. Loads OPENAI_API_KEY from .env at repo root.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

try:
    sys.stdout.reconfigure(encoding="utf-8")
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


def make_call_fn(model: str = "gpt-4.1-mini",
                 max_output_tokens: int = 2000,
                 temperature: float = 0.0,
                 timeout: float = 120.0) -> Callable[[str, str], str]:
    repo_root = Path(__file__).resolve().parent.parent.parent
    if load_dotenv is not None:
        load_dotenv(repo_root / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not in environment / .env")
    client = OpenAI(api_key=api_key, timeout=timeout)

    def call(prompt: str, task: str) -> str:
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                wait = min(2 ** attempt, 60)
                if attempt == 4:
                    print(f"  [!] OpenAI error after 5 retries: {exc}", file=sys.stderr)
                    return ""
                print(f"  [!] OpenAI error (attempt {attempt+1}/5, retry in {wait}s): {exc}",
                      file=sys.stderr)
                time.sleep(wait)
        return ""

    return call
