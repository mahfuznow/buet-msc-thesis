"""TigerLLM-9B backend (HuggingFace Transformers, bfloat16, single-GPU).

Loads md-nishat-008/TigerLLM-9B-it once and returns a callable that produces
JSON-shaped text. Mirrors the loading recipe from scripts/evaluate_cot_tigerllm.py.

Requires: torch, transformers. ~18 GB GPU memory in bfloat16.
"""

from __future__ import annotations

import sys
from typing import Callable

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_ID = "md-nishat-008/TigerLLM-9B-it"


def make_call_fn(max_new_tokens: int = 1024, temperature: float = 0.0) -> Callable[[str, str], str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[tigerllm] loading {MODEL_ID} in bfloat16 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[tigerllm] ready on {model.device}")

    do_sample = temperature > 0.0

    def call(prompt: str, task: str) -> str:
        # Use the chat template so the JSON-shaped response training kicks in.
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs.shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    return call
