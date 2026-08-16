"""BanglaLLM/bangla-llama-13b-instruct-v0.1 backend for full_ecot_run.

Uses a 4-bit load by default so the model fits on a cost-efficient 24 GB GPU.
"""

from __future__ import annotations

import sys
from typing import Callable

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_ID = "BanglaLLM/bangla-llama-13b-instruct-v0.1"


def make_call_fn(
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    quantize: bool = True,
) -> Callable[[str, str], str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[banglallama] loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        print("[banglallama] using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model.eval()
    print(f"[banglallama] ready on {next(model.parameters()).device}")

    do_sample = temperature > 0.0

    def call(prompt: str, task: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            prompt_text = prompt

        enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][enc.input_ids.shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    return call