from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_MODEL_ID = "microsoft/phi-2"


@lru_cache(maxsize=1)
def _load():
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def complete(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    tokenizer, model = _load()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature or 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()


def complete_with_context(question: str, context: str) -> str:
    prompt = (
        f"Instruct: Answer the question using only the context below.\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Output:"
    )
    return complete(prompt)
