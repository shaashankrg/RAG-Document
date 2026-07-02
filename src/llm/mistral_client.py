from __future__ import annotations

import os

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


_DEFAULT_MODEL = "mistral-small-latest"


def _client() -> MistralClient:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not set")
    return MistralClient(api_key=api_key)


def complete(
    prompt: str,
    system: str = "You are a precise clinical document analyst.",
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    client = _client()
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=prompt),
    ]
    resp = client.chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content.strip()


def complete_with_context(question: str, context: str, model: str = _DEFAULT_MODEL) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above."
    return complete(prompt, model=model)
