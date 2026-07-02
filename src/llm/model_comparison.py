from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.llm import mistral_client, phi2_client


@dataclass
class ModelResult:
    model: str
    answer: str
    latency_s: float
    error: str = ""


@dataclass
class ComparisonReport:
    question: str
    context: str
    results: list[ModelResult] = field(default_factory=list)


def _run_gemini(question: str, context: str) -> ModelResult:
    import os
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    t0 = time.perf_counter()
    try:
        resp = model.generate_content(prompt)
        return ModelResult("gemini-1.5-flash", resp.text.strip(), time.perf_counter() - t0)
    except Exception as e:
        return ModelResult("gemini-1.5-flash", "", time.perf_counter() - t0, str(e))


def compare_models(question: str, context: str) -> ComparisonReport:
    report = ComparisonReport(question=question, context=context)

    for name, fn in [
        ("mistral", lambda: mistral_client.complete_with_context(question, context)),
        ("phi-2", lambda: phi2_client.complete_with_context(question, context)),
    ]:
        t0 = time.perf_counter()
        try:
            answer = fn()
            report.results.append(ModelResult(name, answer, time.perf_counter() - t0))
        except Exception as e:
            report.results.append(ModelResult(name, "", time.perf_counter() - t0, str(e)))

    report.results.append(_run_gemini(question, context))
    return report
