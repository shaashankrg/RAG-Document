from __future__ import annotations

import re
from dataclasses import dataclass


BOUNDARY_SIGNALS = [
    re.compile(r"^\s*(?:CONFIDENTIAL|PROPRIETARY)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:Protocol|Study)\s+(?:Number|No\.?|#)\s*[:\-]?\s*[A-Z0-9\-]{4,}", re.IGNORECASE),
    re.compile(r"\bTable\s+of\s+Contents\b", re.IGNORECASE),
    re.compile(r"^1\.\s+(?:INTRODUCTION|BACKGROUND|SYNOPSIS)\s*$", re.IGNORECASE | re.MULTILINE),
]


@dataclass
class DocumentBoundary:
    start_page: int
    end_page: int
    confidence: float


def detect_boundaries(page_texts: list[str]) -> list[DocumentBoundary]:
    """Detect where one document ends and another begins within a multi-doc PDF."""
    scores: list[float] = []
    for text in page_texts:
        hit = sum(1 for p in BOUNDARY_SIGNALS if p.search(text))
        scores.append(hit / len(BOUNDARY_SIGNALS))

    boundaries: list[DocumentBoundary] = []
    starts = [0]
    for i, score in enumerate(scores[1:], 1):
        if score >= 0.5 and (scores[i - 1] < 0.3 or i == 1):
            starts.append(i)

    for j, start in enumerate(starts):
        end = starts[j + 1] - 1 if j + 1 < len(starts) else len(page_texts) - 1
        avg_score = sum(scores[start: end + 1]) / (end - start + 1)
        boundaries.append(DocumentBoundary(start, end, avg_score))

    return boundaries
