from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher

from dateutil import parser as date_parser

DEFAULT_THRESHOLD = 0.75

_DATE_ANCHOR = datetime(2000, 1, 1)

_DATE_CANDIDATE_RE = re.compile(
    r"\b\d{1,2}/\d{1,2}/\d{4}\b"
    r"|\b\d{1,2}/\d{4}\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{4}\b"
    r"|\b(?:19|20)\d{2}\b"
)


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def best_window_match(needle: str, haystack: str) -> float:
    """Slide a needle-length word window across haystack and keep the best score.

    A whole-text comparison (text_similarity(needle, haystack)) collapses toward 0
    whenever haystack is a full OCR page and needle is a short field value, even when
    the field is present verbatim -- SequenceMatcher penalizes all the surrounding
    text as "unmatched". Scoring the best-matching window instead of the whole blob
    fixes that.
    """
    needle = needle.strip()
    words = haystack.split()
    needle_len = max(len(needle.split()), 1)

    if len(words) <= needle_len:
        return text_similarity(needle, haystack)

    best = 0.0
    for i in range(len(words) - needle_len + 1):
        window = " ".join(words[i : i + needle_len])
        score = text_similarity(needle, window)
        if score > best:
            best = score
    return best


def is_match(ground_truth: str, ocr_text: str, threshold: float = DEFAULT_THRESHOLD) -> tuple[bool, float]:
    score = best_window_match(ground_truth, ocr_text)
    return score >= threshold, score


def _date_granularity(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"(19|20)\d{2}", value):
        return "year"
    if re.fullmatch(r"\d{1,2}/\d{4}", value):
        return "month"
    return "day"


def _parse_date(value: str) -> datetime | None:
    try:
        return date_parser.parse(value, default=_DATE_ANCHOR)
    except (ValueError, OverflowError):
        return None


def _dates_equal(a: str, b: str, granularity: str) -> bool:
    parsed_a, parsed_b = _parse_date(a), _parse_date(b)
    if parsed_a is None or parsed_b is None:
        return False
    if granularity == "year":
        return parsed_a.year == parsed_b.year
    if granularity == "month":
        return (parsed_a.year, parsed_a.month) == (parsed_b.year, parsed_b.month)
    return (parsed_a.year, parsed_a.month, parsed_a.day) == (parsed_b.year, parsed_b.month, parsed_b.day)


def is_date_match(ground_truth: str, ocr_text: str) -> tuple[bool, str | None]:
    """Strict comparator: parse ground_truth and every date-shaped candidate found
    in ocr_text, and report a match only on exact parsed-date equality (at whatever
    granularity ground_truth specifies -- year-only, month/year, or full date).
    Unlike is_match, this is not a similarity score: dates either match or they don't.
    """
    granularity = _date_granularity(ground_truth)
    for candidate in _DATE_CANDIDATE_RE.findall(ocr_text):
        if _dates_equal(ground_truth, candidate, granularity):
            return True, candidate
    return False, None
