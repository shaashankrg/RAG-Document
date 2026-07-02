from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


DOC_TYPE_PATTERNS = {
    "CSR": re.compile(r"Clinical\s+Study\s+Report", re.IGNORECASE),
    "IB": re.compile(r"Investigator\s+Brochure", re.IGNORECASE),
    "Protocol": re.compile(r"\bProtocol\b(?!\s+Number|\s+No)", re.IGNORECASE),
    "ICF": re.compile(r"Informed\s+Consent", re.IGNORECASE),
    "AE_Report": re.compile(r"Adverse\s+Event\s+Report", re.IGNORECASE),
    "SAP": re.compile(r"Statistical\s+Analysis\s+Plan", re.IGNORECASE),
}

SECTION_PATTERNS = {
    "synopsis": re.compile(r"\bSYNOPSIS\b", re.IGNORECASE),
    "methods": re.compile(r"\b(?:METHODS|STUDY\s+DESIGN)\b", re.IGNORECASE),
    "results": re.compile(r"\bRESULTS\b", re.IGNORECASE),
    "safety": re.compile(r"\b(?:SAFETY|ADVERSE\s+EVENTS?)\b", re.IGNORECASE),
    "efficacy": re.compile(r"\bEFFICACY\b", re.IGNORECASE),
}

DATE_RE = re.compile(
    r"(?:Date\s*[:\-]?\s*)?(\d{1,2}[\s\-/]\w{3,9}[\s\-/]\d{4}|\w{3,9}\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


@dataclass
class DocumentMetadata:
    doc_type: Optional[str] = None
    section: Optional[str] = None
    date: Optional[str] = None
    page_num: Optional[int] = None


def tag_page(text: str, page_num: int | None = None) -> DocumentMetadata:
    meta = DocumentMetadata(page_num=page_num)

    for dtype, pat in DOC_TYPE_PATTERNS.items():
        if pat.search(text):
            meta.doc_type = dtype
            break

    for section, pat in SECTION_PATTERNS.items():
        if pat.search(text):
            meta.section = section
            break

    m = DATE_RE.search(text)
    if m:
        meta.date = m.group(1).strip()

    return meta
