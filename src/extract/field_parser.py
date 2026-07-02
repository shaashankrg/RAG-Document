from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedFields:
    protocol_number: Optional[str] = None
    sponsor: Optional[str] = None
    study_title: Optional[str] = None
    phase: Optional[str] = None
    indication: Optional[str] = None
    document_date: Optional[str] = None
    document_type: Optional[str] = None


_PATTERNS: dict[str, re.Pattern] = {
    "protocol_number": re.compile(
        r"(?:protocol\s+(?:number|no\.?|#)\s*[:\-]?\s*)([A-Z0-9\-]{4,20})"
        r"|\b((?:NDA|BLA)\s?\d{5,7})\b",
        re.IGNORECASE,
    ),
    "sponsor": re.compile(
        r"(?:sponsor\s*[:\-]?\s*)([A-Za-z][\w\s,\.]+?)(?:\n|$)"
        r"|(?:Labeler\s*-\s*)([A-Za-z][\w\s,\.&]+?)(?:\s*\(|\n|$)"
        r"|(?:(?:Distributed|Manufactured|Marketed)\s+by\s*[:\-]?\s*)([A-Za-z][\w\s,\.&]+?)(?:\n|$)",
        re.IGNORECASE,
    ),
    "study_title": re.compile(
        r"(?:(?:study|trial)\s+title\s*[:\-]?\s*)(.{10,200}?)(?:\n|$)",
        re.IGNORECASE,
    ),
    "phase": re.compile(
        r"\bPhase\s+(I{1,3}V?|[1-4](?:[Aa]|[Bb])?)\b",
        re.IGNORECASE,
    ),
    "indication": re.compile(
        # Case-sensitive on purpose: the real section header is always full-caps
        # "INDICATIONS AND USAGE", while the RECENT MAJOR CHANGES changelog references
        # it in title case ("Indications and Usage 04/2023 ...") and would otherwise
        # match first since it appears earlier in the document.
        r"INDICATIONS AND USAGE\s+(.{10,400}?)(?:\n|$)",
    ),
    "document_date": re.compile(
        r"(?:date\s*[:\-]?\s*)(\d{1,2}[\s\-/]\w+[\s\-/]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})"
        r"|(?:Revised\s*[:\-]?\s*)(\d{1,2}/\d{4}|\w+\s+\d{4})",
        re.IGNORECASE,
    ),
    "document_type": re.compile(
        r"\b(Clinical\s+Study\s+Report|Investigator\s+Brochure|Protocol\s+Amendment|Informed\s+Consent|Adverse\s+Event\s+Report"
        r"|Highlights\s+of\s+Prescribing\s+Information|Prescribing\s+Information|Drug\s+Facts|Medication\s+Guide)\b",
        re.IGNORECASE,
    ),
}


def parse_fields(text: str) -> ParsedFields:
    result = ParsedFields()
    for field_name, pattern in _PATTERNS.items():
        m = pattern.search(text)
        if m:
            value = m.group(m.lastindex).strip() if m.lastindex else m.group(0).strip()
            setattr(result, field_name, value)
    return result
