from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz


@dataclass
class ExtractedTable:
    page_num: int
    bbox: tuple[float, float, float, float]
    rows: list[list[str]] = field(default_factory=list)

    @property
    def header(self) -> list[str]:
        return self.rows[0] if self.rows else []

    def to_dict_list(self) -> list[dict]:
        if len(self.rows) < 2:
            return []
        headers = self.header
        return [dict(zip(headers, row)) for row in self.rows[1:]]


def extract_tables(pdf_path: str | Path) -> list[ExtractedTable]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    tables: list[ExtractedTable] = []

    for page_num, page in enumerate(doc):
        for tab in page.find_tables():
            raw = tab.extract()
            cleaned = [[cell or "" for cell in row] for row in raw]
            tables.append(ExtractedTable(
                page_num=page_num,
                bbox=tuple(tab.bbox),
                rows=cleaned,
            ))

    doc.close()
    return tables
