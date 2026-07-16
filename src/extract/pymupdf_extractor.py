from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

@dataclass
class PageBlock:
    page_num: int
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    block_type: str  # "text" | "image"

@dataclass
class ExtractedDocument:
    path: str
    num_pages: int
    blocks: list[PageBlock] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.block_type == "text")

    def page_text(self, page_num: int) -> str:
        return "\n".join(
            b.text for b in self.blocks
            if b.block_type == "text" and b.page_num == page_num
        )


def extract_pdf(pdf_path: str | Path) -> ExtractedDocument:
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    result = ExtractedDocument(path=str(pdf_path), num_pages=len(doc))

    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:  # text block
                text = " ".join(
                    span["text"]
                    for line in block["lines"]
                    for span in line["spans"]
                ).strip()
                if text:
                    result.blocks.append(PageBlock(
                        page_num=page_num,
                        text=text,
                        bbox=tuple(block["bbox"]),
                        block_type="text",
                    ))
            elif block["type"] == 1:  # image block
                result.blocks.append(PageBlock(
                    page_num=page_num,
                    text="",
                    bbox=tuple(block["bbox"]),
                    block_type="image",
                ))

    doc.close()
    return result


def iter_pages(pdf_path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield (page_num, page_text) for each page."""
    doc = extract_pdf(pdf_path)
    for p in range(doc.num_pages):
        yield p, doc.page_text(p)


# Vertical gap (in points) below which two line fragments are treated as the
# same visual row (e.g. a bullet glyph and the bullet's text, which these SPL
# PDFs emit as separate blocks with identical baselines).
_ROW_TOLERANCE = 2.0


def _page_text_reading_order(page: fitz.Page) -> str:
    """Page text with lines in visual reading order (top-to-bottom, then
    left-to-right), instead of PDF content-stream order.

    These SPL-generated label PDFs emit blocks out of visual order -- section
    headers and bullet glyphs first, bullet body text appended at the end of
    the stream -- so stream-order extraction places content under the wrong
    section header (e.g. NORVASC's contraindication line lands after DRUG
    INTERACTIONS). Verified against rendered pages: the layout itself is
    single-column; only the stream order is scrambled.

    Assumes single-column pages. Measured across all 16 sample documents'
    first 2 pages (2026-07): zero two-column pages, so no column handling is
    built here. A genuinely two-column page would interleave under this sort;
    if multi-column documents ever enter the corpus, re-run that layout scan
    and add column banding first.
    """
    fragments = []  # (y0, x0, text)
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            text = " ".join(s["text"] for s in line["spans"]).strip()
            if text:
                x0, y0, _, _ = line["bbox"]
                fragments.append((y0, x0, text))
    fragments.sort(key=lambda f: (f[0], f[1]))

    # Group same-baseline fragments into one row, ordered left-to-right, so a
    # bullet glyph rejoins the text it belongs to on a single output line.
    rows: list[list[tuple[float, str]]] = []
    row_y = None
    for y0, x0, text in fragments:
        if row_y is None or y0 - row_y > _ROW_TOLERANCE:
            rows.append([])
            row_y = y0
        rows[-1].append((x0, text))
    return "\n".join(" ".join(t for _, t in sorted(row)) for row in rows)


def extract_pages_reading_order(pdf_path: str | Path, max_pages: int | None = None) -> list[str]:
    """Per-page text in visual reading order (see _page_text_reading_order)."""
    doc = fitz.open(str(pdf_path))
    try:
        n = len(doc) if max_pages is None else min(max_pages, len(doc))
        return [_page_text_reading_order(doc[p]) for p in range(n)]
    finally:
        doc.close()
