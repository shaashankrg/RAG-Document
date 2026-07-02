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
