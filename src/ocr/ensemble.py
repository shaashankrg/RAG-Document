from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ocr.preprocess import preprocess_pdf_page
from src.ocr.tesseract_engine import run_tesseract_with_confidence
from src.ocr.paddleocr_engine import run_paddleocr_with_confidence
from src.ocr.easyocr_engine import run_easyocr_with_confidence


_ENGINES = {
    "tesseract": run_tesseract_with_confidence,
    "paddleocr": run_paddleocr_with_confidence,
    "easyocr": run_easyocr_with_confidence,
}


def run_ocr_ensemble(
    pdf_path: str | Path,
    engines: list[str] | None = None,
    strategy: str = "best_confidence",
) -> str:
    """Return OCR text for all pages of a PDF.

    strategy:
      - "best_confidence": pick the engine with the highest avg confidence per page
      - "concatenate": join outputs from all engines (useful for debugging)
    """
    pdf_path = Path(pdf_path)
    import fitz
    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)
    doc.close()

    selected_engines = {k: v for k, v in _ENGINES.items() if (engines is None or k in engines)}
    page_texts: list[str] = []

    for page_num in range(num_pages):
        img = preprocess_pdf_page(pdf_path, page_num)
        results = {name: fn(img) for name, fn in selected_engines.items()}

        if strategy == "best_confidence":
            best = max(results.values(), key=lambda r: r["avg_confidence"])
            page_texts.append(best["text"])
        else:
            combined = "\n---\n".join(
                f"[{name}]\n{r['text']}" for name, r in results.items()
            )
            page_texts.append(combined)

    return "\n\n".join(page_texts)
