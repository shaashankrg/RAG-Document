from __future__ import annotations

from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image


def run_tesseract(img: np.ndarray, lang: str = "eng", psm: int = 3) -> str:
    pil_img = Image.fromarray(img)
    config = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(pil_img, lang=lang, config=config).strip()


def run_tesseract_with_confidence(img: np.ndarray, lang: str = "eng") -> dict:
    pil_img = Image.fromarray(img)
    data = pytesseract.image_to_data(pil_img, lang=lang, output_type=pytesseract.Output.DICT)
    words = [
        {"text": w, "conf": int(c)}
        for w, c in zip(data["text"], data["conf"])
        if w.strip() and int(c) > 0
    ]
    text = " ".join(d["text"] for d in words)
    avg_conf = sum(d["conf"] for d in words) / len(words) if words else 0.0
    return {"text": text, "avg_confidence": avg_conf, "words": words}
