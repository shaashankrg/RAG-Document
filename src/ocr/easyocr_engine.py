from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _get_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


def run_easyocr(img: np.ndarray) -> str:
    reader = _get_reader()
    results = reader.readtext(img, detail=0, paragraph=True)
    return "\n".join(results)


def run_easyocr_with_confidence(img: np.ndarray) -> dict:
    reader = _get_reader()
    results = reader.readtext(img, detail=1)
    if not results:
        return {"text": "", "avg_confidence": 0.0, "words": []}

    words = [{"text": r[1], "conf": r[2]} for r in results]
    text = " ".join(w["text"] for w in words)
    avg_conf = sum(w["conf"] for w in words) / len(words) if words else 0.0
    return {"text": text, "avg_confidence": avg_conf, "words": words}
