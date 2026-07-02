from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _get_paddle():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def run_paddleocr(img: np.ndarray) -> str:
    ocr = _get_paddle()
    result = ocr.ocr(img, cls=True)
    if not result or not result[0]:
        return ""
    lines = [word_info[1][0] for line in result for word_info in line]
    return "\n".join(lines)


def run_paddleocr_with_confidence(img: np.ndarray) -> dict:
    ocr = _get_paddle()
    result = ocr.ocr(img, cls=True)
    if not result or not result[0]:
        return {"text": "", "avg_confidence": 0.0, "words": []}

    words = [
        {"text": word_info[1][0], "conf": word_info[1][1]}
        for line in result
        for word_info in line
    ]
    text = " ".join(w["text"] for w in words)
    avg_conf = sum(w["conf"] for w in words) / len(words) if words else 0.0
    return {"text": text, "avg_confidence": avg_conf, "words": words}
