from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_page_image(pdf_path: str | Path, page_num: int, dpi: int = 300) -> np.ndarray:
    import fitz
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    doc.close()
    return img


def deskew(img: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(img < 128))
    if len(coords) == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(img, h=10)


def binarize(img: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )


def preprocess_page(img: np.ndarray) -> np.ndarray:
    img = denoise(img)
    img = deskew(img)
    img = binarize(img)
    return img


def preprocess_pdf_page(pdf_path: str | Path, page_num: int, dpi: int = 300) -> np.ndarray:
    img = load_page_image(pdf_path, page_num, dpi)
    return preprocess_page(img)
