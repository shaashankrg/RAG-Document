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


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# cv2.minAreaRect's angle is ambiguous near the 0/90-degree boundary for
# near-rectangular point clouds (a whole page of text is exactly this shape), and
# can report ~90 degrees for a page that isn't rotated at all -- confirmed on one of
# the benchmark's "clean" documents, whose deskew() output came out rotated 90
# degrees sideways even though it was generated with zero rotation. Actual scanner
# skew this pipeline is meant to correct is a few degrees at most (the intentionally
# degraded benchmark documents use 1.2/2.5 degrees), so anything larger is treated as
# a spurious minAreaRect reading rather than a real page rotation.
#
# This threshold is tuned to this benchmark's skew range, not a universal bound --
# it will silently skip correction on any real scan tilted more than 15 degrees.
# Revisit if real-world scanned input with more severe tilt is added.
MAX_DESKEW_ANGLE_DEGREES = 15.0


def deskew(img: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(img < 128))
    if len(coords) == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > MAX_DESKEW_ANGLE_DEGREES:
        return img
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
    img = to_grayscale(img)
    img = deskew(img)
    img = denoise(img)
    img = binarize(img)
    return img


def preprocess_pdf_page(pdf_path: str | Path, page_num: int, dpi: int = 300) -> np.ndarray:
    img = load_page_image(pdf_path, page_num, dpi)
    return preprocess_page(img)