"""
Rasterize existing digital FDA-label PDFs into image-only "fake scanned" PDFs.

Each output page is a plain image (no text layer), so OCR engines have to read
pixels the same way they would on a genuinely scanned document. A couple of
docs are additionally degraded (rotation, low DPI, blur, noise, JPEG artifacts)
to avoid benchmarking OCR only against pristine renders.

Ground truth is NOT regenerated -- these are rasterized versions of PDFs we
already have hand-verified ground truth for in data/ground_truth/, keyed by
the same filename stem.

Usage: python scripts/make_fake_scans.py
"""

from __future__ import annotations

import io
import random

import fitz
import numpy as np
from PIL import Image, ImageFilter

SRC_DIR = "data/raw"
DST_DIR = "data/raw_scanned"

# (filename stem, dpi, rotation_degrees, add_noise_blur, jpeg_quality)
# jpeg_quality is also the storage quality for every page (real scanners emit JPEG,
# not raw bitmaps) -- degraded jobs just use a lower quality to add compression artifacts.
JOBS = [
    ("20140820_2468ba8d-4c77-4ea0-88d8-b64497a72222", 200, 0.0, False, 90),   # CHANTIX -- clean, long/dense
    ("20260624_e0e6412f-50b4-4fd4-9364-62818d121a07", 200, 0.0, False, 90),   # IBRANCE -- clean, long/dense
    ("20240627_a60cc18b-0631-4cf0-b021-9f52224ece65", 130, 2.5, False, 75),   # LIPITOR -- low-res + skewed
    ("20250410_a6cc97d8-252a-4527-a470-6d9e356342fd", 110, 1.2, True, 40),    # ADVIL -- low-res + noisy/blurry/compressed
]


def degrade(img: Image.Image, rotation_deg: float, add_noise_blur: bool) -> Image.Image:
    if rotation_deg:
        img = img.rotate(rotation_deg, expand=True, fillcolor="white", resample=Image.BICUBIC)

    if add_noise_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        arr = np.array(img).astype(np.int16)
        noise = np.random.normal(0, 12, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def make_fake_scan(stem: str, dpi: int, rotation_deg: float, add_noise_blur: bool, jpeg_quality: int) -> None:
    random.seed(42)
    np.random.seed(42)

    src_path = f"{SRC_DIR}/{stem}.pdf"
    dst_path = f"{DST_DIR}/{stem}.pdf"

    src = fitz.open(src_path)
    out = fitz.open()

    for page in src:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        img = degrade(img, rotation_deg, add_noise_blur)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        img_bytes = buf.getvalue()

        page_w_pt = img.width * 72 / dpi
        page_h_pt = img.height * 72 / dpi
        new_page = out.new_page(width=page_w_pt, height=page_h_pt)
        new_page.insert_image(fitz.Rect(0, 0, page_w_pt, page_h_pt), stream=img_bytes)

    page_count = len(out)
    out.save(dst_path)
    out.close()
    src.close()
    print(f"{stem}: {page_count} pages -> {dst_path} "
          f"(dpi={dpi}, rotation={rotation_deg}, noise_blur={add_noise_blur}, jpeg_q={jpeg_quality})")


if __name__ == "__main__":
    import os
    os.makedirs(DST_DIR, exist_ok=True)
    for job in JOBS:
        make_fake_scan(*job)
