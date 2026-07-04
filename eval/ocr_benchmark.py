from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from src.ocr.preprocess import preprocess_pdf_page
from src.ocr.tesseract_engine import run_tesseract_with_confidence
from src.ocr.paddleocr_engine import run_paddleocr_with_confidence
from src.ocr.easyocr_engine import run_easyocr_with_confidence
from eval.scoring import DEFAULT_THRESHOLD, is_date_match, is_match

DATA_DIR = Path("data")
SCANNED_DIR = DATA_DIR / "raw_scanned"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
RESULTS_DIR = Path("eval") / "results"

# The first 2 pages cover the FDA "HIGHLIGHTS OF PRESCRIBING INFORMATION" section,
# where nearly every scored field lives (brand/generic name, indication, boxed
# warning, revision dates). These documents run 10-43 pages; OCR-ing all of them
# with 3 engines isn't necessary for these fields and would make the benchmark far
# slower. Fields that genuinely only appear later in the document will show up as
# non-matches here -- that's a scope limitation of this benchmark, not an OCR
# engine failure, and should be read that way in the results.
PAGES_TO_SCAN = 2

DOCS = [
    "20240627_a60cc18b-0631-4cf0-b021-9f52224ece65",  # LIPITOR
    "20250410_a6cc97d8-252a-4527-a470-6d9e356342fd",  # ADVIL
    "20140820_2468ba8d-4c77-4ea0-88d8-b64497a72222",  # CHANTIX
    "20260624_e0e6412f-50b4-4fd4-9364-62818d121a07",  # IBRANCE
]

DATE_FIELD_PATHS: list[tuple[str, ...]] = [
    ("initial_us_approval",),
    ("revision_dates", "prescribing_information"),
    ("revision_dates", "spl_document"),
    ("revision_dates", "medication_guide"),
]

# has_boxed_warning is a structural boolean, not a text/date value -- it would need
# presence-of-header detection logic, not either comparator here, so it's excluded.
#
# application_number and ndc are excluded too, but for a scope reason, not a type
# reason: confirmed by a full-document page scan (see scratch investigation) that
# both live in the "HOW SUPPLIED" / package-labeling section near the end of these
# PI documents (e.g. pages 33-41 of LIPITOR's 43), never within PAGES_TO_SCAN. The
# low-but-nonzero scores they produced when scored anyway were coincidental
# similarity against unrelated text, not a matcher tokenization defect -- when OCR
# does hit the real field (LIPITOR page 40), it reads "NDA020702" as a single token,
# identical in format to ground truth. Revisit if PAGES_TO_SCAN ever covers back
# matter.
EXCLUDED_FIELDS = {"revision_dates", "has_boxed_warning", "application_number", "ndc"}

# Scope gaps confirmed for one specific document, not the field in general -- unlike
# application_number/ndc (out of scope for all 4 docs), manufacturer_labeler is
# correctly in scope and correctly matches for LIPITOR/CHANTIX/ADVIL. Only IBRANCE's
# page 0 layout omits the "Manufactured by" byline entirely (goes straight from the
# product line into HIGHLIGHTS content) -- confirmed absent by full-text search
# across pages 0-1, not a global exclusion. Tracked as (brand_name, field) pairs so
# scoring for other documents on this field isn't affected.
KNOWN_SCOPE_GAPS = {
    ("IBRANCE", "manufacturer_labeler"): (
        "byline not present in pages 0-1 for this document's layout (confirmed absent "
        "via full-text search); other docs put it right after the product line"
    ),
}

ENGINES: dict[str, Callable[..., dict]] = {
    "tesseract": run_tesseract_with_confidence,
    "paddleocr": run_paddleocr_with_confidence,
    "easyocr": run_easyocr_with_confidence,
}

# Tesseract reports confidence on a 0-100 scale; PaddleOCR/EasyOCR report 0-1.
# Normalize everything to 0-1 so engines are comparable in one table.
CONFIDENCE_SCALE = {
    "tesseract": 100.0,
}


def _iter_scored_fields(fields: dict):
    for path in DATE_FIELD_PATHS:
        value = fields
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value:
            yield ".".join(path), value, "date"

    for key, value in fields.items():
        if key in EXCLUDED_FIELDS:
            continue
        if isinstance(value, str) and value:
            yield key, value, "text"


def _ocr_document(stem: str, engine_fn: Callable[..., dict], scale: float) -> tuple[str, float]:
    texts = []
    confidences = []
    pdf_path = SCANNED_DIR / f"{stem}.pdf"
    for page in range(PAGES_TO_SCAN):
        img = preprocess_pdf_page(pdf_path, page)
        result = engine_fn(img)
        texts.append(result["text"])
        confidences.append(result["avg_confidence"])
    combined_text = "\n".join(texts)
    avg_confidence = (sum(confidences) / len(confidences) / scale) if confidences else 0.0
    return combined_text, avg_confidence


def run_benchmark(engines: dict[str, Callable[..., dict]] = ENGINES) -> list[dict]:
    rows = []
    for stem in DOCS:
        gt_path = GROUND_TRUTH_DIR / f"{stem}.json"
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        fields = ground_truth["fields"]
        brand = fields.get("brand_name", stem)

        for engine_name, engine_fn in engines.items():
            scale = CONFIDENCE_SCALE.get(engine_name, 1.0)
            ocr_text, avg_confidence = _ocr_document(stem, engine_fn, scale)

            for field_name, gt_value, field_type in _iter_scored_fields(fields):
                scope_gap = KNOWN_SCOPE_GAPS.get((brand, field_name))
                if scope_gap:
                    continue

                if field_type == "date":
                    matched, candidate = is_date_match(gt_value, ocr_text)
                    score = 1.0 if matched else 0.0
                else:
                    matched, score = is_match(gt_value, ocr_text, DEFAULT_THRESHOLD)
                    candidate = None

                rows.append({
                    "document": brand,
                    "engine": engine_name,
                    "field": field_name,
                    "field_type": field_type,
                    "ground_truth": gt_value,
                    "matched": matched,
                    "matched_candidate": candidate,
                    "score": round(score, 3),
                    "doc_avg_confidence": round(avg_confidence, 3),
                })
    return rows


def print_table(rows: list[dict]) -> None:
    header = f"{'document':<10} {'engine':<10} {'field':<28} {'type':<5} {'match':<6} {'score':<6} ground_truth"
    print(header)
    print("-" * len(header))
    for row in rows:
        gt_preview = row["ground_truth"][:40]
        print(
            f"{row['document']:<10} {row['engine']:<10} {row['field']:<28} "
            f"{row['field_type']:<5} {str(row['matched']):<6} {row['score']:<6} {gt_preview}"
        )


def summarize(rows: list[dict]) -> None:
    per_engine_doc = defaultdict(list)
    for row in rows:
        per_engine_doc[(row["engine"], row["document"])].append(row)

    print()
    print(f"{'engine':<10} {'document':<10} {'fields_matched':<16} {'match_rate':<12} doc_confidence")
    for (engine, doc), doc_rows in per_engine_doc.items():
        matched = sum(1 for r in doc_rows if r["matched"])
        total = len(doc_rows)
        rate = matched / total if total else 0.0
        conf = doc_rows[0]["doc_avg_confidence"]
        print(f"{engine:<10} {doc:<10} {matched}/{total:<14} {rate:<12.1%} {conf:.3f}")


if __name__ == "__main__":
    results = run_benchmark()
    print_table(results)
    summarize(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ocr_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} rows to {out_path}")
