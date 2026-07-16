"""Held-out retrieval eval: one question per routable document, scored against
the top-1 retrieved chunk with the project's existing fuzzy-match scorer.

The question set (eval/rag_eval_set.json) was locked before the first run.
Metric: accuracy@1 = fraction of questions whose expected_phrase fuzzy-matches
(best_window_match >= 0.75) the text of the #1 retrieved chunk. hit@3 is also
reported: the phrase matched any of the top 3 chunks.

The routing functions below are copied verbatim from src/rag/section_router.py
rather than imported, because that file is a top-to-bottom experiment script --
importing it would execute every experiment (including a Gemini call). If the
router logic changes, re-copy; the constants are small and the duplication is
flagged here deliberately.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from llama_index.core.schema import TextNode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.scoring import is_match
from src.extract.pymupdf_extractor import extract_pages_reading_order
from src.rag.embeddings import get_embedding_model

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EVAL_SET = Path(__file__).with_name("rag_eval_set.json")
RESULTS = Path(__file__).with_name("results") / "rag_retrieval_eval.json"

# --- copied from src/rag/section_router.py (see module docstring) ---
SECTION_HEADER_RE = re.compile(
    r"^(WARNING:[A-Z0-9 ,:/&\-()]{2,112}|[A-Z][A-Z0-9 ,:/&\-]{9,120})$", re.MULTILINE
)
QUERY_PHRASE_TO_SECTION_PATTERN = {
    "black box warning": r"^WARNING:",
    "boxed warning": r"^WARNING:",
    "contraindication": r"^CONTRAINDICATIONS",
    "dosage": r"^DOSAGE",
    "indicated": r"^INDICATIONS AND USAGE",
    "used for": r"^INDICATIONS AND USAGE",
    "adverse reaction": r"^ADVERSE REACTIONS",
    "side effect": r"^ADVERSE REACTIONS",
    "drug interaction": r"^DRUG INTERACTIONS",
}
RX_DOCS = {
    "CHANTIX": "20140820_2468ba8d-4c77-4ea0-88d8-b64497a72222",
    "LIPITOR": "20240627_a60cc18b-0631-4cf0-b021-9f52224ece65",
    "IBRANCE": "20260624_e0e6412f-50b4-4fd4-9364-62818d121a07",
    "XELJANZ": "20260329_68e3d6b2-7838-4d2d-a417-09d919b43e13",
    "PREVNAR 20": "20250720_d4e2cf51-e6a8-4103-bb1d-6120c6474ff8",
    "ZITHROMAX": "20250212_8d24bacb-feff-4c6a-b8df-625e1435387a",
    "BENEFIX": "20250227_85faa5bc-cee5-4ef1-8d80-bdbcb7eba1e4",
    "ELIQUIS": "20250504_e9481622-7cc6-418a-acb6-c5450daae9b0",
    "NORVASC": "20250607_abd6a2ca-40c2-485c-bc53-db1c652505ed",
    "ZOLOFT": "20250724_fda754f6-d0f3-4dce-a17a-927d64f912f7",
    "DEPO-PROVERA": "20251219_199cf13e-0859-4a73-9b45-e700d0cd1049",
    "VIAGRA": "20260213_0b0be196-0c62-461c-94f4-9a35339b4501",
    "PAXLOVID": "20260221_8a99d6d6-fd9e-45bb-b1bf-48c7f761232a",
    "NURTEC ODT": "20260607_9ef08e09-1098-35cc-e053-2a95a90a3e1d",
}


def split_by_section_headers(text: str) -> list[TextNode]:
    matches = list(SECTION_HEADER_RE.finditer(text))
    nodes = []
    i = 0
    while i < len(matches):
        m = matches[i]
        start = m.start()
        header = m.group(1).strip()
        j = i + 1
        if header.startswith("WARNING:"):
            while j < len(matches) and not text[m.end():matches[j].start()].strip():
                header = f"{header} {matches[j].group(1).strip()}"
                m = matches[j]
                j += 1
        end = matches[j].start() if j < len(matches) else len(text)
        if not text[m.end():end].strip():  # header-only chunk: see section_router.py
            i = j
            continue
        if header.startswith("WARNING:"):
            body_lines = text[m.end():end].strip().splitlines()
            if body_lines and not body_lines[0].strip().lower().startswith("see full prescribing"):
                header = f"{header} {body_lines[0].strip()}"
        body = text[start:end].strip()
        nodes.append(TextNode(text=body, metadata={"section": header}))
        i = j
    return nodes


def resolve_section_pattern(q: str) -> str | None:
    q_lower = q.lower()
    for phrase, pattern in QUERY_PHRASE_TO_SECTION_PATTERN.items():
        if phrase in q_lower:
            return pattern
    return None


def resolve_document_filter(q: str, brands: list[str]) -> str | None:
    q_lower = q.lower()
    for brand in brands:
        if brand.lower() in q_lower:
            return brand
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
# --- end copied section ---


def main() -> None:
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))["questions"]
    embed_model = get_embedding_model()  # bge-base, the router's default

    all_nodes: list[TextNode] = []
    for brand, stem in RX_DOCS.items():
        # Reading-order extraction (not SimpleDirectoryReader): these PDFs'
        # content streams emit HIGHLIGHTS bullet text detached from its
        # section header, which was the dominant accuracy@1 failure mode in
        # the first run of this eval (see eval/RAG_ROUTING_FINDINGS.md).
        text = "\n".join(extract_pages_reading_order(RAW / f"{stem}.pdf", max_pages=2))
        nodes = split_by_section_headers(text)
        for n in nodes:
            n.metadata["document"] = brand
        all_nodes.extend(nodes)
    print(f"Indexed {len(all_nodes)} section chunks across {len(RX_DOCS)} documents")

    node_embs = [np.array(embed_model.get_text_embedding(n.get_text())) for n in all_nodes]

    brands = list(RX_DOCS.keys())
    results, n_at1, n_at3 = [], 0, 0
    t0 = time.perf_counter()
    for item in eval_set:
        q = item["question"]
        doc_filter = resolve_document_filter(q, brands)
        section_pattern = resolve_section_pattern(q)

        idxs = list(range(len(all_nodes)))
        if doc_filter:
            idxs = [i for i in idxs if all_nodes[i].metadata["document"] == doc_filter]
        if section_pattern:
            matched = [i for i in idxs if re.match(section_pattern, all_nodes[i].metadata["section"])]
            if matched:
                idxs = matched

        q_emb = np.array(embed_model.get_query_embedding(q))
        scored = sorted(
            ((cosine_similarity(q_emb, node_embs[i]), i) for i in idxs), reverse=True
        )[:3]

        top_texts = [all_nodes[i].get_text() for _, i in scored]
        ok1, score1 = is_match(item["expected_phrase"], top_texts[0])
        ok3 = ok1 or any(is_match(item["expected_phrase"], t)[0] for t in top_texts[1:])
        n_at1 += ok1
        n_at3 += ok3

        results.append({
            "id": item["id"],
            "document": item["document"],
            "question": q,
            "expected_phrase": item["expected_phrase"],
            "resolved_document": doc_filter,
            "resolved_section_pattern": section_pattern,
            "top1_section": all_nodes[scored[0][1]].metadata["section"],
            "top1_cosine": round(scored[0][0], 4),
            "top1_fuzzy_score": round(score1, 4),
            "hit_at_1": bool(ok1),
            "hit_at_3": bool(ok3),
        })
        print(f"  [{'PASS' if ok1 else 'FAIL'}@1{' PASS@3' if (not ok1 and ok3) else ''}] "
              f"{item['document']:<14} fuzzy={score1:.2f} top1=[{results[-1]['top1_section'][:50]}]")

    elapsed = time.perf_counter() - t0
    summary = {
        "accuracy_at_1": round(n_at1 / len(eval_set), 4),
        "hit_at_3": round(n_at3 / len(eval_set), 4),
        "n_questions": len(eval_set),
        "n_documents": len(RX_DOCS),
        "embedding_model": "bge-base",
        "eval_wall_seconds": round(elapsed, 1),
    }
    print(f"\naccuracy@1: {n_at1}/{len(eval_set)} = {summary['accuracy_at_1']:.0%}"
          f"   hit@3: {n_at3}/{len(eval_set)} = {summary['hit_at_3']:.0%}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(f"written: {RESULTS}")


if __name__ == "__main__":
    main()
