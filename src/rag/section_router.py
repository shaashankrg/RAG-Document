import os
import re

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.llms.gemini import Gemini

from src.extract.pymupdf_extractor import extract_pages_reading_order
from src.rag.embeddings import get_embedding_model

load_dotenv()  # reads GEMINI_API_KEY from .env

# Load PDF or text document
documents = SimpleDirectoryReader(
    input_files=["C:\\Users\\shash\\pfizer-doc-intelligence\\data\\raw\\20140820_2468ba8d-4c77-4ea0-88d8-b64497a72222.pdf"]
).load_data()
print(f"Loaded {len(documents)} documents.")

# Chunk -> embed -> store, then wrap the index as something queryable.
# This project has no OPENAI_API_KEY (uses Mistral/Gemini/HF instead), so the
# default OpenAI embedding model won't work here -- use the local HF embedding
# model this project already sets up in src/rag/embeddings.py.
index = VectorStoreIndex.from_documents(documents, embed_model=get_embedding_model())

# Same reasoning applies to the query engine's LLM -- default is OpenAI. Gemini's
# own default env var is GOOGLE_API_KEY, but this project's .env convention is
# GEMINI_API_KEY (see .env.example), so pass it explicitly rather than rely on
# the mismatched default.
llm = Gemini(model="models/gemini-flash-latest", api_key=os.getenv("GEMINI_API_KEY"))
query_engine = index.as_query_engine(llm=llm)

query = "What is CHANTIX indicated for, and what is the boxed warning about?"
response = query_engine.query(query)
print(f"Query: {query}")
print(f"Answer: {response}")

# Inspect what actually got retrieved and sent to the LLM, rather than just
# trusting the final answer.
print(f"\nRetrieved {len(response.source_nodes)} chunk(s):")
for i, node in enumerate(response.source_nodes):
    print(f"\n--- chunk {i + 1} (score: {node.score:.4f}) ---")
    print(node.node.get_text())

# Collected across every retrieval call below, to test the length-vs-score
# hypothesis: does a shorter chunk systematically score higher regardless of
# topical relevance, independent of which query or chunking strategy was used?
length_score_data: list[tuple[int, float, str]] = [
    (len(node.node.get_text()), node.score, "compound (initial)") for node in response.source_nodes
]


def print_retrieval(label: str, q: str, top_k: int) -> None:
    """Retrieval only, no LLM call -- for inspecting ranking/scores directly."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(q)
    print(f"\n=== {label} (top_k={top_k}) ===")
    print(f"Query: {q}")
    for i, node in enumerate(nodes):
        text = node.node.get_text()
        preview = text[:150].replace("\n", " ")
        print(f"  chunk {i + 1} (score: {node.score:.4f}, len: {len(text)}): {preview}...")
        length_score_data.append((len(text), node.score, label))


# Experiment 1: does the compound query dilute ranking vs. two single-intent
# queries? If so, the score gap between the "right" chunk (HIGHLIGHTS) and the
# "lucky" chunk (Adverse Reactions) should widen once the intents aren't blended.
print_retrieval("Compound query", query, top_k=2)
print_retrieval("Single-intent: indication", "What is CHANTIX indicated for?", top_k=2)
print_retrieval("Single-intent: boxed warning", "What is the boxed warning for CHANTIX about?", top_k=2)

# Experiment 2: does top_k=1 on the original compound query actually fail the
# way the score gap predicted -- returning only the Adverse Reactions chunk and
# missing the HIGHLIGHTS chunk that actually contains the indication/boxed warning?
print_retrieval("Compound query, top_k=1", query, top_k=1)


# Experiment 3: default chunking never ranked the HIGHLIGHTS chunk #1 for either
# query, regardless of phrasing -- the leading hypothesis is that the chunk is
# too heterogeneous (dosing, contraindications, warnings, and indication all
# mixed into one chunk) for its embedding to represent any single topic well.
# Test that directly: split the HIGHLIGHTS pages at section-header boundaries
# (e.g. WARNING:, INDICATIONS AND USAGE) so each topic gets its own chunk, and
# see whether an isolated boxed-warning chunk now ranks #1 for that query.
# The 60-char cap was sized against CHANTIX's boxed warning title alone
# ("WARNING: SERIOUS NEUROPSYCHIATRIC EVENTS", 41 chars) and broke on the
# first document extended to: XELJANZ's title is 65 chars on its own line
# ("WARNING: SERIOUS INFECTIONS, MORTALITY, MALIGNANCY, MAJOR ADVERSE"), so
# it silently failed to match at all -- confirmed by direct measurement, not
# a guess. Widened with real margin rather than bumped to the exact number.
#
# Parentheses are allowed ONLY after a WARNING: prefix. ELIQUIS's lettered
# two-part boxed warning ("WARNING: (A) PREMATURE DISCONTINUATION...") never
# matched at all because "(" was outside the charset -- so the document had
# no routable WARNING chunk despite genuinely having a boxed warning. The
# obvious fix (adding parens to the shared charset) was false-positive-checked
# against all 16 documents' first-2-page text and promoted exactly one junk
# header: ZOLOFT's dosing-table fragment "PD, PTSD, SAD (", which would have
# split real content out of its DOSAGE AND ADMINISTRATION chunk. Scoping the
# parens to WARNING:-prefixed lines keeps the fix and drops that regression
# (re-checked: only the two ELIQUIS WARNING lines change across all 16 docs).
SECTION_HEADER_RE = re.compile(
    r"^(WARNING:[A-Z0-9 ,:/&\-()]{2,112}|[A-Z][A-Z0-9 ,:/&\-]{9,120})$", re.MULTILINE
)


def split_by_section_headers(text: str) -> list[TextNode]:
    matches = list(SECTION_HEADER_RE.finditer(text))
    nodes = []
    i = 0
    while i < len(matches):
        m = matches[i]
        start = m.start()
        header = m.group(1).strip()
        j = i + 1

        # A boxed-warning title can also wrap onto a line that IS itself
        # header-shaped (ELIQUIS: "WARNING: (A) PREMATURE DISCONTINUATION OF
        # ELIQUIS INCREASES THE RISK OF" wraps to "THROMBOTIC EVENTS", fully
        # ALL-CAPS). Without this merge, the WARNING chunk ends at the wrap
        # line with an EMPTY body -- routable but content-free, a silent
        # wrong answer -- and the real warning prose lands in a bogus
        # "THROMBOTIC EVENTS" section the ^WARNING: filter can never see.
        # An empty body is the trigger: a genuine new section never begins
        # zero characters after a WARNING: title line. Verified against all
        # 16 documents: only ELIQUIS's chunks change under this merge.
        if header.startswith("WARNING:"):
            while j < len(matches) and not text[m.end():matches[j].start()].strip():
                header = f"{header} {matches[j].group(1).strip()}"
                m = matches[j]
                j += 1

        end = matches[j].start() if j < len(matches) else len(text)

        # Discard chunks that contain nothing but their own header line. A
        # content-free chunk can never contain an answer, but it can still
        # WIN retrieval: after the reading-order extraction fix, ELIQUIS's
        # TOC-duplicated boxed-warning title became a bare 63-char chunk that
        # outranked the real 879-char WARNING chunk (cosine 0.6931 vs 0.6355)
        # -- the same short-chunk score bias already logged twice (CHANTIX
        # CONTRAINDICATIONS, ZOLOFT DOSAGE FORMS), now in its most degenerate
        # form. Dropping it lets the section filter fall through to chunks
        # that actually have content.
        if not text[m.end():end].strip():
            i = j
            continue

        # A boxed-warning title can also wrap onto a second line that isn't
        # fully ALL-CAPS (XELJANZ: "...MAJOR ADVERSE\nCARDIOVASCULAR EVENTS,
        # and THROMBOSIS" -- lowercase "and"), so that continuation line never
        # matches SECTION_HEADER_RE on its own and would otherwise be silently
        # dropped from the label (though it's still present in the chunk body
        # either way). Merge it into the label when it directly precedes the
        # standard FDA boilerplate line that always follows a boxed warning's
        # title.
        if header.startswith("WARNING:"):
            body_lines = text[m.end():end].strip().splitlines()
            if body_lines and not body_lines[0].strip().lower().startswith("see full prescribing"):
                header = f"{header} {body_lines[0].strip()}"

        body = text[start:end].strip()
        nodes.append(TextNode(text=body, metadata={"section": header}))
        i = j
    return nodes


highlights_text = documents[0].text + "\n" + documents[1].text
section_nodes = split_by_section_headers(highlights_text)

print(f"\n=== Section-header chunking produced {len(section_nodes)} chunks ===")
for node in section_nodes:
    print(f"  [{node.metadata['section']}] ({len(node.text)} chars)")

section_index = VectorStoreIndex(section_nodes, embed_model=get_embedding_model())
section_retriever = section_index.as_retriever(similarity_top_k=3)
section_results = section_retriever.retrieve("What is the boxed warning for CHANTIX about?")

print("\n=== Section-chunked retrieval: boxed warning query ===")
for i, node in enumerate(section_results):
    text = node.node.get_text()
    print(f"  chunk {i + 1} (score: {node.score:.4f}, len: {len(text)}) [{node.node.metadata['section']}]:")
    print(f"    {text[:150]}...")
    length_score_data.append((len(text), node.score, "section-chunked"))


# Length-vs-score correlation, across every retrieval call above: does a
# shorter chunk systematically score higher regardless of relevance? A strong
# negative correlation (shorter = higher score) would support that; a weak or
# inconsistent one means length isn't the driver and points at the vocabulary-
# overlap hypothesis instead.
import numpy as np

print(f"\n=== Length vs. score, {len(length_score_data)} data points ===")
for length, score, label in sorted(length_score_data, key=lambda x: x[1], reverse=True):
    print(f"  score={score:.4f}  len={length:>5}  ({label})")

lengths = np.array([d[0] for d in length_score_data])
scores = np.array([d[1] for d in length_score_data])
correlation = np.corrcoef(lengths, scores)[0, 1]
print(f"\nPearson correlation (length vs. score): {correlation:.3f}")


# Experiment 4: is CONTRAINDICATIONS a generically "central" chunk that scores
# well against almost any query, or did it just happen to win the one query
# tested so far? Run several topically distinct queries against the same
# section-chunked index and check whether the #1 result keeps being the same
# short chunk regardless of what's actually being asked.
cross_queries = [
    "What are the contraindications for CHANTIX?",
    "What is the recommended dosage for CHANTIX?",
    "What is CHANTIX used for?",
    "What is the boxed warning for CHANTIX about?",
]


def run_cross_query_check(label: str, retriever) -> None:
    print(f"\n=== Cross-query check: {label} ===")
    for q in cross_queries:
        results = retriever.retrieve(q)
        print(f"  Query: {q!r}")
        for i, node in enumerate(results):
            text = node.node.get_text()
            print(f"    #{i + 1}: [{node.node.metadata['section']}] score={node.score:.4f} len={len(text)}")


run_cross_query_check("bge-base (original embedding model)", section_retriever)

# Experiment 5: swap the embedding model entirely and repeat the same check.
# If a short/generic chunk still wins most queries under a different
# architecture, that's evidence of an embedding-space-geometry effect (or a
# property of very short clinical-safety text in general), not a bge-base-
# specific quirk. If the ranking changes, that points the other way.
alt_nodes = split_by_section_headers(highlights_text)  # fresh nodes, avoid shared state across indices
alt_index = VectorStoreIndex(alt_nodes, embed_model=get_embedding_model("minilm"))
alt_retriever = alt_index.as_retriever(similarity_top_k=3)

run_cross_query_check("minilm (different embedding model)", alt_retriever)


# Experiment 6: the anomaly is a clean structural failure, not a fuzzy edge
# case -- "boxed warning" doesn't lexically overlap with the actual warning
# text (agitation, suicidality, neuropsychiatric), so no amount of better
# embedding will reliably fix it. But FDA HIGHLIGHTS section headers are a
# small, fixed, predictable vocabulary, so this is a solvable lookup problem:
# map common query phrasings to the actual header (or header pattern, since a
# boxed warning's title varies per drug -- "WARNING: <TITLE>" is the fixed
# part), filter candidates to matching sections first, and only then rank by
# similarity (still useful for picking among multiple matches, e.g. this
# document has the WARNING section appearing twice).
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


def resolve_section_pattern(q: str) -> str | None:
    q_lower = q.lower()
    for phrase, pattern in QUERY_PHRASE_TO_SECTION_PATTERN.items():
        if phrase in q_lower:
            return pattern
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def filtered_retrieve(q: str, nodes: list[TextNode], embed_model, top_k: int = 3):
    pattern = resolve_section_pattern(q)
    candidates = nodes
    if pattern:
        matched = [n for n in nodes if re.match(pattern, n.metadata["section"])]
        if matched:
            candidates = matched

    query_emb = np.array(embed_model.get_query_embedding(q))
    scored = []
    for node in candidates:
        node_emb = np.array(embed_model.get_text_embedding(node.get_text()))
        scored.append((cosine_similarity(query_emb, node_emb), node))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k], pattern


print("\n=== Filter-then-score retrieval (section-alias lookup + cosine rank) ===")
bge_embed_model = get_embedding_model()
for q in cross_queries:
    results, pattern = filtered_retrieve(q, section_nodes, bge_embed_model)
    filter_note = f"filtered to sections matching {pattern!r}" if pattern else "no filter matched, full-index search"
    print(f"  Query: {q!r}  ({filter_note})")
    for i, (score, node) in enumerate(results):
        print(f"    #{i + 1}: [{node.metadata['section']}] score={score:.4f}")


# === Multi-document extension ===
# Everything above only covers CHANTIX. Extend section-based routing to the
# other 4 prescription-format documents in this project's sample set --
# LIPITOR, IBRANCE, XELJANZ, PREVNAR 20 -- all confirmed (by direct header
# search across each document's full text) to share the same SPL section-
# header vocabulary CHANTIX uses: CONTRAINDICATIONS, DOSAGE AND
# ADMINISTRATION, INDICATIONS AND USAGE always present and identically
# phrased; WARNING: present only where a boxed warning genuinely exists
# (XELJANZ has one, titled differently -- the prefix match already handles
# that) and correctly absent otherwise (LIPITOR, IBRANCE, PREVNAR 20 have none).
#
# ADVIL is deliberately excluded. It's an OTC "Drug Facts" label, not an SPL
# prescribing-information document: a direct header search found none of the
# four section headers anywhere in it as a structural line -- only "warning:"
# as lowercase inline prose ("Stomach bleeding warning: ..."). That's a
# different label schema entirely (Uses/Directions/inline warnings), not a
# phrasing variant this lookup could be extended to cover. Supporting it would
# need its own investigation of the Drug Facts format, not an extension of
# this one -- out of scope here.
#
# PREMARIN IV (20250531_87e2da8d) is deliberately excluded for the same class
# of reason, with a different mechanism: it's a pre-PLR (pre-2006 Physician
# Labeling Rule) label with NO "HIGHLIGHTS OF PRESCRIBING INFORMATION" block,
# so this router's core scoping assumption -- first 2 pages contain every
# section-labeled summary -- is false for it. Audited directly: its first 2
# pages expose only the boxed warning plus stray CLINICAL STUDIES cross-
# reference lines; INDICATIONS AND USAGE / DOSAGE AND ADMINISTRATION /
# CONTRAINDICATIONS / ADVERSE REACTIONS exist but deep in the document body,
# and DRUG INTERACTIONS doesn't exist at all (old format nests interactions
# under PRECAUTIONS). Unlike ADVIL the section *vocabulary* mostly matches --
# it's the *location* assumption that breaks -- so full-document scoping for
# pre-PLR labels is a plausible future extension, but that's a separate design
# decision, not a patch to slip in here.
RX_DOCS = {
    "CHANTIX": "20140820_2468ba8d-4c77-4ea0-88d8-b64497a72222",
    "LIPITOR": "20240627_a60cc18b-0631-4cf0-b021-9f52224ece65",
    "IBRANCE": "20260624_e0e6412f-50b4-4fd4-9364-62818d121a07",
    "XELJANZ": "20260329_68e3d6b2-7838-4d2d-a417-09d919b43e13",
    "PREVNAR 20": "20250720_d4e2cf51-e6a8-4103-bb1d-6120c6474ff8",
    # Batch 2 (2026-07): the 9 routable new documents. PREMARIN IV and ADVIL
    # excluded per the comments above; audit that admitted these is in
    # data/ground_truth/*.json notes and the step-2 header audit.
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


def load_highlights_section_nodes(stem: str, brand: str) -> list[TextNode]:
    """Load a document's first 2 pages (the FDA HIGHLIGHTS section, where every
    section header and every field scored in the OCR benchmark lives -- same
    scoping used throughout this project) and split at section-header
    boundaries, tagging each chunk with which document it came from.

    2026-07: switched from SimpleDirectoryReader to reading-order extraction.
    The held-out eval showed 9/14 questions failing because these PDFs'
    content streams emit HIGHLIGHTS bullet text out of visual order (headers
    and bullet glyphs first, bullet bodies appended at the stream's end), so
    stream-order text places content under the wrong section header --
    NORVASC's contraindication line landed in the DRUG INTERACTIONS chunk.
    Verified against rendered pages: single-column layout, scrambled stream.
    The CHANTIX experiments above are unaffected (measured: CHANTIX is the
    one document with zero out-of-order lines, which is why this never
    surfaced during single-document development).
    """
    path = f"C:\\Users\\shash\\pfizer-doc-intelligence\\data\\raw\\{stem}.pdf"
    text = "\n".join(extract_pages_reading_order(path, max_pages=2))
    nodes = split_by_section_headers(text)
    for node in nodes:
        node.metadata["document"] = brand
    return nodes


def resolve_document_filter(q: str, brands: list[str]) -> str | None:
    q_lower = q.lower()
    for brand in brands:
        if brand.lower() in q_lower:
            return brand
    return None


def filtered_retrieve_multi_doc(
    q: str, nodes: list[TextNode], embed_model, brands: list[str], top_k: int = 3
):
    """Same filter-then-score approach as filtered_retrieve, extended with a
    document filter: narrow to the named drug first (if the query names one),
    then to the matching section, then rank what's left by similarity.
    """
    doc_filter = resolve_document_filter(q, brands)
    section_pattern = resolve_section_pattern(q)

    candidates = nodes
    if doc_filter:
        candidates = [n for n in candidates if n.metadata["document"] == doc_filter]
    if section_pattern:
        section_matched = [n for n in candidates if re.match(section_pattern, n.metadata["section"])]
        if section_matched:
            candidates = section_matched

    query_emb = np.array(embed_model.get_query_embedding(q))
    scored = []
    for node in candidates:
        node_emb = np.array(embed_model.get_text_embedding(node.get_text()))
        scored.append((cosine_similarity(query_emb, node_emb), node))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k], doc_filter, section_pattern


print("\n\n" + "=" * 70)
print("MULTI-DOCUMENT EXTENSION: section routing across all 5 Rx documents")
print("=" * 70)

all_rx_nodes: list[TextNode] = []
for brand, stem in RX_DOCS.items():
    doc_nodes = load_highlights_section_nodes(stem, brand)
    print(f"  {brand}: {len(doc_nodes)} section chunks")
    all_rx_nodes.extend(doc_nodes)

multi_doc_queries = [
    "What is the boxed warning for XELJANZ about?",
    "What are the contraindications for LIPITOR?",
    "What is the recommended dosage for PREVNAR 20?",
    "What is IBRANCE used for?",
    "What is the boxed warning for CHANTIX about?",
    # Batch 2 queries. The first exercises the regex fix (ELIQUIS's lettered
    # "(A)/(B)" boxed warning title previously matched nothing); the BENEFIX
    # one deliberately targets a section that document does not have anywhere,
    # to observe the designed fallback (no section match -> unfiltered
    # similarity search within the document) rather than assume it behaves.
    "What is the boxed warning for ELIQUIS about?",
    "What are the drug interactions for BENEFIX?",
    "What is the recommended dosage for ZOLOFT?",
    "What are the contraindications for PAXLOVID?",
    "What is NURTEC ODT used for?",
    "What is the boxed warning for DEPO-PROVERA about?",
    # Coverage completion: the last 3 batch-2 documents unexercised by any
    # query above -- a clean header audit and clean routing under a real
    # query are different claims.
    "What is ZITHROMAX indicated for?",
    "What are the contraindications for NORVASC?",
    "What are the drug interactions for VIAGRA?",
]

print(f"\n=== Multi-document filtered retrieval, {len(all_rx_nodes)} total chunks ===")
rx_brands = list(RX_DOCS.keys())
for q in multi_doc_queries:
    results, doc_filter, section_pattern = filtered_retrieve_multi_doc(
        q, all_rx_nodes, bge_embed_model, rx_brands
    )
    print(f"\n  Query: {q!r}")
    print(f"    resolved document={doc_filter!r}, section pattern={section_pattern!r}")
    for i, (score, node) in enumerate(results):
        print(f"    #{i + 1}: [{node.metadata['document']} / {node.metadata['section']}] score={score:.4f}")
