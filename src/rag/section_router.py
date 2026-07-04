import os
import re

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.llms.gemini import Gemini

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
SECTION_HEADER_RE = re.compile(r"^([A-Z][A-Z0-9 ,:/&\-]{9,60})$", re.MULTILINE)


def split_by_section_headers(text: str) -> list[TextNode]:
    matches = list(SECTION_HEADER_RE.finditer(text))
    nodes = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        header = m.group(1).strip()
        body = text[start:end].strip()
        nodes.append(TextNode(text=body, metadata={"section": header}))
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
