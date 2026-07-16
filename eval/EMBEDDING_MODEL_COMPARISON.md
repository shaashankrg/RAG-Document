# Embedding Model Comparison — bge-base vs. MiniLM

Formal writeup of the two-model comparison run during the boxed-warning
retrieval investigation (`src/rag/section_router.py`, Experiments 4–5), plus
directly measured latency. Retrieval-quality numbers are from the recorded
experiment output (CHANTIX HIGHLIGHTS, 13 section chunks, 4 topically distinct
queries, unfiltered similarity search, top-3 reported); latency was measured
on this machine (CPU) with warmed models.

## Models

| | bge-base | MiniLM |
|---|---|---|
| HF id | `BAAI/bge-base-en-v1.5` | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dim | 768 | 384 |
| Query embed (ms/ea, CPU) | 82.6 | 25.0 |
| Chunk embed (ms/ea, CPU) | 169.1 | 53.0 |
| First-call overhead (ms) | 245 | 31 |

MiniLM is ~3.3x faster per embed and its index vectors take half the memory
(384 vs 768 floats/chunk). Chunk counts are identical under both models —
chunking happens before embedding and is model-independent (the "chunk count"
column some comparisons include is a non-axis here).

## Retrieval quality (4 cross-queries, unfiltered, CHANTIX)

Top-1 correctness — did the topically right section chunk rank #1?

| Query targets | bge-base | MiniLM |
|---|---|---|
| CONTRAINDICATIONS | correct (0.8442) | correct (0.8099) |
| DOSAGE AND ADMINISTRATION | correct (0.7645) | correct (0.7425) |
| INDICATIONS AND USAGE | correct (0.7822) | correct (0.8031) |
| WARNING: (boxed warning) | **wrong** — CONTRAINDICATIONS 0.6776 edges WARNING 0.6718 | **wrong** — WARNING chunk not even top-3 |

Both models: 3/4. The shared failure is the structural lexical-gap case that
motivated the section router: "boxed warning" shares no vocabulary with the
warning's actual text (agitation, suicidality, neuropsychiatric), so no
embedding model was expected to fix it — and neither did. Degree differs
though: bge-base missed by 0.006 with the right chunk at #2, while MiniLM
dropped the right chunk out of the top 3 entirely. On this (small) evidence
bge-base degrades more gracefully on vocabulary-gap queries.

## Interaction with the section router

The filter-then-score design changes what the embedding model is for: after
section filtering, the model only ranks *within* an already-correct candidate
set (often 1–2 chunks), so top-1 model quality matters much less than in
unfiltered mode. Under the router, both models produce the same section
choices on the queries tested; the cosine scores differ but the ranking of
the filtered candidates did not.

## Conclusion

Keeping **bge-base** as the default: retrieval is offline/batch in this
project, so 3.3x CPU latency is not binding, and its graceful degradation on
vocabulary-gap queries is worth having wherever the section filter doesn't
apply (queries with no mapped section phrase fall back to unfiltered search).
**MiniLM is a legitimate swap** if embed latency or index memory becomes a
constraint — with the caveat that unfiltered fallback queries will lean
harder on it, and that is exactly where it showed the sharper failure.

Caveats: quality evidence is one document and four queries; latency is one
machine, CPU-only, batch size 1. The registry also contains `bge-small` and
`mpnet` — neither has been tested; nothing here speaks to them.
