# Section-Router Findings — Batch 2 Document Extension

Findings from extending `src/rag/section_router.py` from 5 to 14 routable
documents (2026-07). Companion to the in-code experiment log in that file;
ground-truth values and per-document ambiguities live in
`data/ground_truth/*.json` notes.

## Documents in scope

14 of 16 sample documents route. Two are excluded by design, for different
reasons:

| Excluded | Why | Class |
|---|---|---|
| ADVIL (20250410) | OTC "Drug Facts" schema — none of the routed section headers exist as structural lines anywhere in it | Different vocabulary |
| PREMARIN IV (20250531) | Pre-PLR (pre-2006) label — vocabulary mostly matches, but there is no HIGHLIGHTS block, so the router's first-2-pages scoping assumption finds almost nothing | Different location |

Premarin IV's distinction matters: because only the *location* assumption
breaks, full-document scoping for pre-PLR labels is a plausible future
extension. Advil would need a genuinely different router.

## Defects found and fixed (both regression-checked against all 16 docs)

1. **Lettered boxed-warning titles fell outside `SECTION_HEADER_RE`.**
   ELIQUIS's "WARNING: (A) PREMATURE DISCONTINUATION..." never matched because
   "(" was outside the header charset — the document had no routable WARNING
   chunk at all. Fix: parens admitted only after a `WARNING:` prefix. The
   naive fix (parens in the shared charset) was rejected by the false-positive
   check: it promoted ZOLOFT's dosing-table fragment "PD, PTSD, SAD (" to a
   header, splitting real content out of its DOSAGE AND ADMINISTRATION chunk.

2. **Header-shaped wrap lines split the boxed warning's body away from its
   chunk.** ELIQUIS's title wraps onto "THROMBOTIC EVENTS" — fully ALL-CAPS,
   so it matched as a new section, leaving the WARNING chunk body-less
   (routable but content-free: correct document, correct section, plausible
   score, no actual warning text — a silent wrong answer). Fix: when a
   WARNING: match has an empty body, the next header-shaped line is treated as
   the wrapped remainder of the title. Empty body is the trigger because a
   genuine new section never begins zero characters after a WARNING: title.
   Regression check: only ELIQUIS's chunking changes across all 16 documents.

## Known limitations (documented deliberately, not built around)

1. **Retrieval cannot express "this section does not exist."** BENEFIX has no
   DRUG INTERACTIONS section anywhere (biologic label; interactions content
   simply absent). A drug-interactions query resolves the section pattern,
   finds zero matching chunks, and falls back — by design — to unfiltered
   within-document similarity, returning adjacent-but-wrong chunks
   (HIGHLIGHTS, WARNINGS AND PRECAUTIONS) at deceptively healthy scores
   (0.65–0.68). The honest answer would be "no such section"; nearest-neighbor
   retrieval structurally cannot say that. A future fix is an explicit
   no-match signal when a section filter resolves but matches nothing, letting
   the caller distinguish "answered from the right section" from "fell back."
   One document / one section type today, so documented rather than built.

2. **Short-chunk score bias, second sighting.** For "What is the recommended
   dosage for ZOLOFT?", DOSAGE FORMS AND STRENGTHS (short) outranks DOSAGE
   AND ADMINISTRATION (long, and the actually-responsive section) 0.8007 vs
   0.7380. Same phenomenon as the CONTRAINDICATIONS-beats-boxed-warning
   investigation in the in-code experiment log — short/generic chunks score
   competitively against longer, specific ones — now observed in a second
   document. PREVNAR 20's identical section pair ranks correctly, so it is
   marginal, not systematic. Both `^DOSAGE`-prefixed sections are retrieved
   (top-2), so the responsive content is still returned, just not ranked #1.

## Noise chunks that are tolerated (inert by construction)

- TOC-duplicate WARNING chunks (ZOLOFT, DEPO-PROVERA, PAXLOVID, CHANTIX):
  the contents page repeats the boxed-warning line; similarity ranking picks
  among duplicates.
- Labeler-name pseudo-headers (NORVASC, VIAGRA: "PFIZER LABORATORIES DIV
  PFIZER INC" is ALL-CAPS and header-shaped): no routed pattern matches them.

## Held-out retrieval eval (2026-07-06) — superseded by the extraction fix below

14 questions, one per routable document, locked before the first run
(`eval/rag_eval_set.json`), scored against the top-1 retrieved chunk with the
existing fuzzy-match scorer (`eval/rag_retrieval_eval.py`; results in
`eval/results/rag_retrieval_eval.json`).

| Metric | Result |
|---|---|
| Section routing correctness (right section chunk chosen) | **14/14 = 100%** |
| End-to-end accuracy@1 (expected phrase in the routed chunk) | **5/14 = 36%** |

The gap between those two numbers is the finding. Every failure was verified
individually: in all 9, the expected phrase exists at fuzzy >= 0.99 in the
first-2-page corpus — but in the *wrong* chunk. PDF extraction reading order
detaches HIGHLIGHTS bullet content from its header (two-column layout), so
e.g. NORVASC's contraindication text lands in the DRUG INTERACTIONS chunk and
VIAGRA's interaction text lands in the table-of-contents chunk. The router
correctly routes to a section whose content has been displaced elsewhere.

Implication: the retrieval bottleneck is not ranking (embeddings) and not
routing (section filter) — it is chunk *assembly* upstream, i.e. reassociating
displaced bullet content with its section header at extraction time. This was
already documented as a single-field curiosity in LIPITOR's ground-truth notes
("column/reading-order artifact"); the eval shows it is the dominant failure
mode across the corpus, affecting 9 of 14 documents' primary sections.
hit@3 equals hit@1 (36%) because section filtering excludes the displaced
chunks from the candidate set entirely — better ranking cannot recover this.

Honest-number note: 36% is the defensible end-to-end figure. Quoting the 100%
routing figure alone would overstate the system; quoting 36% without the
diagnosis would understate what works.

## Reading-order extraction fix (2026-07-16)

### The diagnosis above was half right

The eval section above attributed the displaced content to a two-column
layout. Verifying the mechanism against rendered pages before fixing anything
showed that is wrong in its specifics: **every one of the 32 first-2-pages in
this corpus is single-column** (measured: no page has more than a handful of
lines confined to each half with few midline-crossers; most lines span the
full width). The real mechanism is **PDF content-stream order**: these
SPL-generated PDFs emit section headers and bullet glyphs first and append
the bullet body text at the end of the stream, and both loaders in use
(SimpleDirectoryReader/pypdf and unsorted PyMuPDF) preserve stream order, not
visual order. NORVASC page 0 renders in perfect reading order but emits its
CONTRAINDICATIONS body line as block 34 of 37 — after DRUG INTERACTIONS
(block 22). Symptom identical to column interleaving; cause entirely
different.

The same measurement explains why this never surfaced during development:
CHANTIX is the one document in the corpus with **zero** out-of-order lines
(LIPITOR page 0 has 39 of 68 lines displaced; every other document has
double-digit displacement on at least one of its first two pages). The
document everything was built against was the one document without the bug.

### The fix

`extract_pages_reading_order()` in `src/extract/pymupdf_extractor.py`:
line-level extraction sorted by visual position (top-to-bottom, then
left-to-right), with same-baseline fragments merged into one row so bullet
glyphs rejoin their text. Wired into `load_highlights_section_nodes` and the
eval loader in place of SimpleDirectoryReader. No column handling is built,
deliberately: the corpus measurably contains no multi-column pages, so column
banding would be speculative code with zero positive examples. If
multi-column documents ever enter the corpus, the sort would interleave them
— re-run the layout scan and add banding first.

### The one regression, and the fourth short-chunk sighting

First re-run: 13/14 accuracy@1 — every previous failure fixed, but ELIQUIS
(previously passing) flipped. Cause: ELIQUIS's contents page repeats the
boxed-warning title, and under reading order that duplicate becomes a bare
63-char header-only chunk, which outranked the real 879-char WARNING chunk
0.6931 vs 0.6355 — the short-chunk score bias already logged twice, now in
its most degenerate form (a chunk with literally no content winning). Fix:
`split_by_section_headers` now drops chunks whose body is empty beyond the
header line — a content-free chunk can never contain an answer, only steal
rank from one that does.

### Results (same locked 14-question set, bge-base)

| Metric | Before | After |
|---|---|---|
| Section routing correctness | 14/14 = 100% | 14/14 = 100% |
| End-to-end accuracy@1 | 5/14 = 36% | **14/14 = 100%** |
| hit@3 | 5/14 = 36% | **14/14 = 100%** |

ZOLOFT's short-chunk limitation (above) also resolved: with its dosage
content correctly assembled under DOSAGE AND ADMINISTRATION, that section now
ranks #1 for the dosage query.

### Regression check across all 16 documents

Chunk-inventory diff, old vs. new loading, all 16 documents. No real section
was lost anywhere. Three classes of *noise* chunks disappeared, each with a
confirmed mechanism, none accidental:

1. **"FULL PRESCRIBING INFORMATION: CONTENTS" pseudo-sections (13 docs)** —
   pypdf silently dropped the trailing asterisk from "CONTENTS*"; PyMuPDF
   preserves it, and "*" is outside the header charset, so the TOC line never
   header-matches. Noise removed by higher extraction fidelity.
2. **PREMARIN IV's four stray "CLINICAL STUDIES" pseudo-sections** — these
   were fragments of "(See CLINICAL STUDIES and WARNINGS...)" cross-reference
   sentences that stream-order extraction had split onto their own lines; the
   same-baseline row merge reassembles the sentences, so no header-shaped
   fragment survives. (PREMARIN IV remains excluded from routing regardless.)
3. **ELIQUIS's TOC-duplicate WARNING chunk** — dropped by the header-only
   filter above. TOC-duplicate WARNING chunks in ZOLOFT/DEPO-PROVERA/
   PAXLOVID/CHANTIX retain TOC body text and survive; they remain inert
   (similarity picks among duplicates) as before.

Caveat unchanged from the original eval: 14 questions, one per document,
top-1 scoring against a locked set. 100% on this set means the displaced-
content failure mode is gone, not that retrieval is solved in general — the
BENEFIX "section does not exist" limitation above still stands.

## Operational note

Running `python src/rag/section_router.py` from the repo root fails with
`ModuleNotFoundError: No module named 'src'` — the `from src...` imports need
the repo root on `sys.path`. Use `PYTHONPATH=. python src/rag/section_router.py`
(or run via an IDE configuration that sets the working directory as source
root).
