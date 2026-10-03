# Local mini-corpus retrieval experiment

## Decision

Use a staged evaluation. First inventory and parse the current local source corpus
into ignored draft artifacts. Build a reviewed v1 from 20–24 unique documents,
stratified by content-derived topic and file format. After v1 is reviewed and
evaluated, expand to the complete byte-unique corpus in a separately reviewed
snapshot and report the expanded run as a second experiment.

Directory names are provenance only. They must not be copied into source topics,
virtual paths, planner entities, or judgments unless a person verifies that they
match the document contents. Do not rename or modify original files. Exclude
filesystem/Office lock artifacts from indexing, deduplicate byte-identical files
by SHA-256, and preserve their paths as local provenance.

## Data preparation

Read the four input formats through the existing default ingestion mapping:
normal `PDFReader` for PDF, `UnstructuredReader` for DOCX, `TxtReader` for
Markdown, and `ExcelRowReader` (with the configured pandas fallback) for XLSX.
Use the current normal PDF extraction mode first. Record per-source and
per-page extraction quality; do not silently accept empty or unusable pages.
Put low-quality sources in a local review queue. If OCR is needed, use it as a
separately recorded parser configuration and regenerate a new snapshot rather
than mixing parser outputs invisibly.

Create one source record per unique original file. Derive stable source and chunk
IDs from source content hashes and deterministic chunk positions. Preserve
relative source paths and hashes in the local manifest. Use content-derived
document names, topics, virtual paths, and entities as draft metadata; include a
human review status for every field that affects scoping. Chunk with the
existing configured splitter (the current default is 1,024 tokens with 256
tokens of overlap) and record its exact version and parameters.

All source-derived material stays under the Git-ignored `local/` directory:
source inventory, extraction report, draft records, candidate judgments, review
evidence, snapshots, traces, and result artifacts. CI tests use generated
fixtures only. No source text, query text, gold labels, or private-corpus metric
artifacts are added to tracked files or published in the existing PR without
explicit authorization.

## Golden v1

Select 20–24 unique documents from the parsed corpus, covering each usable file
format and the substantive content topics found during extraction. Keep related
hard negatives in the same sample. Prepare roughly 20–30 candidate questions,
with the final count determined by evidence quality. Include explicit
document/entity/path questions, content questions, cross-document questions,
and ambiguous/global questions where the source material supports them.

Use chunk-level judgments throughout v1 to preserve the metric unit used by the
existing synthetic evaluation. Each question must cite local source path, page
or sheet/section, supporting text location, and all reviewed relevant chunk IDs.
Questions and labels must be created independently of planner outputs; gold
answers and judgment IDs must not be included in indexed text or metadata. Set
`disallowed_source_ids` only for sources that clearly violate an explicit scope;
irrelevance alone is not wrong scope. Use no wrong-scope label for ambiguous or
global questions. The current fixture schema rejects mixed judgment levels and
requires at least one relevant ID, so do not mix source-level labels or encode
zero-result cases in the judged v1 fixture.

Before freezing v1, present the user a local-only review package containing the
source-to-content-topic mapping, excluded/duplicate file list, extraction
quality summary, chunk previews, and per-question evidence, relevant IDs, and
wrong-scope IDs. Do not run or report gold-based metrics until the user approves
the labels. Freeze approved files under `local/snapshots/v1/` with exact-byte
SHA-256 hashes for `records.json` and `judgments.jsonl`, source hashes, schema
and parser/chunker versions, counts, and review status/date.

For the later full-corpus snapshot, re-review relevance and scope labels against
the added sources. In particular, update `disallowed_source_ids` where new
sources clearly violate a query's explicit scope. Freeze it as a new version;
never mutate v1.

## Paired K=5 experiment

Load only an explicitly selected, hash-verified local snapshot. Reuse the
existing `KnowledgeService`, `VectorRetrieval`, and paired evaluator path. Build
one deterministic local hashed-feature embedding index and share the same
immutable index, document store, records, judgments, and query order between
both arms. Compare the identity/global baseline planner with `QueryPlanner`.
Use K=5 and candidate_k=5; keep retrieval mode, embedding, seed, query,
filters, and all other configuration fixed. Disable rerankers, query rewriting,
MMR, and result extension so the planner is the only variable.

Report Hit@5, Recall@5, MRR@5, and Wrong-scope@5 with baseline, planned, signed
delta, query count, and wrong-scope denominator. Include per-query results and
traces in ignored local artifacts. The deterministic hashed-feature backend
matches the current offline harness and makes the planner comparison
reproducible; it is a controlled retrieval proxy, not a claim about a configured
production embedding model.

## Implementation and verification tasks

1. Add TDD-covered local inventory, artifact exclusion, content-hash dedup,
   parsing, quality reporting, deterministic IDs, and draft generation. Write
   only to the ignored local directory.
2. Add a snapshot loader/validator that requires a reviewed manifest, verifies
   exact payload hashes, checks all IDs/relationships, and accepts an explicit
   snapshot path without changing the synthetic default.
3. Add an offline paired K=5 runner using the existing retrieval service and
   deterministic vector adapter. Tests must prove both arms share the same
   index/configuration and differ only by planner.
4. Generate and present the local-only v1 review package. After user approval,
   freeze v1 and run the paired experiment. Repeat for the expanded snapshot
   only after its labels are reviewed.

## Acceptance criteria

- The local source tree is unchanged and remains Git-ignored.
- Repeated inventory and parsing produce stable IDs and equivalent draft
  payloads under the same parser/chunker versions.
- Artifacts and duplicate source files are accounted for, and low-quality pages
  are surfaced rather than silently omitted.
- Snapshot hash mismatch, missing review status, unknown IDs, or invalid source
  relations fail before retrieval.
- The v1 fixture has one judgment level, reviewed evidence for every relevant
  chunk, and only explicit wrong-scope labels.
- The paired report shows configuration parity except for planner and reports
  all four requested K=5 metrics with denominators and signed deltas.
- Private source-derived data and results remain local unless the user later
  explicitly approves publication.
