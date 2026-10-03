# Local mini-corpus Task 2 report

### TDD evidence

Added generated, multi-format fixtures under `tmp_path` before implementing the draft builder. The first focused run was:

```text
uv run pytest libs/kotaemon/tests/test_knowledge_eval_local_ingest.py -q
```

It exited 2 during collection with:

```text
ModuleNotFoundError: No module named 'kotaemon.indices.knowledge.evaluation.local_ingest'
```

That confirmed the absent implementation. Subsequent red runs exposed an `AttributeError` while checking reader diagnostics, `KeyError: 'ignored.py'` when the unsupported path was absent from exclusions, and propagation of `ValueError: Token splitter returned a segment outside its source unit` instead of a review issue. A later regression case also caught an invalid span endpoint (`char_end` could exceed the unit length even when Python slicing returned matching text); the implementation now derives the end from the final mapped part and validates both bounds and exact slice equality.

After implementation, the focused Task 1 inventory and Task 2 ingest suite passed:

```text
uv run pytest libs/kotaemon/tests/test_knowledge_eval_local_corpus.py libs/kotaemon/tests/test_knowledge_eval_local_ingest.py -q
8 passed, 6 warnings in 4.66s
```

The six warnings are dependency deprecations from PyMuPDF bindings and the installed cryptography provider. Formatting and whitespace checks passed:

```text
uv run black --check libs/kotaemon/kotaemon/indices/knowledge/evaluation/local_corpus.py libs/kotaemon/kotaemon/indices/knowledge/evaluation/local_ingest.py libs/kotaemon/tests/test_knowledge_eval_local_ingest.py
3 files would be left unchanged.
git diff --check
exit 0
```

### Implementation

- Added `build_local_draft` with deterministic records and chunk IDs, root-relative source provenance, empty `anchors.json`, and writes restricted to the caller-provided output directory.
- Reads only selected canonical supported sources. Byte duplicates remain in the exclusion/provenance report and have an observable test assertion that only the canonical path gets a reader diagnostic. Supported files are hashed by the inventory stage to establish content identity; unsupported files are listed by relative path without opening or hashing their contents.
- Added PDF, DOCX, Markdown, and XLSX generated fixtures covering reader locators, empty PDF pages, blank spreadsheet rows, malformed DOCX, duplicate bytes, unsupported files, and deterministic output in separate draft directories. The Task 1 Markdown-only `make_generated_corpus` contract remains unchanged.
- Records the selected parser and attempts in source configuration and quality diagnostics. In this environment, `UnstructuredReader(split_documents=True)` cannot initialize because `libmagic` is unavailable; a local `DocxReader` fallback is selected and explicitly reported as document granularity with `needs_review=True` because it coalesces paragraph text. No element locator is invented. Corrupt DOCX input remains in the report as an extraction failure.
- Records the token splitter configuration and version (`main-token-only-v1`, size 1024, overlap 256, separator, and backup separators) in the deterministic records. Chunk spans are checked for `0 <= start <= end <= len(unit)` and exact normalized-text equality. Unmappable offsets preserve the source unit and produce a review-needed quality issue rather than guessed chunks.
- Topic candidates come from extracted content only. Sources without a candidate are marked for topic review. `anchors_payload` is deterministic empty bytes because this API has no query input.
- Unsupported regular-file paths are listed without reading content. Office lock files, `.DS_Store`, and symlinks remain excluded.

### Changed files

- `docs/superpowers/plans/2026-10-03-mini-corpus-golden-experiment.md` — incorporated the Task 2 design amendment for queryless anchors, provenance, output boundaries, locators, extraction quality, and splitter configuration.
- `libs/kotaemon/kotaemon/indices/knowledge/evaluation/local_corpus.py` — added canonical-source selection and safe unsupported-path listing.
- `libs/kotaemon/kotaemon/indices/knowledge/evaluation/local_ingest.py` — added local parsing, chunk drafts, metadata, and quality reporting.
- `libs/kotaemon/tests/test_knowledge_eval_local_ingest.py` — added generated multi-format and regression tests.
- `.superpowers/sdd/local-mini-corpus-task-2-report.md` — this implementation report.

### Limitations

- The local review-draft API does not create query-linked evidence anchors; `anchors.json` is empty until reviewed questions and spans exist.
- The production CLI that enforces containment under the ignored `local/draft/` tree is outside Task 2. This API writes only beneath the explicit `output_dir` argument, and tests use temporary directories.
- DOCX parsing uses the local fallback in this environment and requires human review of paragraph-level evidence granularity.
- The prohibited `libs/kotaemon/tests/fixtures/knowledge_eval/local/sources/` tree was not accessed; all ingestion tests use generated temporary files.

### Commit

Implementation commit SHA: to be filled after committing the implementation.
