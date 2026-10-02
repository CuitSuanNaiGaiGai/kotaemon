# Task 7 Implementation Report — Scoped Hybrid Retrieval

## Commits

- Implementation: `6da79a438b5be8ffae9d418116c83f6359b668db` — `fix: enforce scoped retrieval boundaries`
- Base: `2eb35779111f89fce7e27d02b8524032d4636fc3`

## Changes

- `libs/kotaemon/kotaemon/indices/vectorindex.py`
  - Made `scope`, `fallback_scope`, and optional dict `trace` explicit `VectorRetrieval.run` arguments while preserving the existing `scope` call pattern and result type.
  - Vector queries now pass chunk IDs as `ids`; lexical queries pass the same IDs as `doc_ids`. A legacy `doc_ids` argument is consumed as a scope and is never forwarded to a vector backend.
  - `scope=[]` returns before embedding or store access. `scope=None` searches globally unless a caller-visible `fallback_scope` is supplied, in which case retrieval is constrained to that allowlist.
  - Scoped calls fetch a bounded expanded candidate pool, then filter by chunk ID before reranking and again after every reranker. Planned scope is intersected with caller-visible IDs. An empty scoped result may retry once against the caller allowlist; the retry reason and candidate count are recorded.
  - Corrected score-to-ID association when `doc_store.get` reorders documents and made hybrid merging deterministic and unique by `doc_id`.
  - Global lexical search runs only when the docstore declares or implements lexical query support. Empty and unavailable lexical branches are reported distinctly. Hybrid worker exceptions are recorded; usable results from the other branch are retained, and failures are surfaced when no branch returns usable results.
- `libs/kotaemon/kotaemon/storages/docstores/in_memory.py` declares its lexical search unavailable.
- `libs/ktem/ktem/index/file/pipelines.py` forwards an optional trace, retains the selected-file early return, and constrains extra-table retrieval to the selected chunk IDs.
- Added `libs/kotaemon/tests/test_knowledge_scoped_retrieval.py` and `libs/ktem/ktem_tests/test_knowledge_scoped_pipeline.py`.

## TDD evidence

### RED

Before production changes, ran:

```text
uv run pytest libs/kotaemon/tests/test_knowledge_scoped_retrieval.py libs/ktem/ktem_tests/test_knowledge_scoped_pipeline.py -q
```

Result: **14 failed, 3 passed**. Failures exposed the old vector `doc_ids` argument, empty-scope store work, absent global lexical retrieval, scope leaks, score/ID misalignment, missing dedup/fallback/trace handling, and swallowed worker errors. The adapter boundary and no-selected-files UI behavior already passed.

### GREEN

Final focused regression command:

```text
uv run pytest libs/kotaemon/tests/test_knowledge_scoped_retrieval.py libs/ktem/ktem_tests/test_knowledge_scoped_pipeline.py libs/kotaemon/tests/test_indexing_retrieval.py libs/kotaemon/tests/test_knowledge_planner.py libs/ktem/ktem_tests/test_knowledge_catalog.py -q
```

Result: **41 passed**, with 4 existing Pydantic deprecation warnings from `kotaemon/embeddings/openai.py`.

Formatting and checks passed:

```text
uv run black --check <five changed Python files>
uv run isort --profile black --check-only <five changed Python files>
uv run flake8 --ignore=E501,E203,W503 <five changed Python files>
git diff --check
```

## Limits

- Lexical availability is explicit for the in-memory store. Other legacy stores with an overridden `query` method are attempted unless they set `supports_lexical_search = False`; empty output is recorded as `empty`, not treated as evidence of BM25 coverage.
- A scoped-to-visible retry requires the caller to provide `fallback_scope`. Without a caller allowlist, the retriever does not widen a planned scope.
- The trace is intentionally a small optional dict hook for scope, candidate count, lexical status, and branch errors. Task 10 remains responsible for the full structured trace collector.
- No `Document`/`RetrievedDocument` schema change, database migration, new retrieval stack, or production recall claim was added.
