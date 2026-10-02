# Task 6 Implementation Report — Deterministic Scope Planner and Catalog

## Commits

- Implementation: `8691e88765890c9df7c71343b04373bb45f8f461` — `feat: add deterministic knowledge retrieval planning`
- Base: `79770d372f9de5273ed8e3d83e2e1f84b21190cd`

## Files and interfaces

- `libs/kotaemon/kotaemon/indices/knowledge/planning/retrieval_plan.py`
  - Adds immutable `RetrievalPlan`, with JSON-compatible `to_dict()` output.
  - `source_ids=None` means uncertain/global planning; `source_ids=()` is an explicit empty scope.
- `libs/kotaemon/kotaemon/indices/knowledge/planning/query_planner.py`
  - Adds `KnowledgeSource`, the `SourceCatalog` protocol, logical-path validation, and deterministic `QueryPlanner.plan(...)`.
  - Exact, unambiguous entity/path/document references resolve to source IDs. Duplicate references, unknown names, and incomplete metadata fall back to global planning while keeping the caller allowlist available to the consumer.
  - Explicit path, source-type, and entity constraints intersect with each other and with caller-allowed source IDs. Unsupported source types and filter keys raise `ValueError` instead of silently weakening the request.
  - `semantic_query` remains identical to the original query.
- `libs/ktem/ktem/index/file/knowledge_catalog.py`
  - Adds SQL-backed `KnowledgeCatalog.list_sources`, `resolve_source_ids`, and `chunk_ids` over the existing `Source` and `Index` models.
  - Private indexes require a user ID and apply `Source.user == user_id`. Any caller allowlist is an additional SQL `IN` constraint. Chunk lookup joins `Index` to `Source` and applies the private-user condition in the same query.
  - Reads logical paths only from `note["knowledge"]["virtual_path"]`; it never reads `Source.path` or accesses the filesystem.
  - Missing or malformed notes produce entries without canonical fields. Only supported source types, safe logical paths, simple document names, and scalar entity values are used for planning.
- `libs/kotaemon/tests/test_knowledge_planner.py`
  - Covers Zhang San exact routing, Zhang San versus Zhang Sanfeng, longest known Chinese entity, duplicate names across branches, exact path segment matching, explicit intersection, unchanged semantic query, serialization, unknown types, empty filters, empty scope, legacy/malformed metadata, and empty catalog behavior.
- `libs/ktem/ktem_tests/test_knowledge_catalog.py`
  - Uses temporary SQLite `Source` and `Index` tables to verify public/private visibility, explicit allowlist intersection, exact path/type/entity resolution, document-only chunk mapping, malformed notes, no-user private denial, and separation between logical path and upload `Source.path`.

## TDD evidence

### RED

Before implementation, ran:

```text
uv run pytest libs/kotaemon/tests/test_knowledge_planner.py libs/ktem/ktem_tests/test_knowledge_catalog.py -q
```

Collection failed as expected because `kotaemon.indices.knowledge.planning` and `ktem.index.file.knowledge_catalog` did not yet exist (`ModuleNotFoundError`).

### GREEN

After implementation, ran:

```text
uv run pytest libs/kotaemon/tests/test_knowledge_metadata.py libs/kotaemon/tests/test_knowledge_planner.py libs/ktem/ktem_tests/test_knowledge_catalog.py libs/kotaemon/tests/test_indexing_retrieval.py -q
```

Result: **26 passed**, with 4 existing `PydanticDeprecatedSince20` warnings from `libs/kotaemon/kotaemon/embeddings/openai.py`.

Formatting and whitespace checks passed:

```text
uv run black --check libs/kotaemon/kotaemon/indices/knowledge/planning libs/kotaemon/tests/test_knowledge_planner.py libs/ktem/ktem/index/file/knowledge_catalog.py libs/ktem/ktem_tests/test_knowledge_catalog.py
uv run isort --profile black --check-only libs/kotaemon/kotaemon/indices/knowledge/planning libs/kotaemon/tests/test_knowledge_planner.py libs/ktem/ktem/index/file/knowledge_catalog.py libs/ktem/ktem_tests/test_knowledge_catalog.py
git diff --check
```

## Self-review and limits

- No `Document` or `RetrievedDocument` schema changes, migrations, parser dependencies, Wiki connector, vector database, or parallel retrieval stack were added.
- Empty catalogs and an empty caller allowlist yield `source_ids=()` so a later caller cannot mistake “no visible sources” for unrestricted recall. Non-empty catalogs with uncertain queries use global planning; the retrieval caller must still apply its separate allowed-source boundary.
- SQL visibility follows the current file-index behavior: `Source.user` is enforced only when the index is configured private. Public indexes remain public, while explicit caller IDs are still intersected in either mode.
- Chinese has no whitespace token boundaries. The planner uses conservative CJK contextual boundaries and suppresses a shorter match when a longer known value matches at the same position. Unrecognized query continuations can produce a safe global fallback rather than a scoped result; this is a precision limitation, not a broadening path.
- This task adds no relevance judgments or production recall claims. Evaluation remains scheduled for Task 11.
- Task 7 retrieval execution has not been started in this commit.
