# Agent Knowledge Retrieval Evaluation

> All metrics in this report are **synthetic fixture evaluation**. They do not
> estimate production recall, answer quality, or general retrieval performance.

## Scope and pipeline comparison

The comparison uses five version 1 JSONL judgments over four indexed synthetic
Sources and eleven chunks. Corpus text and logical metadata are stored in
[`records.json`](../../../libs/kotaemon/tests/fixtures/knowledge_eval/records.json);
canonical query judgments are stored separately in
[`judgments.jsonl`](../../../libs/kotaemon/tests/fixtures/knowledge_eval/judgments.jsonl).
All judgment IDs were validated against the fixture's immutable Source/chunk
catalog before either arm ran. Both arms indexed the same corpus once and ran
the same queries in the same order. The integration harness asserts that no
relevance-label key enters indexed chunk metadata or either shared run config.

The baseline injects an identity/global planner into `KnowledgeService`; the
planned arm injects `QueryPlanner`. Both call `KnowledgeService.search`, which
applies caller visibility and explicit path/type/metadata constraints before
the existing `VectorRetrieval` stage. The planner can narrow that mandatory
scope. An ambiguous or legacy query stays global within the visible catalog.
The existing file UI behavior is unchanged: an empty selected-source list
returns no results, and the explicit empty-allowlist check passed without a
vector-store call. No production planner switch or schema change was added.

| Setting | Baseline | Planned |
| --- | --- | --- |
| Planner | identity/global | `QueryPlanner` |
| Retrieval mode | vector | vector |
| Top K | 2 | 2 |
| Configured and observed candidate K | 2 | 2 |
| First-round multiplier / `do_extend` | 1 / false | 1 / false |
| Rerankers | none | none |
| Diversity cap / MMR | disabled (`null`) / false | disabled (`null`) / false |
| Context budget | 4096 characters (`len` counter) | 4096 characters (`len` counter) |
| Vector backend | deterministic synthetic cosine adapter | same indexed adapter |
| Document store | `InMemoryDocumentStore` | same store |
| Lexical capability | `supports_lexical_search=False`; lexical branch not used | same |

The fixture uses a fixed hashed character-bigram embedding and cosine ranking.
It is a deterministic synthetic vector adapter, not a production embedding or
backend. The configured mode is vector-only; the trace records lexical as
`not_used`, and no BM25 result or score is fabricated. Both configs match in
every field except planner identity. Shared config fingerprint:

```text
a3a197bbbea956400cda93ed15084c6b3e8f6198f743a92e1613063a9776d432
```

Fixture SHA-256:

```text
4150e95b07a1778db3dca01f816a31708c214f2ca4a9600178f0d310ce968998
```

## Synthetic fixture evaluation metrics

Hit@2, Recall@2, and MRR@2 use macro means over all five judged queries,
including any zero-result query. Wrong-scope rate is pooled over unique
returned chunks from the three queries with `disallowed_source_ids` labels.
Queries with a null scope label do not enter that denominator.

| Metric | Identity/global baseline | Planned | Denominator |
| --- | ---: | ---: | --- |
| Hit@2 | 1.0000 | 1.0000 | 5 judged queries |
| Macro Recall@2 | 0.7667 | 0.8667 | 5 judged queries |
| MRR@2 | 0.7000 | 0.8000 | 5 judged queries |
| Wrong-scope@2 | 0.3333 (2/6) | 0.1667 (1/6) | 3 labeled queries; pooled unique returned chunks |

The per-query results below include every judgment. `—` means the case has no
scope label, so wrong-scope rate is undefined rather than zero.

| Query ID | Judgment IDs | Baseline top-2; Hit / Recall / RR; wrong scope | Planned top-2; Hit / Recall / RR; wrong scope | Planned route |
| --- | --- | --- | --- | --- |
| `zhang-internship` | `chunk-zhang-rag`, `chunk-zhang-api` | `chunk-li-zhang-question`, `chunk-zhang-rag`; 1 / 0.5 / 0.5; 1/2 = 0.5 | `chunk-zhang-rag`, `chunk-zhang-api`; 1 / 1 / 1; 0/2 = 0 | scoped to `source-zhang` |
| `li-frontend` | `chunk-li-frontend` | `chunk-li-frontend`, `chunk-li-wang-question`; 1 / 1 / 1; 0/2 = 0 | `chunk-li-frontend`, `chunk-li-wang-question`; 1 / 1 / 1; 0/2 = 0 | explicit path/type/filter scope |
| `wang-automation` | `chunk-wang-automation` | `chunk-li-wang-question`, `chunk-wang-automation`; 1 / 1 / 0.5; 1/2 = 0.5 | `chunk-li-wang-question`, `chunk-wang-automation`; 1 / 1 / 0.5; 1/2 = 0.5 | global; current planner did not resolve this wording |
| `ambiguous-team` | `chunk-zhang-rag`, `chunk-li-frontend`, `chunk-wang-team-tech` | `chunk-li-wang-question`, `chunk-wang-team-tech`; 1 / 0.3333 / 0.5; — | `chunk-li-wang-question`, `chunk-wang-team-tech`; 1 / 0.3333 / 0.5; — | global directly for ambiguous query |
| `legacy-source` | `chunk-legacy-api` | `chunk-legacy-api`, `chunk-legacy-token`; 1 / 1 / 1; — | `chunk-legacy-api`, `chunk-legacy-token`; 1 / 1 / 1; — | global; legacy Source has no canonical planner metadata |

The planned aggregate improved macro Recall@2 from 0.7667 to 0.8667 and lowered
pooled wrong-scope rate from 0.3333 to 0.1667 without a fixture recall loss.
The Wang case still returns a Li Source distractor because the current planner
enters global retrieval for that wording; the 0.5 per-query wrong-scope value
remains in both the report and machine-readable output. The query, judgment,
and distractor corpus text were not changed to conceal that result.

## Zhang trace and citation lookup

The complete default-redacted trace is
[`agent-retrieval-zhang-trace.json`](artifacts/agent-retrieval-zhang-trace.json).
The paired metrics, observations, candidate counts, and frozen resolved
judgments are in
[`agent-retrieval-evaluation.json`](artifacts/agent-retrieval-evaluation.json).
The planned Zhang trace records this stage order:

1. Request records the original query `张三实习期间做了什么工作？`.
2. Explicit filters are empty. The deterministic plan resolves the exact
   `person=张三` entity to `source-zhang` at confidence 1.0 and keeps the
   original semantic query.
3. The caller-visible and mandatory scope contains all four fixture Sources;
   the planned chunk scope contains the three Zhang chunks.
4. One vector recall attempt runs with observed `candidate_k=2`. The vector
   branch is available, with candidates `chunk-zhang-rag` (0.3015) and
   `chunk-zhang-api` (0.0711). Lexical is `not_used`; there are no lexical
   candidates, reranker outputs, or fallback attempts.
5. Merge and diversity keep the same two IDs. The final IDs are
   `chunk-zhang-rag` and `chunk-zhang-api`.
6. Evidence construction retains both complete chunks and records 319
   characters against the 4096-character budget. The default trace contains
   IDs, logical paths, scores, and stages, with no full chunk text.

Citation lookup used the literal synthetic quote `负责构建检索 API`. It appears
in the evidence context; `AnswerWithContextPipeline.match_evidence_with_context`
returned span `[17, 27)` for `chunk-zhang-rag`, and slicing that span from the
original returned chunk produced the same quote. This checks string-to-source
mapping only; no LLM answer was generated or evaluated.

## TDD and verification evidence

Before implementation, the pure evaluator command failed as expected because
the evaluator package was absent. All 14 cases stopped at the explicit missing
package guard; that first RED run did not reach the arithmetic or validation
assertions:

```text
$ uv run pytest libs/kotaemon/tests/test_knowledge_retrieval_eval.py -q
14 failed in 3.09s
```

After the evaluator implementation was added, the first behavioral run reached
the assertions: 13 passed and one failed because the candidate-count diagnostic
message differed from the expected wording. The assertion was tightened around
the behavior, and the pure suite passed. The hand-worked arithmetic oracle uses
`K=3`, relevant `{a,b}`, results `[x,a,a,b]`: it yields Hit 1, Recall 1/2,
RR 1/2, and wrong-scope 1/2 over unique results; a second judged empty result
has Hit/Recall/RR 0 and no wrong-scope denominator. Their macro Hit/Recall/MRR
are 1/2, 1/4, and 1/4. Additional tests distinguish a null scope label from an
empty disallowed-source list and reject missing observed candidate counts.

Final commands and observed results:

```text
$ uv run pytest libs/kotaemon/tests/test_knowledge_retrieval_eval.py -q
18 passed in 2.25s

$ uv run pytest libs/kotaemon/tests/test_knowledge_retrieval_eval.py libs/ktem/ktem_tests/test_knowledge_retrieval_eval_integration.py libs/kotaemon/tests/test_knowledge_service.py libs/kotaemon/tests/test_knowledge_scoped_retrieval.py libs/kotaemon/tests/test_knowledge_context.py libs/kotaemon/tests/test_knowledge_trace.py libs/ktem/ktem_tests/test_knowledge_service_integration.py libs/ktem/ktem_tests/test_knowledge_trace_integration.py -q
110 passed in 6.72s

$ uv run black --check libs/kotaemon/kotaemon/indices/knowledge/evaluation/retrieval_eval.py libs/kotaemon/tests/test_knowledge_retrieval_eval.py libs/ktem/ktem_tests/test_knowledge_retrieval_eval_integration.py
All done! ✨ 🍰 ✨
3 files would be left unchanged.

$ uv run flake8 --max-line-length=88 --extend-ignore=E203 libs/kotaemon/kotaemon/indices/knowledge/evaluation/retrieval_eval.py libs/kotaemon/tests/test_knowledge_retrieval_eval.py libs/ktem/ktem_tests/test_knowledge_retrieval_eval_integration.py
exit 0; no diagnostics

$ uv run isort --check-only --profile black libs/kotaemon/kotaemon/indices/knowledge/evaluation/retrieval_eval.py libs/kotaemon/tests/test_knowledge_retrieval_eval.py libs/ktem/ktem_tests/test_knowledge_retrieval_eval_integration.py
exit 0; no diagnostics

$ git diff --check
exit 0; no diagnostics
```

An intervening standalone focused invocation stopped at collection while
importing the ktem integration module. LlamaIndex's NLTK initialization found
only cached `punkt_tab`, requested `punkt`, then its attempted downloader index
response failed XML parsing (`unclosed token: line 33, column 4`). The final
aggregate invocation above subsequently passed with both Task 11 modules
included. No runtime dependencies or NLTK cache files were changed.

```text
$ uv run pytest libs/kotaemon/tests/test_knowledge_retrieval_eval.py libs/ktem/ktem_tests/test_knowledge_retrieval_eval_integration.py -q
ERROR collecting libs/ktem/ktem_tests/test_knowledge_retrieval_eval_integration.py
LookupError: Resource punkt not found; attempted punkt_tab download
xml.etree.ElementTree.ParseError: unclosed token: line 33, column 4
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 3.54s
```

## Files and limits

Task 11 adds the framework-independent fixture loader, resolver, scorer, and
paired runner under `libs/kotaemon/kotaemon/indices/knowledge/evaluation/`,
unit tests under `libs/kotaemon/tests/`, and the synthetic corpus, judgments,
trace, machine-readable results, and this report. The focused ktem integration
test runs through the existing `KnowledgeService`, `QueryPlanner`,
`VectorRetrieval`, `PrepareEvidencePipeline`, and Task 10 `RetrievalTrace` API.

The fixture is synthetic and small; it contains no enterprise Wiki corpus or
human-labeled production judgments. Its cosine adapter uses character bigrams,
not the configured production embedding or reranker. QueryPlanner did not
scope the Wang wording, and ambiguous/legacy queries entered global retrieval.
No LLM answer quality, production citation precision, lexical/BM25 behavior, or
general retrieval performance is measured here.
