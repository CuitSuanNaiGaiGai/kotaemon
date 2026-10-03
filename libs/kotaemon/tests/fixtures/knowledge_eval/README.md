# Knowledge evaluation fixtures and local corpus

## Tracked default fixture

The public default fixture consists of the root-level `records.json` and
`judgments.jsonl`. It contains four synthetic sources (Zhang, Li, Wang, and a
legacy source), 11 chunks, and five synthetic judgments. There are no original
source documents in this fixture. Its content is synthetic data, not users'
documents.

The evaluation runner currently reads only these two root-level files. Keep
them as the unchanged default fixture; the runner does not discover or read
anything under `local/`.

## Optional local model runtime

Local BGE evaluation requires the `kotaemon[local-eval]` extra, pinned to
`FlagEmbedding==1.4.2`, and explicit local directories for `BAAI/bge-m3` and
`BAAI/bge-reranker-v2-m3`. Start the inference process with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Before importing
FlagEmbedding, the adapters check those environment flags, the Hub and
Transformers offline states, and the active Hub session's offline adapters. If
any library or session was initialized online, loading fails and asks for a
fresh offline inference process; the adapters do not change cached process
state after imports.

Preflight parses `config.json` and checks that the required tokenizer and
weight files are present. It cannot guarantee that the local tokenizer or
weights can be loaded. If model construction fails, the error identifies the
BAAI model ID and local path and recommends repairing or re-downloading that
snapshot. Inference never downloads models or sends queries and passages to a
remote service.

Adapter metadata includes model IDs, weight-file SHA-256 hashes, device,
FlagEmbedding version, and truncation lengths. Revision and cached/downloaded
origin are included only when supplied by the model-management flow; adapters
do not infer them from a directory.

## Local source drop-zone and draft outputs

Put user-provided original materials under
`libs/kotaemon/tests/fixtures/knowledge_eval/local/sources/`. A convenient
layout is one directory per stable source ID, with the original filename kept
inside it:

```text
local/
├── sources/
│   └── <source-id>/
│       └── <original-filename>
├── draft/
│   ├── records.json
│   ├── judgments.jsonl
│   └── anchors.jsonl
└── snapshots/
```

Write candidate extracted records, source-level judgments, and independent
evidence anchors to the ignored files in `local/draft/`. The judgment file uses
stable source IDs so its labels remain unchanged across chunking arms. Anchor
records identify query IDs, source hashes, and page, sheet, or section locations
plus normalized-text offsets and an evidence digest. Require at least one anchor
for each relevant query/source pair. Do not put per-arm chunk IDs in
the frozen gold. These files are working data for preparation and review, not
runner inputs.

## Review and fixed snapshots

Before freezing a snapshot, have a person review the content-derived source
topics, extracted text and chunk boundaries, source IDs, evidence anchors,
evaluation queries, relevant source IDs, and disallowed source IDs. After
review, copy the approved data into a versioned snapshot:

```text
local/snapshots/<version>/
├── records.json
├── judgments.jsonl
├── anchors.jsonl
└── manifest.json
```

### Snapshot schema version 1

`records.json` uses the Task 2 schema version 1. Its source rows retain the
exact-byte SHA-256 `source_id`, a root-relative POSIX `relative_path`, byte
size, suffix, duplicate provenance, and reader/status fields. Duplicate-byte
rows may repeat a `source_id` only when they point to the one canonical
root-relative path and agree on source ID, SHA-256, and byte size. A source
unit and each chunk must map back to that canonical source path; chunk offsets
are half-open character offsets into the unit's `normalized_text`.

`topic_review_source_ids` must contain unique, valid SHA-256 IDs for selected
canonical source rows. A selected row may be `parsed` or `failed`; it need not
have a chunk because this list marks sources that still need topic review.
`topic_candidates` must resolve its source ID, unit ID, root-relative path, and
unit locator. A candidate may add locator detail such as a Markdown
`heading_line`, while preserving the unit locator fields and values. Source
references in quality diagnostics and excluded-input records must also map to
the corresponding source rows.

`judgments.jsonl` uses the existing retrieval-evaluation fixture schema version
1. Every judgment must have `judgment_level: "source"`; relevant, allowed,
and disallowed IDs refer to indexable canonical source IDs: the source row must
have `status: "parsed"` and at least one validated chunk. Failed,
not-selected, and chunkless source rows remain provenance only and cannot carry
judgment labels. A relevant ID must be in the allowed set when one is supplied,
and relevant/disallowed and
allowed/disallowed sets must not overlap.

`anchors.jsonl` is UTF-8 JSONL. Each row has exactly these fields:

```json
{
  "schema_version": 1,
  "id": "anchor-1",
  "query_id": "q-1",
  "source_id": "<64-character source SHA-256>",
  "source_sha256": "<64-character source SHA-256>",
  "unit_id": "<source-unit ID>",
  "locator": {"page_label": "1"},
  "char_start": 0,
  "char_end": 12,
  "evidence_sha256": "<SHA-256 of normalized_text[char_start:char_end]>"
}
```

Offsets are half-open and non-empty, and `evidence_sha256` hashes the exact
UTF-8 encoding of the selected normalized-text slice. The locator must equal
the referenced source unit's locator. Each anchor must name a known query and
canonical source relevant to that query. Every relevant query/source pair
needs at least one anchor. The span text is not duplicated in the anchor.

`manifest.json` is a JSON object with exactly these fields:

| Field | Version 1 meaning |
| --- | --- |
| `schema_version` | Manifest schema, integer `1`. |
| `snapshot_version` | Safe version name matching the snapshot directory name. |
| `schema_versions` | Exactly `{"records": 1, "judgments": 1, "anchors": 1}`. |
| `payload_sha256` | Exact map of the three payload filenames to lowercase SHA-256 of their exact bytes. |
| `review_status` | `approved` for evaluation or `draft` for inspection. |
| `review_date` | ISO `YYYY-MM-DD` approval date, or `null` for a draft. |
| `source_root` | Repository-relative root corresponding to the root passed to local preparation. |
| `source_provenance` | One ordered row per `records.json` source path: `source_id`, `repository_path`, and `sha256`. Each repository path must equal `source_root/relative_path`. |
| `parser_configuration` | Exact copy of `records.json`'s parser configuration. |
| `splitter_configuration` | Exact copy of `records.json`'s splitter configuration. |
| `counts` | Exactly `source_paths`, `documents`, `source_units`, `chunks`, `queries`, and `anchors`, matching the payloads; `documents` counts unique canonical sources with `status: "parsed"` and at least one chunk. |

For example, if `source_root` is
`libs/kotaemon/tests/fixtures/knowledge_eval/local/sources` and a record's
root-relative path is `<source-id>/<filename>`, its manifest path is
`libs/kotaemon/tests/fixtures/knowledge_eval/local/sources/<source-id>/<filename>`.
Never record machine-absolute paths. The payload hash map must contain exactly
`records.json`, `judgments.jsonl`, and `anchors.jsonl` with valid lowercase
64-character hex digests.

Before evaluating a snapshot, recompute the hashes for all three payload files
and compare them with the corresponding manifest entries before parsing. The
loader rejects symlink roots/payloads, missing or extra hash entries, unsafe
paths, source/unit/chunk mapping errors, invalid source-level judgments, and
invalid or incomplete anchors. Its snapshot fingerprint is SHA-256 of the exact
`manifest.json` bytes. A reviewed snapshot is immutable: any change to its
payloads, manifest, or documented source/loader/chunker inputs must be
published under a new `local/snapshots/<version>/` directory with an updated
manifest; do not overwrite an existing version. Task 7's freeze command enforces
the no-overwrite rule. The evaluation loader requires `review_status: "approved"`
by default; `require_reviewed=False` is only for inspecting a fully populated,
otherwise valid draft snapshot.

The synthetic fixture remains the runner's default. Local experiment runs must
explicitly select a reviewed snapshot path and enforce the hash check described
above.

## Local-data handling

Git ignores the entire `local/` directory as a convenience, so its contents
are not added to a PR by default. `.gitignore` is not a security boundary:
ignored files can still be force-staged or disclosed by other means. Do not
use `git add -f` for this data or publish original materials, drafts, or local
snapshots without explicit authorization.
