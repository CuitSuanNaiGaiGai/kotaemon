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
records identify source hashes and page, sheet, or section locations plus
normalized-text offsets and an evidence digest. Do not put per-arm chunk IDs in
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

The manifest should record the records/judgments/anchors schema versions and
snapshot version; each original source file's repository-relative path (for example,
`libs/kotaemon/tests/fixtures/knowledge_eval/local/sources/<source-id>/<filename>`)
and SHA-256; the SHA-256 of the exact bytes of the frozen `records.json`,
`judgments.jsonl`, and `anchors.jsonl` payloads; loader and chunker names,
versions, and relevant configuration; document, chunk, and query counts; and
review status and date. Record the three snapshot payload hashes in a
`payload_sha256` mapping keyed by filename. Use repository-relative paths rather
than machine-specific absolute paths.

Before evaluating a snapshot, recompute the hashes for all three payload files
and compare them with the corresponding manifest entries. Do not evaluate the
snapshot if a hash is missing or mismatched. A
reviewed snapshot is immutable: any change to its payloads, manifest, or
documented source/loader/chunker inputs must be published under a new
`local/snapshots/<version>/` directory with an updated manifest; do not
overwrite an existing version. When explicit snapshot-path selection is added
to the runner, it must perform this hash check before evaluation.

The synthetic fixture remains the runner's default. Local experiment runs must
explicitly select a reviewed snapshot path and enforce the hash check described
above.

## Local-data handling

Git ignores the entire `local/` directory as a convenience, so its contents
are not added to a PR by default. `.gitignore` is not a security boundary:
ignored files can still be force-staged or disclosed by other means. Do not
use `git add -f` for this data or publish original materials, drafts, or local
snapshots without explicit authorization.
