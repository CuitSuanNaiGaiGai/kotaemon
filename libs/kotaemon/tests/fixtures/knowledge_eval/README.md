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
│   └── judgments.jsonl
└── snapshots/
```

Write candidate extracted records and judgments to the ignored files
`libs/kotaemon/tests/fixtures/knowledge_eval/local/draft/records.json` and
`libs/kotaemon/tests/fixtures/knowledge_eval/local/draft/judgments.jsonl`.
These are working files for preparation and review, not runner inputs.

## Review and fixed snapshots

Before freezing a snapshot, have a person review the source text, extracted
text and chunk boundaries, source/chunk IDs, evaluation queries, relevant IDs,
and disallowed IDs. After review, copy the approved data into a versioned
snapshot:

```text
local/snapshots/<version>/
├── records.json
├── judgments.jsonl
└── manifest.json
```

The manifest should record the records/judgments schema versions and snapshot
version; each original source file's repository-relative path (for example,
`libs/kotaemon/tests/fixtures/knowledge_eval/local/sources/<source-id>/<filename>`)
and SHA-256; loader and chunker names, versions, and relevant configuration;
document, chunk, and query counts; and review status and date. Use
repository-relative paths rather than machine-specific absolute paths.

The current runner will not load a local draft or snapshot. Once source
materials are supplied and a snapshot is reviewed, add explicit snapshot-path
selection in a separate change. That change should preserve the synthetic
root-level fixture as the default.

## Local-data handling

Git ignores the entire `local/` directory as a convenience, so its contents
are not added to a PR by default. `.gitignore` is not a security boundary:
ignored files can still be force-staged or disclosed by other means. Do not
use `git add -f` for this data or publish original materials, drafts, or local
snapshots without explicit authorization.
