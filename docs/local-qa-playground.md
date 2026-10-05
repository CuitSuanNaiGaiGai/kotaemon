# Local snapshot QA playground

The standalone workbench opens one approved frozen snapshot and builds an
isolated retrieval index in memory through the shared knowledge runtime. It
uses local BGE-M3 embeddings over the baseline token chunks, a local
cross-encoder when its verified weights are available, and an ephemeral FTS5
lexical route. The runtime gathers up to 20 vector candidates (plus lexical
candidates when FTS5 is available), selects the seed window from the top five
distinct sources, then considers authorized neighboring chunks. The evidence
panel labels these limits as chunk seed K=20 and source K=5. If FTS5 or the
local reranker is unavailable, the UI reports that degraded route and keeps
the usable retrieval paths.

Each question is handled independently. Before generation, complete evidence
cards are packed against the configured model budget. A matching cached Qwen
tokenizer is used when present; otherwise the client counts UTF-8 bytes and
labels the budget estimated. If no seed card fits, generation is skipped. The
client also checks the final rendered system and user messages against the
budget before sending them. Route status, fusion IDs, reranking, expansion,
packing, and omitted-card diagnostics appear with the evidence. The answer
panel cites only cards in that final packed context.

## Prerequisites

- Use an approved snapshot under your Git-ignored local root and the local
  embedding and reranker model directories associated with that root.
- Install and start Ollama on the local machine, then download the configured
  model before asking questions:

  ```bash
  ollama pull qwen2.5:7b
  ```

The UI reports the snapshot version and configured generation model when it
starts. It binds to `127.0.0.1`; Gradio sharing and automatic browser opening
are disabled. The Ollama endpoint accepts only an HTTP loopback IP address.

## Launch

Run this from the repository root, replacing each placeholder with a path on
your machine:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --package ktem python -m ktem.local_qa_playground \
  --local-root /path/to/git-ignored/local-root \
  --snapshot /path/to/git-ignored/local-root/snapshots/v2 \
  --embedding-model-dir /path/to/git-ignored/local-root/models/bge-m3 \
  --reranker-model-dir /path/to/git-ignored/local-root/models/bge-reranker \
  --ollama-endpoint http://127.0.0.1:11434 --model qwen2.5:7b \
  --server-port 7860
```

The question field is independent for every Ask action. The evidence panel
shows source rank, candidate chunk rank, relative source label, locator, score,
and the exact retrieved chunk text. Clear removes the question, answer, and
evidence from the page.

Only when an Ask action has a non-empty question and at least one seed fits the
budget are the original question and exactly the packed evidence cards sent over
the loopback HTTP interface to the local Ollama process. Streaming and
non-streaming requests use the same packed cards. Empty questions, searches
without evidence, and requests where no seed fits skip generation. Generation
failures display an error instead of a generated answer. The workbench does not
write questions, answers, evidence, or traces to the snapshot or its own files.
It does not control Ollama's process-memory or retention behavior.
