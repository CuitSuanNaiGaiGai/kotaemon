# Local snapshot QA playground

The standalone workbench opens one approved frozen snapshot and builds its
retrieval index in memory. It uses local BGE-M3 embeddings over the baseline
token chunks, retrieves a pool of 20 vector candidates, and displays chunks
from the top five distinct sources. Each question is handled independently.
The answer panel shows citations generated from the displayed evidence cards.

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
are disabled.

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

For each Ask action, the question and exact retrieved evidence cards are sent
over the loopback HTTP interface to the local Ollama process for generation;
the generated answer is returned to the UI. The workbench does not write
questions, answers, or evidence to the snapshot or its own trace file. The
workbench does not control Ollama's process-memory or retention behavior.
