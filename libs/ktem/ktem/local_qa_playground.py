"""Standalone Gradio workbench for questions over an approved local snapshot."""

from __future__ import annotations

import argparse
import urllib.error
from collections.abc import Iterator, Sequence
from pathlib import Path
from time import monotonic
from typing import Any

import gradio as gr

from ktem.local_qa_core import LocalQA
from ktem.local_qa_core import open_playground
from ktem.local_qa_ollama import OllamaLocalClient

_STREAM_UPDATE_INTERVAL = 0.1


def answer_question(
    qa: LocalQA,
    generator: OllamaLocalClient,
    question: str,
) -> tuple[str, list[dict[str, Any]]]:
    clean_question = question.strip()
    if not clean_question:
        return "Enter a question.", []

    cards = qa.retrieve(clean_question)
    evidence = [
        {
            "source_rank": card.source_rank,
            "chunk_rank": card.chunk_rank,
            "source": card.source_label,
            "locator": dict(card.locator),
            "score": card.score,
            "text": card.text,
        }
        for card in cards
    ]
    if not cards:
        return "No evidence was retrieved from the frozen snapshot.", []

    try:
        answer = generator.generate(clean_question, cards)
    except (OSError, ValueError, urllib.error.URLError):
        answer = (
            "Local answer generation failed. Start Ollama and run "
            f"`ollama pull {generator.model}`."
        )
    return answer, evidence


def stream_answer_question(
    qa: LocalQA,
    generator: OllamaLocalClient,
    question: str,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    clean_question = question.strip()
    if not clean_question:
        yield "Enter a question.", []
        return

    cards = qa.retrieve(clean_question)
    evidence = [
        {
            "source_rank": card.source_rank,
            "chunk_rank": card.chunk_rank,
            "source": card.source_label,
            "locator": dict(card.locator),
            "score": card.score,
            "text": card.text,
        }
        for card in cards
    ]
    if not cards:
        yield "No evidence was retrieved from the frozen snapshot.", []
        return

    yield "Generating locally…", evidence
    answer = ""
    last_emitted_answer = ""
    last_update = monotonic()
    try:
        for chunk in generator.generate_stream(clean_question, cards):
            answer += chunk
            now = monotonic()
            if (
                answer != last_emitted_answer
                and now - last_update >= _STREAM_UPDATE_INTERVAL
            ):
                yield answer, gr.skip()
                last_emitted_answer = answer
                last_update = now
    except (OSError, ValueError, urllib.error.URLError):
        yield (
            "Local answer generation failed. Check that Ollama is running and "
            "the configured model is available.",
            gr.skip(),
        )
        return

    if answer and answer != last_emitted_answer:
        yield answer, gr.skip()


def clear_outputs() -> tuple[str, str, list[dict[str, Any]]]:
    return "", "", []


def build_ui(
    qa: LocalQA,
    generator: OllamaLocalClient,
    *,
    snapshot_version: str,
) -> gr.Blocks:
    description = (
        f"Frozen snapshot version: `{snapshot_version}`. "
        "Retrieval uses local BGE-M3 embeddings with baseline token chunks, "
        "a pool of 20 vector candidates, and a window showing the top five "
        "distinct sources. "
        f"Answer generation uses the configured local model `{generator.model}`."
    )

    def ask_question(question_text: str):
        yield from stream_answer_question(qa, generator, question_text)

    with gr.Blocks(title="Local Snapshot QA", analytics_enabled=False) as demo:
        gr.Markdown(description)
        question = gr.Textbox(label="Question", lines=3)
        with gr.Row():
            ask = gr.Button("Ask", variant="primary")
            clear = gr.Button("Clear")
        answer = gr.Textbox(label="Answer", lines=8, interactive=False)
        evidence = gr.JSON(label="Retrieved evidence")
        ask_event = ask.click(
            fn=ask_question,
            inputs=[question],
            outputs=[answer, evidence],
        )
        clear.click(
            fn=clear_outputs,
            inputs=[],
            outputs=[question, answer, evidence],
            cancels=[ask_event],
        )
    return demo


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask independent questions over an approved local snapshot."
    )
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--embedding-model-dir", type=Path, required=True)
    parser.add_argument("--reranker-model-dir", type=Path, required=True)
    parser.add_argument(
        "--ollama-endpoint",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args(argv)

    qa = open_playground(
        args.local_root,
        args.snapshot,
        args.embedding_model_dir,
        args.reranker_model_dir,
    )
    generator = OllamaLocalClient(args.ollama_endpoint, args.model)
    demo = build_ui(qa, generator, snapshot_version=args.snapshot.name)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.server_port,
        share=False,
        inbrowser=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
