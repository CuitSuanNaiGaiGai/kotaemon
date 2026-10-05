"""Standalone Gradio workbench for questions over an approved local snapshot."""

from __future__ import annotations

import argparse
import urllib.error
from collections.abc import Iterator, Sequence
from pathlib import Path
from time import monotonic
from typing import Any

import gradio as gr
from ktem.local_qa_core import LocalQA, open_playground
from ktem.local_qa_ollama import OllamaLocalClient

_STREAM_UPDATE_INTERVAL = 0.1


def answer_question(
    qa: LocalQA,
    generator: OllamaLocalClient,
    question: str,
) -> tuple[str, Any]:
    clean_question = question.strip()
    if not clean_question:
        return "Enter a question.", []

    cards, evidence, status = _retrieve_generation_evidence(
        qa, generator, clean_question
    )
    if status == "insufficient_evidence":
        return "No seed evidence fits within the configured context budget.", evidence
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
) -> Iterator[tuple[str, Any]]:
    clean_question = question.strip()
    if not clean_question:
        yield "Enter a question.", []
        return

    cards, evidence, status = _retrieve_generation_evidence(
        qa, generator, clean_question
    )
    if status == "insufficient_evidence":
        yield "No seed evidence fits within the configured context budget.", evidence
        return
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


def _retrieve_generation_evidence(qa, generator, question):
    retrieve_result = getattr(qa, "retrieve_result", None)
    if callable(retrieve_result):
        options = {}
        generation_budget = getattr(generator, "generation_budget", None)
        count_tokens = getattr(generator, "count_tokens", None)
        base_prompt = getattr(generator, "base_prompt", None)
        if generation_budget is not None:
            options["generation_budget"] = generation_budget
            options["count_tokens"] = count_tokens
            options["base_prompt"] = (
                base_prompt(question) if callable(base_prompt) else ""
            )
        result = retrieve_result(question, **options)
        cards = tuple(result.cards)
        diagnostics = dict(result.diagnostics or {})
        evidence_cards = [
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
        if diagnostics or result.status == "insufficient_evidence":
            diagnostics["generation_budget"] = (
                "estimated"
                if diagnostics.get("budget_estimated") is True
                else (
                    "measured"
                    if diagnostics.get("budget_estimated") is False
                    else "not_configured"
                )
            )
            evidence = {
                "cards": evidence_cards,
                "diagnostics": diagnostics,
                "status": result.status,
            }
        else:
            evidence = evidence_cards
        return cards, evidence, result.status

    cards = tuple(qa.retrieve(question))
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
    return cards, evidence, "ready" if cards else "no_evidence"


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
        "Retrieval uses local BGE-M3 embeddings over baseline token chunks, "
        "a local cross-encoder when its verified weights are available, "
        "ephemeral FTS5 diagnostics, a pool of 20 vector candidates, and a "
        "seed window from the top five distinct sources (chunk seed K=20; source K=5). "
        "Evidence expansion is scope-checked and generation context is packed "
        "before Ollama receives it. Diagnostics report unavailable routes and "
        "estimated token budgets. "
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
