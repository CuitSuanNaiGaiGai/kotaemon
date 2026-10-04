"""Synthetic tests for the standalone local QA workbench."""

from __future__ import annotations

from pathlib import Path

import pytest

from ktem.local_qa_core import EvidenceCard
from ktem.local_qa_playground import (
    answer_question,
    build_ui,
    clear_outputs,
    main,
)


class FakeQA:
    def __init__(self, cards=(), *, cards_by_question=None):
        self.cards = list(cards)
        self.cards_by_question = cards_by_question
        self.calls = []

    def retrieve(self, question):
        self.calls.append(question)
        if self.cards_by_question is not None:
            return tuple(self.cards_by_question[question])
        return tuple(self.cards)


class FakeGenerator:
    def __init__(self, answer="synthetic answer", *, error=None, model="synthetic:tag"):
        self.answer = answer
        self.error = error
        self.model = model
        self.calls = []

    def generate(self, question, cards):
        self.calls.append((question, cards))
        if self.error is not None:
            raise self.error
        return self.answer


def card(
    *,
    source_rank=1,
    chunk_rank=1,
    source_id="synthetic-source",
    source_label="synthetic/doc.md",
    locator=None,
    score=0.75,
    text="synthetic evidence",
):
    return EvidenceCard(
        source_rank=source_rank,
        chunk_rank=chunk_rank,
        source_id=source_id,
        source_label=source_label,
        locator=locator if locator is not None else {"page": 2},
        score=score,
        text=text,
    )


def test_answer_question_generates_from_exact_retrieval_cards():
    evidence_card = card(text="evidence")
    qa = FakeQA(cards=[evidence_card])
    generator = FakeGenerator(answer="The answer is 42 [1].")

    answer, evidence = answer_question(qa, generator, "What is the answer?")

    assert answer == "The answer is 42 [1]."
    assert qa.calls == ["What is the answer?"]
    assert generator.calls == [("What is the answer?", (evidence_card,))]
    assert evidence == [
        {
            "source_rank": 1,
            "chunk_rank": 1,
            "source": "synthetic/doc.md",
            "locator": {"page": 2},
            "score": 0.75,
            "text": "evidence",
        }
    ]


def test_empty_retrieval_skips_generation_and_clear_resets_outputs():
    qa = FakeQA(cards=[])
    generator = FakeGenerator(answer="must not be used")

    answer, evidence = answer_question(qa, generator, "unknown")

    assert "no evidence" in answer.lower()
    assert evidence == []
    assert generator.calls == []
    assert clear_outputs() == ("", "", [])


def test_blank_question_is_rejected_before_retrieval_or_generation():
    qa = FakeQA(cards=[card()])
    generator = FakeGenerator()

    answer, evidence = answer_question(qa, generator, "  \n  ")

    assert answer == "Enter a question."
    assert evidence == []
    assert qa.calls == []
    assert generator.calls == []


def test_independent_asks_use_only_each_questions_own_cards():
    first_card = card(source_id="synthetic-a", text="first evidence")
    second_card = card(source_id="synthetic-b", text="second evidence")
    qa = FakeQA(
        cards_by_question={
            "first question": [first_card],
            "second question": [second_card],
        }
    )
    generator = FakeGenerator()

    first_answer, first_evidence = answer_question(qa, generator, "first question")
    second_answer, second_evidence = answer_question(qa, generator, "second question")

    assert first_answer == "synthetic answer"
    assert second_answer == "synthetic answer"
    assert [item["text"] for item in first_evidence] == ["first evidence"]
    assert [item["text"] for item in second_evidence] == ["second evidence"]
    assert qa.calls == ["first question", "second question"]
    assert generator.calls == [
        ("first question", (first_card,)),
        ("second question", (second_card,)),
    ]


@pytest.mark.parametrize("error", [OSError("secret question secret source")])
def test_generator_failure_preserves_evidence_without_echoing_private_text(error):
    evidence_card = card(
        source_label="secret/source.md",
        text="secret evidence text",
    )
    qa = FakeQA(cards=[evidence_card])
    generator = FakeGenerator(error=error, model="synthetic-model")

    answer, evidence = answer_question(qa, generator, "secret question")

    assert "start ollama" in answer.lower()
    assert "ollama pull synthetic-model" in answer
    assert "secret question" not in answer
    assert "secret/source.md" not in answer
    assert "secret evidence text" not in answer
    assert evidence[0]["source"] == "secret/source.md"
    assert evidence[0]["text"] == "secret evidence text"
    assert generator.calls == [("secret question", (evidence_card,))]


def test_generator_value_error_preserves_evidence_and_hides_error_details():
    evidence_card = card(text="synthetic visible chunk")
    qa = FakeQA(cards=[evidence_card])
    generator = FakeGenerator(error=ValueError("private prompt details"))

    answer, evidence = answer_question(qa, generator, "synthetic question")

    assert "private prompt details" not in answer
    assert evidence[0]["text"] == "synthetic visible chunk"


def test_ui_wires_ask_and_clear_and_disables_analytics():
    evidence_card = card(text="synthetic UI evidence")
    qa = FakeQA(cards=[evidence_card])
    generator = FakeGenerator(answer="synthetic UI answer")
    demo = build_ui(qa, generator, snapshot_version="synthetic-v2")
    config = demo.get_config_file()
    components = {component["id"]: component for component in config["components"]}
    description = "\n".join(
        str(component["props"].get("value", ""))
        for component in config["components"]
        if component["type"] == "markdown"
    )

    assert config["analytics_enabled"] is False
    assert "synthetic-v2" in description
    assert "BGE-M3" in description
    assert "baseline token chunks" in description
    assert "20 vector candidates" in description
    assert "top five distinct sources" in description
    assert "synthetic:tag" in description
    assert not any(
        component["type"] in {"chatbot", "state"} for component in config["components"]
    )

    question_id = next(
        component_id
        for component_id, component in components.items()
        if component["type"] == "textbox"
        and component["props"].get("label") == "Question"
    )
    answer_id = next(
        component_id
        for component_id, component in components.items()
        if component["type"] == "textbox"
        and component["props"].get("label") == "Answer"
    )
    evidence_id = next(
        component_id
        for component_id, component in components.items()
        if component["type"] == "json"
        and component["props"].get("label") == "Retrieved evidence"
    )
    ask_id = next(
        component_id
        for component_id, component in components.items()
        if component["type"] == "button" and component["props"].get("value") == "Ask"
    )
    clear_id = next(
        component_id
        for component_id, component in components.items()
        if component["type"] == "button" and component["props"].get("value") == "Clear"
    )
    dependencies = config["dependencies"]
    ask_event = next(
        event for event in dependencies if (ask_id, "click") in event["targets"]
    )
    clear_event = next(
        event for event in dependencies if (clear_id, "click") in event["targets"]
    )

    assert ask_event["inputs"] == [question_id]
    assert ask_event["outputs"] == [answer_id, evidence_id]
    ask_fn = demo.fns[ask_event["id"]].fn
    assert ask_fn("Which evidence?") == (
        "synthetic UI answer",
        [
            {
                "source_rank": 1,
                "chunk_rank": 1,
                "source": "synthetic/doc.md",
                "locator": {"page": 2},
                "score": 0.75,
                "text": "synthetic UI evidence",
            }
        ],
    )
    assert generator.calls == [("Which evidence?", (evidence_card,))]

    assert clear_event["inputs"] == []
    assert clear_event["outputs"] == [question_id, answer_id, evidence_id]
    clear_fn = demo.fns[clear_event["id"]].fn
    assert clear_fn() == ("", "", [])


def test_main_forwards_cli_arguments_and_uses_loopback_launch(monkeypatch):
    import ktem.local_qa_playground as playground

    expected_paths = (
        Path("/synthetic/local-root"),
        Path("/synthetic/local-root/snapshots/v2"),
        Path("/synthetic/models/bge-m3"),
        Path("/synthetic/models/bge-reranker"),
    )
    open_calls = []
    generator_calls = []
    qa = object()
    generator = object()

    def fake_open(*args):
        open_calls.append(args)
        return qa

    def fake_generator(endpoint, model):
        generator_calls.append((endpoint, model))
        return generator

    class FakeDemo:
        def __init__(self):
            self.launch_kwargs = None

        def launch(self, **kwargs):
            self.launch_kwargs = kwargs

    demo = FakeDemo()
    build_calls = []

    def fake_build(actual_qa, actual_generator, *, snapshot_version):
        build_calls.append((actual_qa, actual_generator, snapshot_version))
        return demo

    monkeypatch.setattr(playground, "open_playground", fake_open)
    monkeypatch.setattr(playground, "OllamaLocalClient", fake_generator)
    monkeypatch.setattr(playground, "build_ui", fake_build)

    result = main(
        [
            "--local-root",
            str(expected_paths[0]),
            "--snapshot",
            str(expected_paths[1]),
            "--embedding-model-dir",
            str(expected_paths[2]),
            "--reranker-model-dir",
            str(expected_paths[3]),
            "--ollama-endpoint",
            "http://127.0.0.1:11435",
            "--model",
            "synthetic:tag",
            "--server-port",
            "7865",
        ]
    )

    assert result == 0
    assert open_calls == [expected_paths]
    assert generator_calls == [("http://127.0.0.1:11435", "synthetic:tag")]
    assert build_calls == [(qa, generator, "v2")]
    assert demo.launch_kwargs == {
        "server_name": "127.0.0.1",
        "server_port": 7865,
        "share": False,
        "inbrowser": False,
    }
