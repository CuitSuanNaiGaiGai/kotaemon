"""Synthetic whole-unit generator-budget tests."""

from __future__ import annotations

from kotaemon.base import RetrievedDocument
from kotaemon.indices.knowledge.retrieval.context_budget import (
    EvidenceBundle,
    GenerationBudget,
    pack_evidence,
)


def make_doc(doc_id, text, **metadata):
    return RetrievedDocument(id_=doc_id, text=text, metadata=metadata)


def test_seed_first_packing():
    seed_a = make_doc("seed-a", "1234")
    seed_b = make_doc("seed-b", "5678")
    neighbor = make_doc("neighbor", "n")
    bundle = EvidenceBundle((seed_a, seed_b), (neighbor,), ())

    packed = pack_evidence(
        bundle,
        budget=GenerationBudget(model_context=8, output_reserve=0, format_reserve=0),
        count_tokens=len,
        base_prompt="",
        render_context=lambda documents: "".join(
            document.text for document in documents
        ),
    )

    assert [document.doc_id for document in packed.documents] == ["seed-a", "seed-b"]
    assert packed.status == "ready"
    assert packed.token_count == 8
    assert packed.token_count <= packed.available_tokens
    assert packed.omitted_ids == ("neighbor",)


def test_no_seed_fits_means_insufficient():
    seed = make_doc("s", "x" * 100)
    neighbor = make_doc("n", "ok")
    bundle = EvidenceBundle((seed,), (neighbor,), ())

    packed = pack_evidence(
        bundle,
        budget=GenerationBudget(20, 5, 0),
        count_tokens=len,
        base_prompt="base",
        render_context=lambda documents: "".join(
            document.text for document in documents
        ),
    )

    assert packed.status == "insufficient_evidence"
    assert packed.documents == ()


def test_rendered_header_counts_in_budget():
    from kotaemon.indices.qa.format_context import format_evidence_unit

    seed = make_doc(
        "seed",
        "body",
        file_name="a-very-long-file-name.md",
        virtual_path="/a/long/path",
    )

    def render(documents):
        return "".join(format_evidence_unit(document)[1] for document in documents)

    rendered_seed = render((seed,))
    packed = pack_evidence(
        EvidenceBundle((seed,), (), ()),
        budget=GenerationBudget(len(rendered_seed) - 1, 0, 0),
        count_tokens=len,
        base_prompt="",
        render_context=render,
    )

    assert packed.status == "insufficient_evidence"
    assert packed.documents == ()
    assert packed.available_tokens == len(rendered_seed) - 1

    def render_with_envelope(documents):
        return "<evidence>" + "".join(
            format_evidence_unit(document)[1] for document in documents
        )

    empty_rendered = render_with_envelope(())
    empty_context = pack_evidence(
        EvidenceBundle((seed,), (), ()),
        budget=GenerationBudget(len(rendered_seed), 0, 0),
        count_tokens=len,
        base_prompt="",
        render_context=render_with_envelope,
    )
    assert empty_context.status == "insufficient_evidence"
    assert empty_context.token_count == len(empty_rendered)
