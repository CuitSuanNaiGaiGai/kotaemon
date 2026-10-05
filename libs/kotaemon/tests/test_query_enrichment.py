from __future__ import annotations

from dataclasses import fields

import pytest

from kotaemon.indices.knowledge.planning.query_enrichment import QueryEnricher


def test_enrichment_retains_original_and_exact_code_path_and_version():
    query = "How do I configure `BGE-M3` using docs/models/v1.2/config.yaml?"

    enriched = QueryEnricher().enrich(query)

    assert enriched.original_query == query
    assert enriched.standalone_query == query
    assert enriched.variants[0] == enriched.standalone_query
    assert query in enriched.variants
    lexical_variants = [variant for variant in enriched.variants if variant != query]
    assert lexical_variants
    lexical_variant = lexical_variants[0]
    assert "BGE-M3" in lexical_variant
    assert "docs/models/v1.2/config.yaml" in lexical_variant
    assert "v1.2" in lexical_variant
    assert len(enriched.variants) <= 3


@pytest.mark.parametrize("rewritten", ["   ", None, 42])
def test_invalid_rewriter_output_falls_back_to_original(rewritten):
    enriched = QueryEnricher(rewriter=lambda _query, _history: rewritten).enrich(
        "current question"
    )

    assert enriched.standalone_query == "current question"
    assert enriched.variants[0] == "current question"
    assert "current question" in enriched.variants


def test_rewriter_exception_falls_back_to_original():
    def broken_rewriter(_query, _history):
        raise RuntimeError("rewrite failed")

    enriched = QueryEnricher(rewriter=broken_rewriter).enrich(
        "current question", user_history=("recent user turn",)
    )

    assert enriched.standalone_query == "current question"
    assert enriched.variants[0] == "current question"


def test_rewriter_receives_only_bounded_user_turns():
    calls = []

    def rewriter(query, user_turns):
        calls.append((query, user_turns))
        return "standalone follow-up"

    enriched = QueryEnricher(rewriter=rewriter).enrich(
        "and what about Linux?",
        user_history=(
            "older user turn",
            "user turn one",
            "user turn two",
            "user turn three",
        ),
    )

    assert calls == [
        (
            "and what about Linux?",
            ("user turn one", "user turn two", "user turn three"),
        )
    ]
    assert enriched.original_query == "and what about Linux?"
    assert enriched.standalone_query == "standalone follow-up"
    assert enriched.variants[0] == enriched.standalone_query
    assert enriched.original_query in enriched.variants


def test_enrichment_contract_has_no_authority_or_scope_fields():
    enriched = QueryEnricher().enrich("question")

    assert {field.name for field in fields(enriched)} == {
        "original_query",
        "standalone_query",
        "variants",
        "reason",
    }


def test_rewritten_and_original_routes_are_unique_under_normalization():
    enriched = QueryEnricher(
        rewriter=lambda _query, _history: "  CURRENT QUESTION  ", max_variants=2
    ).enrich("current question")

    assert enriched.variants == ("current question",)
