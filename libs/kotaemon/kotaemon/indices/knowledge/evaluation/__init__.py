"""Offline evaluation helpers for retrieval pipelines."""

from .retrieval_eval import (
    CaseObservation,
    Comparison,
    EvaluationCase,
    EvaluationFixture,
    IndexedCatalog,
    QueryMetrics,
    ResolvedCase,
    RunMetrics,
    compare_runs,
    load_fixture,
    resolve_judgments,
    score_case,
    score_run,
)
from .local_corpus import SourceFile, scan_sources
from .source_metrics import (
    AnchorCoverage,
    anchor_coverage,
    score_source_run,
    source_ranked_chunks,
)

__all__ = [
    "CaseObservation",
    "AnchorCoverage",
    "Comparison",
    "EvaluationCase",
    "EvaluationFixture",
    "IndexedCatalog",
    "QueryMetrics",
    "ResolvedCase",
    "RunMetrics",
    "SourceFile",
    "compare_runs",
    "load_fixture",
    "scan_sources",
    "anchor_coverage",
    "resolve_judgments",
    "score_case",
    "score_run",
    "score_source_run",
    "source_ranked_chunks",
]
