"""Run controlled, local four-arm retrieval experiments on a reviewed snapshot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from kotaemon.base import Document, DocumentWithEmbedding
from kotaemon.embeddings import BaseEmbeddings
from kotaemon.indices import VectorIndexing, VectorRetrieval
from kotaemon.indices.knowledge.chunking.registry import get_chunk_strategy
from kotaemon.indices.knowledge.evaluation.local_models import (
    EMBEDDING_MODEL_ID,
    RERANKER_MODEL_ID,
    BgeM3Embeddings,
    BgeM3Reranking,
    LocalModelPaths,
)
from kotaemon.indices.knowledge.evaluation.local_ingest import DraftChunk
from kotaemon.indices.knowledge.evaluation.local_snapshot import LocalSnapshot
from kotaemon.indices.knowledge.evaluation.retrieval_eval import (
    IndexedCatalog,
    ResolvedCase,
    resolve_judgments,
)
from kotaemon.indices.knowledge.evaluation.source_metrics import (
    anchor_coverage,
    score_source_run,
    source_ranked_chunks,
)
from kotaemon.indices.knowledge.metadata import infer_source_type
from kotaemon.indices.knowledge.planning.query_planner import (
    KnowledgeSource,
)
from kotaemon.indices.knowledge.planning.retrieval_plan import RetrievalPlan
from kotaemon.indices.knowledge.retrieval import KnowledgeService, RetrievalTrace
from kotaemon.indices.rankings import BaseReranking
from kotaemon.indices.splitters import TokenSplitter
from kotaemon.storages import InMemoryDocumentStore, InMemoryVectorStore

K = 5
CANDIDATE_K = 20
ANCHOR_COVERAGE_CUTOFF = "top_5_distinct_sources_per_query"
ANCHOR_DENOMINATOR_SEMANTICS = "covered / (covered + uncovered); unresolved excluded"
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_METRIC_FIELDS = ("hit_at_k", "recall_at_k", "mrr_at_k", "wrong_scope_at_k")


@dataclass(frozen=True)
class ExperimentAnchorCoverage:
    """Anchor coverage with unresolved offsets excluded from its denominator."""

    total_anchors: int
    covered_anchors: int
    uncovered_anchor_ids: tuple[str, ...]
    unresolved_anchor_ids: tuple[str, ...]
    rate: float | None

    @property
    def uncovered_anchors(self) -> int:
        return len(self.uncovered_anchor_ids)

    @property
    def unresolved_anchors(self) -> int:
        return len(self.unresolved_anchor_ids)

    @property
    def coverage_denominator(self) -> int:
        return self.covered_anchors + self.uncovered_anchors


@dataclass(frozen=True)
class _UnresolvedChunkOffsets:
    source_id: str
    unit_id: str
    locator: Mapping[str, Any]
    candidate_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ArmReport:
    """Per-arm retrieval evidence and source-level scores.

    Anchor coverage uses each query's first K distinct-source result projection;
    ``anchor_coverage_cutoff`` records that cutoff in serialized reports.
    """

    component_config: Mapping[str, Any]
    config_fingerprint: str
    query_count: int
    query_ids: tuple[str, ...]
    vector_candidate_ids: Mapping[str, tuple[str, ...]]
    vector_candidate_counts: Mapping[str, int]
    input_candidate_ids: Mapping[str, tuple[str, ...]]
    final_result_ids: Mapping[str, tuple[str, ...]]
    metrics: Any
    anchor_coverage: ExperimentAnchorCoverage
    anchor_coverage_cutoff: str
    trace_artifacts: Mapping[str, str]


@dataclass(frozen=True)
class ExperimentReport:
    """Reviewable summary of one four-arm local experiment."""

    k: int
    candidate_k: int
    query_count: int
    snapshot_fingerprint: str
    arms: Mapping[str, ArmReport]
    metric_baseline: Mapping[str, float | None]
    metric_deltas: Mapping[str, Mapping[str, float | None]]
    wrong_scope_numerators: Mapping[str, int]
    wrong_scope_denominators: Mapping[str, int]
    artifact_manifest: tuple[Mapping[str, Any], ...]


class _HashedFeatureEmbeddings(BaseEmbeddings):
    """Stable, dependency-free hashed-feature control for baseline arms."""

    dimension: int = 256

    def __init__(self, dimension: int = 256):
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def run(self, text, *args, **kwargs):
        if isinstance(text, (str, Document)):
            values = [text]
        elif isinstance(text, list) and all(
            isinstance(item, (str, Document)) for item in text
        ):
            values = text
        else:
            raise TypeError("embedding input must be a string, Document, or list")

        result = []
        for value in values:
            document = value if isinstance(value, Document) else Document(content=value)
            vector = [0.0] * self.dimension
            for token in _TOKEN_RE.findall(document.text.casefold()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[bucket] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
            else:
                vector[0] = 1.0
            result.append(DocumentWithEmbedding(content=document, embedding=vector))
        return result


def _create_embedding(
    model_dir: Path,
    *,
    revision: str,
    weight_source: Literal["cached", "downloaded"],
) -> BaseEmbeddings:
    """Factory isolated for synthetic tests; the default loads local weights."""
    return BgeM3Embeddings(
        model_path=model_dir,
        revision=revision,
        weight_source=weight_source,
    )


def _create_reranker(
    model_dir: Path,
    *,
    revision: str,
    weight_source: Literal["cached", "downloaded"],
) -> BaseReranking:
    """Factory isolated for synthetic tests; the default loads local weights."""
    return BgeM3Reranking(
        model_path=model_dir,
        revision=revision,
        weight_source=weight_source,
    )


class _SnapshotCatalog:
    """Small in-memory source and chunk catalog built only from snapshot data."""

    def __init__(
        self, sources: Sequence[KnowledgeSource], chunk_map: Mapping[str, Sequence[str]]
    ):
        self._sources = tuple(sources)
        self._chunk_map = {key: tuple(value) for key, value in chunk_map.items()}

    def list_sources(self, allowed_source_ids=None):
        if allowed_source_ids is None:
            return list(self._sources)
        allowed = set(allowed_source_ids)
        return [source for source in self._sources if source.source_id in allowed]

    def chunk_ids(self, source_ids, relation_type="document"):
        if relation_type != "document":
            raise ValueError(
                "The local experiment catalog supports document chunks only"
            )
        return {
            source_id: list(self._chunk_map.get(source_id, ()))
            for source_id in source_ids
        }


class _GlobalIdentityPlanner:
    """Keep each query global while retaining explicit reviewed constraints."""

    def plan(
        self,
        query: str,
        catalog,
        *,
        path: str | None = None,
        source_types: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        allowed_source_ids: Sequence[str] | None = None,
    ) -> RetrievalPlan:
        del catalog, allowed_source_ids
        return RetrievalPlan(
            query=query,
            semantic_query=query,
            source_types=tuple(source_types or ()),
            virtual_paths=() if path is None else (path,),
            metadata_filters=dict(filters or {}),
            confidence=0.0,
            reason="global identity plan; caller constraints remain enforced",
            source_ids=None,
        )


class _AuditedReranker:
    """Capture the actual ordered input passed through VectorRetrieval."""

    def __init__(self, delegate: BaseReranking):
        self.delegate = delegate
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def name(self):
        return getattr(self.delegate, "name", type(self.delegate).__name__)

    def run(self, documents, query):
        self.calls.append((query, tuple(_document_id(doc) for doc in documents)))
        return self.delegate.run(documents=documents, query=query)


def _document_id(document: Any) -> str:
    value = getattr(document, "doc_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("Retrieval produced a document without a stable ID")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(config)).hexdigest()


def _verify_snapshot(snapshot: LocalSnapshot) -> None:
    if not isinstance(snapshot, LocalSnapshot):
        raise TypeError("snapshot must be a validated LocalSnapshot")
    manifest_path = snapshot.root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "The reviewed snapshot manifest is unavailable or invalid"
        ) from error
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != snapshot.fingerprint:
        raise ValueError("The reviewed snapshot manifest changed after validation")
    if not isinstance(manifest, dict) or manifest.get("review_status") != "approved":
        raise ValueError("run_local_experiment requires an approved snapshot")


def _validate_baseline_splitter(snapshot: LocalSnapshot) -> None:
    configuration = snapshot.records.get("splitter_configuration")
    expected = {
        "version": "main-token-only-v1",
        "name": "TokenSplitter",
        "chunk_size": 1024,
        "chunk_overlap": 256,
        "separator": "\n\n",
        "backup_separators": ("\n", ".", " ", "\u200b"),
    }
    if not isinstance(configuration, Mapping):
        raise ValueError(
            "The reviewed snapshot does not match the required baseline splitter configuration"
        )
    for key, expected_value in expected.items():
        actual_value = configuration.get(key)
        if key == "backup_separators":
            actual_value = (
                tuple(actual_value) if isinstance(actual_value, Sequence) else None
            )
        if actual_value != expected_value:
            raise ValueError(
                "The reviewed snapshot does not match the required baseline splitter configuration"
            )


def _validate_model_provenance(model_paths: LocalModelPaths) -> None:
    entries = (
        (
            "embedding",
            model_paths.embedding_revision,
            model_paths.embedding_weight_source,
        ),
        (
            "reranker",
            model_paths.reranker_revision,
            model_paths.reranker_weight_source,
        ),
    )
    for label, revision, weight_source in entries:
        if (
            not isinstance(revision, str)
            or not revision.strip()
            or weight_source not in ("cached", "downloaded")
        ):
            raise ValueError(
                f"Local {label} requires a verified model revision and weight source"
            )


def _model_metadata(model: Any, label: str) -> dict[str, Any]:
    metadata = getattr(model, "metadata", None)
    if is_dataclass(metadata):
        payload = asdict(metadata)
    elif isinstance(metadata, Mapping):
        payload = dict(metadata)
    else:
        raise ValueError(f"The local {label} adapter did not expose model metadata")
    required = {
        "model_id",
        "revision",
        "weights_sha256",
        "device",
        "flag_embedding_version",
        "query_max_length",
        "passage_max_length",
        "weight_source",
    }
    if not required.issubset(payload):
        raise ValueError(f"The local {label} adapter metadata is incomplete")
    return json.loads(_canonical_json(payload))


def _text_spans(source_text: str, chunk_text: str) -> tuple[tuple[int, int], ...]:
    """Return every exact occurrence so ambiguous offsets stay unresolved."""
    spans = []
    start = source_text.find(chunk_text)
    while start >= 0:
        spans.append((start, start + len(chunk_text)))
        start = source_text.find(chunk_text, start + 1)
    return tuple(spans)


def _build_chunk_documents(
    snapshot: LocalSnapshot,
    *,
    chunking_arm: bool,
) -> tuple[
    list[Document],
    dict[str, str],
    dict[str, tuple[str, ...]],
    dict[str, str],
    dict[str, DraftChunk],
    dict[str, _UnresolvedChunkOffsets],
]:
    source_by_id = {source["source_id"]: source for source in snapshot.selected_sources}
    units = [
        unit
        for unit in snapshot.records["source_units"]
        if unit["source_id"] in source_by_id
    ]
    if not units:
        raise ValueError("The reviewed snapshot has no selected source units")

    token_splitter = TokenSplitter(
        chunk_size=1024,
        chunk_overlap=256,
        separator="\n\n",
        backup_separators=["\n", ".", " ", "\u200b"],
    )
    strategies = {}
    chunks: list[Document] = []
    chunk_to_source: dict[str, str] = {}
    source_chunks: dict[str, list[str]] = {}
    source_types: dict[str, str] = {}
    draft_chunks: dict[str, DraftChunk] = {}
    unresolved_offsets: dict[str, _UnresolvedChunkOffsets] = {}
    for unit in units:
        source_id = unit["source_id"]
        source = source_by_id[source_id]
        source_type = infer_source_type({}, "source" + source["suffix"])
        source_types[source_id] = source_type
        metadata = {
            "source_id": source_id,
            "unit_id": unit["unit_id"],
            "unit_ordinal": unit["unit_ordinal"],
            "document_id": source_id,
            "virtual_path": "/",
            "source_type": source_type,
            "section_path": [],
            "entity": {},
            **dict(unit["locator"]),
        }
        source_document = Document(
            text=unit["normalized_text"],
            id_=unit["unit_id"],
            metadata=metadata,
        )
        if chunking_arm:
            strategy = strategies.get(source_type)
            if strategy is None:
                strategy = get_chunk_strategy(source_type, token_splitter)
                strategies[source_type] = strategy
        else:
            strategy = get_chunk_strategy("other", token_splitter)

        split_documents = strategy.split(source_document)
        for ordinal, split_document in enumerate(split_documents):
            if not split_document.text:
                continue
            chunk_id = (
                "local-"
                + hashlib.sha256(
                    f"{unit['unit_id']}\0{ordinal}\0{split_document.text}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:24]
            )
            chunk_metadata = dict(split_document.metadata or {})
            chunk_metadata.update(
                source_id=source_id,
                unit_id=unit["unit_id"],
                unit_ordinal=unit["unit_ordinal"],
                source_type=source_type,
                virtual_path="/",
                document_id=source_id,
                chunk_id=chunk_id,
                parent_id=unit["unit_id"],
            )
            chunks.append(
                Document(
                    text=split_document.text,
                    id_=chunk_id,
                    metadata=chunk_metadata,
                )
            )
            spans = _text_spans(unit["normalized_text"], split_document.text)
            if len(spans) == 1:
                span = spans[0]
                draft_chunks[chunk_id] = DraftChunk(
                    chunk_id=chunk_id,
                    source_id=source_id,
                    relative_path=unit["relative_path"],
                    unit_id=unit["unit_id"],
                    unit_ordinal=unit["unit_ordinal"],
                    chunk_ordinal=ordinal,
                    locator=dict(unit["locator"]),
                    text=split_document.text,
                    char_start=span[0],
                    char_end=span[1],
                )
            else:
                unresolved_offsets[chunk_id] = _UnresolvedChunkOffsets(
                    source_id=source_id,
                    unit_id=unit["unit_id"],
                    locator=dict(unit["locator"]),
                    candidate_spans=spans,
                )
            chunk_to_source[chunk_id] = source_id
            source_chunks.setdefault(source_id, []).append(chunk_id)

    if not chunks:
        raise ValueError("The reviewed snapshot produced no non-empty chunks")
    duplicate_ids = len(chunk_to_source) != len(chunks)
    if duplicate_ids:
        raise ValueError("Chunk generation produced duplicate chunk IDs")
    chunk_map = {
        source_id: tuple(sorted(ids)) for source_id, ids in source_chunks.items()
    }
    return (
        chunks,
        chunk_to_source,
        chunk_map,
        source_types,
        draft_chunks,
        unresolved_offsets,
    )


def _make_bundle(
    snapshot: LocalSnapshot,
    *,
    embedding: BaseEmbeddings,
    chunking_arm: bool,
):
    (
        chunks,
        chunk_to_source,
        chunk_map,
        source_types,
        draft_chunks,
        unresolved_offsets,
    ) = _build_chunk_documents(snapshot, chunking_arm=chunking_arm)
    vector_store = InMemoryVectorStore()
    doc_store = InMemoryDocumentStore()
    indexer = VectorIndexing(
        vector_store=vector_store,
        doc_store=doc_store,
        embedding=embedding,
    )
    indexer.run(chunks)
    catalog = _SnapshotCatalog(
        [
            KnowledgeSource(
                source_id=source["source_id"],
                source_type=infer_source_type({}, "source" + source["suffix"]),
                virtual_path="/",
                entity={},
            )
            for source in snapshot.selected_sources
        ],
        chunk_map,
    )
    sources_by_id = {
        source["source_id"]: source for source in snapshot.selected_sources
    }
    return {
        "chunks": chunks,
        "chunk_to_source": chunk_to_source,
        "chunk_map": chunk_map,
        "source_types": source_types,
        "draft_chunks": draft_chunks,
        "unresolved_offsets": unresolved_offsets,
        "vector_store": indexer.vector_store,
        "doc_store": indexer.doc_store,
        "embedding": embedding,
        "catalog": catalog,
        "sources_by_id": sources_by_id,
    }


def _service(bundle, *, rerankers=()):
    retrieval = VectorRetrieval(
        vector_store=bundle["vector_store"],
        doc_store=bundle["doc_store"],
        embedding=bundle["embedding"],
        rerankers=list(rerankers),
        top_k=CANDIDATE_K,
        first_round_top_k_mult=1,
        retrieval_mode="vector",
        max_per_parent_or_section=None,
    )
    return KnowledgeService(
        planner=_GlobalIdentityPlanner(),
        catalog=bundle["catalog"],
        retriever=retrieval,
        docstore=bundle["doc_store"],
    )


def _trace_candidate_ids(
    trace_data: Mapping[str, Any], case_id: str
) -> tuple[str, ...]:
    events = trace_data.get("events")
    if not isinstance(events, list):
        raise ValueError(f"Query {case_id!r}: trace events are missing or ambiguous")
    no_search_events = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("stage") == "no_search"
    ]
    if no_search_events:
        if len(no_search_events) != 1:
            raise ValueError(f"Query {case_id!r}: multiple no-search events")
        reason = no_search_events[0].get("reason")
        if reason not in {
            "empty_visibility",
            "explicit_filters_no_match",
            "no_visible_sources",
        }:
            raise ValueError(f"Query {case_id!r}: unsupported no-search trace")
        if trace_data.get("attempts", []) not in ([], None):
            raise ValueError(f"Query {case_id!r}: no-search trace contains attempts")
        if trace_data.get("merged_ids", []) not in ([], None):
            raise ValueError(f"Query {case_id!r}: no-search trace contains merged IDs")
        if trace_data.get("scope_fallback") is True:
            raise ValueError(f"Query {case_id!r}: no-search trace used fallback")
        if any(
            isinstance(event, Mapping)
            and event.get("stage") in {"recall_attempt", "scope_fallback", "merged"}
            for event in events
        ):
            raise ValueError(
                f"Query {case_id!r}: no-search trace contains search events"
            )
        return ()
    if trace_data.get("scope_fallback") is not False:
        raise ValueError(f"Query {case_id!r}: expected no retrieval fallback")
    if trace_data.get("candidate_k") != CANDIDATE_K:
        raise ValueError(f"Query {case_id!r}: trace candidate_k is not {CANDIDATE_K}")
    attempts = trace_data.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        raise ValueError(f"Query {case_id!r}: expected exactly one retrieval attempt")
    attempt = attempts[0]
    if not isinstance(attempt, Mapping):
        raise ValueError(f"Query {case_id!r}: retrieval attempt has an invalid shape")
    vector_status = attempt.get("vector_status")
    if vector_status not in {"available", "empty"}:
        raise ValueError(f"Query {case_id!r}: vector recall was not available")
    if attempt.get("lexical_status") != "not_used":
        raise ValueError(
            f"Query {case_id!r}: vector-only retrieval used a lexical branch"
        )
    vector_rows = attempt.get("vector_candidates")
    if not isinstance(vector_rows, list):
        raise ValueError(f"Query {case_id!r}: vector_candidates trace is ambiguous")
    candidate_ids = []
    for row in vector_rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("id"), str)
            or not row["id"]
        ):
            raise ValueError(f"Query {case_id!r}: malformed vector candidate trace")
        candidate_ids.append(row["id"])
    if (vector_status == "empty") != (not candidate_ids):
        raise ValueError(
            f"Query {case_id!r}: vector status conflicts with candidate count"
        )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"Query {case_id!r}: vector candidate trace has duplicate IDs")
    merged_ids = trace_data.get("merged_ids")
    if not isinstance(merged_ids, list) or tuple(merged_ids) != tuple(candidate_ids):
        raise ValueError(f"Query {case_id!r}: merged IDs differ from vector candidates")
    return tuple(candidate_ids)


def _case_search(service, case: ResolvedCase, trace: RetrievalTrace):
    return service.search(
        case.query,
        path=case.path,
        source_types=case.source_types,
        filters=case.filters,
        allowed_source_ids=case.allowed_source_ids,
        top_k=CANDIDATE_K,
        trace=trace,
    )


def _anchor_coverage_for_results(
    anchors: Sequence[Any],
    ranked_chunks: Sequence[Document],
    chunk_to_source: Mapping[str, str],
    draft_chunks: Mapping[str, DraftChunk],
    unresolved_offsets: Mapping[str, _UnresolvedChunkOffsets],
) -> ExperimentAnchorCoverage:
    """Measure anchors in every returned chunk from the first K source window."""
    representatives = source_ranked_chunks(ranked_chunks, chunk_to_source, limit=K)
    source_window = {chunk_to_source[_document_id(chunk)] for chunk in representatives}
    selected_chunks = tuple(
        chunk
        for chunk in ranked_chunks
        if chunk_to_source[_document_id(chunk)] in source_window
    )
    selected_drafts = [
        draft_chunks[chunk_id]
        for chunk in selected_chunks
        if (chunk_id := _document_id(chunk)) in draft_chunks
    ]
    known_coverage = anchor_coverage(anchors, selected_drafts)
    unmapped_ids = set(known_coverage.unmapped_anchor_ids)
    unresolved_ids = set()
    for anchor in anchors:
        if anchor.id not in unmapped_ids:
            continue
        for chunk in selected_chunks:
            offsets = unresolved_offsets.get(_document_id(chunk))
            if (
                offsets is None
                or offsets.source_id != anchor.source_id
                or offsets.unit_id != anchor.unit_id
                or offsets.locator != anchor.locator
            ):
                continue
            if not offsets.candidate_spans or any(
                start <= anchor.char_start < anchor.char_end <= end
                for start, end in offsets.candidate_spans
            ):
                unresolved_ids.add(anchor.id)
                break

    uncovered_ids = tuple(
        anchor.id
        for anchor in anchors
        if anchor.id in unmapped_ids and anchor.id not in unresolved_ids
    )
    unresolved_anchor_ids = tuple(
        anchor.id for anchor in anchors if anchor.id in unresolved_ids
    )
    covered = known_coverage.total_anchors - len(unmapped_ids)
    denominator = covered + len(uncovered_ids)
    return ExperimentAnchorCoverage(
        total_anchors=known_coverage.total_anchors,
        covered_anchors=covered,
        uncovered_anchor_ids=uncovered_ids,
        unresolved_anchor_ids=unresolved_anchor_ids,
        rate=None if denominator == 0 else covered / denominator,
    )


def _run_arm(
    name: str,
    config: Mapping[str, Any],
    service: KnowledgeService,
    cases: Sequence[ResolvedCase],
    chunk_to_source: Mapping[str, str],
    draft_chunks: Mapping[str, DraftChunk],
    unresolved_offsets: Mapping[str, _UnresolvedChunkOffsets],
    anchors: Sequence[Any],
    *,
    artifact_dir: Path,
    baseline_candidates: Mapping[str, tuple[str, ...]] | None = None,
    audited_reranker: _AuditedReranker | None = None,
):
    vector_ids: dict[str, tuple[str, ...]] = {}
    vector_counts: dict[str, int] = {}
    input_ids: dict[str, tuple[str, ...]] = {}
    final_ids: dict[str, tuple[str, ...]] = {}
    results: dict[str, tuple[Document, ...]] = {}
    trace_paths: dict[str, str] = {}
    reranker_call_index = 0
    for index, case in enumerate(cases):
        trace = RetrievalTrace()
        retrieved = _case_search(service, case, trace)
        trace_data = trace.to_dict()
        candidates = _trace_candidate_ids(trace_data, case.case_id)
        vector_ids[case.case_id] = candidates
        vector_counts[case.case_id] = len(candidates)
        if baseline_candidates is not None:
            expected = baseline_candidates.get(case.case_id)
            if expected is None or candidates != expected:
                raise ValueError(
                    f"Reranker arm changed baseline vector candidates for {case.case_id!r}"
                )
        if audited_reranker is not None:
            if candidates:
                if reranker_call_index >= len(audited_reranker.calls):
                    raise ValueError(f"Reranker was not called for {case.case_id!r}")
                called_query, received = audited_reranker.calls[reranker_call_index]
                if called_query != case.query or received != candidates:
                    raise ValueError(
                        f"Reranker input did not match the ordered vector candidates for {case.case_id!r}"
                    )
                reranker_call_index += 1
            else:
                expected_empty_call = (case.query, ())
                if (
                    reranker_call_index < len(audited_reranker.calls)
                    and audited_reranker.calls[reranker_call_index]
                    == expected_empty_call
                ):
                    reranker_call_index += 1
            input_ids[case.case_id] = candidates
        ids = tuple(_document_id(document) for document in retrieved)
        final_ids[case.case_id] = ids
        results[case.case_id] = tuple(retrieved)
        relative_trace = Path("traces") / name / f"query-{index:04d}.json"
        trace_path = artifact_dir / relative_trace
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_bytes(_canonical_json(trace_data) + b"\n")
        trace_paths[case.case_id] = relative_trace.as_posix()

    if audited_reranker is not None and reranker_call_index != len(
        audited_reranker.calls
    ):
        raise ValueError("Reranker received an unexpected candidate window")
    metrics = score_source_run(
        cases,
        results,
        k=K,
        chunk_to_source=chunk_to_source,
    )
    anchors_by_query: dict[str, list[Any]] = {}
    for anchor in anchors:
        anchors_by_query.setdefault(anchor.query_id, []).append(anchor)
    coverage_by_query = []
    for case in cases:
        coverage_by_query.append(
            _anchor_coverage_for_results(
                anchors_by_query.get(case.case_id, ()),
                results[case.case_id],
                chunk_to_source,
                draft_chunks,
                unresolved_offsets,
            )
        )
    uncovered_anchor_ids = tuple(
        anchor_id
        for coverage in coverage_by_query
        for anchor_id in coverage.uncovered_anchor_ids
    )
    unresolved_anchor_ids = tuple(
        anchor_id
        for coverage in coverage_by_query
        for anchor_id in coverage.unresolved_anchor_ids
    )
    total_anchors = sum(coverage.total_anchors for coverage in coverage_by_query)
    covered_anchors = sum(coverage.covered_anchors for coverage in coverage_by_query)
    coverage_denominator = covered_anchors + len(uncovered_anchor_ids)
    arm_anchor_coverage = ExperimentAnchorCoverage(
        total_anchors=total_anchors,
        covered_anchors=covered_anchors,
        uncovered_anchor_ids=uncovered_anchor_ids,
        unresolved_anchor_ids=unresolved_anchor_ids,
        rate=(
            None
            if coverage_denominator == 0
            else covered_anchors / coverage_denominator
        ),
    )
    return ArmReport(
        component_config=json.loads(_canonical_json(config)),
        config_fingerprint=_fingerprint(config),
        query_count=len(cases),
        query_ids=tuple(case.case_id for case in cases),
        vector_candidate_ids=vector_ids,
        vector_candidate_counts=vector_counts,
        input_candidate_ids=input_ids,
        final_result_ids=final_ids,
        metrics=metrics,
        anchor_coverage=arm_anchor_coverage,
        anchor_coverage_cutoff=ANCHOR_COVERAGE_CUTOFF,
        trace_artifacts=trace_paths,
    )


def _arm_to_dict(arm: ArmReport) -> dict[str, Any]:
    return {
        "component_config": dict(arm.component_config),
        "config_fingerprint": arm.config_fingerprint,
        "query_count": arm.query_count,
        "query_ids": list(arm.query_ids),
        "vector_candidate_ids": {
            query_id: list(ids) for query_id, ids in arm.vector_candidate_ids.items()
        },
        "vector_candidate_counts": dict(arm.vector_candidate_counts),
        "input_candidate_ids": {
            query_id: list(ids) for query_id, ids in arm.input_candidate_ids.items()
        },
        "final_result_ids": {
            query_id: list(ids) for query_id, ids in arm.final_result_ids.items()
        },
        "metrics": arm.metrics.to_dict(),
        "anchor_coverage": {
            "cutoff": arm.anchor_coverage_cutoff,
            "total_anchors": arm.anchor_coverage.total_anchors,
            "covered_anchors": arm.anchor_coverage.covered_anchors,
            "uncovered_anchors": arm.anchor_coverage.uncovered_anchors,
            "unresolved_anchors": arm.anchor_coverage.unresolved_anchors,
            "coverage_denominator": arm.anchor_coverage.coverage_denominator,
            "uncovered_anchor_ids": list(arm.anchor_coverage.uncovered_anchor_ids),
            "unresolved_anchor_ids": list(arm.anchor_coverage.unresolved_anchor_ids),
            "denominator_semantics": ANCHOR_DENOMINATOR_SEMANTICS,
            "rate": arm.anchor_coverage.rate,
        },
        "trace_artifacts": dict(arm.trace_artifacts),
    }


def _report_to_dict(report: ExperimentReport) -> dict[str, Any]:
    return {
        "k": report.k,
        "candidate_k": report.candidate_k,
        "query_count": report.query_count,
        "snapshot_fingerprint": report.snapshot_fingerprint,
        "arms": {name: _arm_to_dict(arm) for name, arm in report.arms.items()},
        "metric_baseline": dict(report.metric_baseline),
        "metric_deltas": {
            name: dict(values) for name, values in report.metric_deltas.items()
        },
        "wrong_scope_numerators": dict(report.wrong_scope_numerators),
        "wrong_scope_denominators": dict(report.wrong_scope_denominators),
        "artifact_manifest": [dict(entry) for entry in report.artifact_manifest],
    }


def _run_local_experiment_in_directory(
    snapshot: LocalSnapshot,
    *,
    embedding: BaseEmbeddings,
    reranker_model: BaseReranking,
    artifact_dir: Path,
) -> ExperimentReport:
    """Write a complete experiment into an unpublished staging directory."""
    root = artifact_dir

    hashed_embedding = _HashedFeatureEmbeddings()
    baseline_bundle = _make_bundle(
        snapshot,
        embedding=hashed_embedding,
        chunking_arm=False,
    )
    generated_catalog = IndexedCatalog(
        baseline_bundle["sources_by_id"], baseline_bundle["chunk_to_source"]
    )
    resolved_cases = resolve_judgments(snapshot.judgments, generated_catalog)

    common_retrieval = {
        "mode": "vector",
        "top_k": CANDIDATE_K,
        "candidate_k": CANDIDATE_K,
        "first_round_top_k_mult": 1,
        "do_extend": False,
        "max_per_parent_or_section": None,
    }
    baseline_config = {
        "chunking": {
            "strategy": "token",
            "chunk_size": 1024,
            "chunk_overlap": 256,
            "separator": "\n\n",
            "backup_separators": ["\n", ".", " ", "\u200b"],
        },
        "embedding": {"method": "hashed_feature_v1", "dimension": 256},
        "reranker": None,
        "retrieval": common_retrieval,
    }
    chunking_config = {
        **baseline_config,
        "chunking": {
            "strategy": "registry",
            "chunk_size": 1024,
            "chunk_overlap": 256,
            "separator": "\n\n",
            "backup_separators": ["\n", ".", " ", "\u200b"],
            "source_strategies": {
                source_type: type(
                    get_chunk_strategy(
                        source_type,
                        TokenSplitter(
                            1024,
                            256,
                            "\n\n",
                            backup_separators=["\n", ".", " ", "\u200b"],
                        ),
                    )
                ).__name__
                for source_type in sorted(set(baseline_bundle["source_types"].values()))
            },
        },
    }
    embedding_config = {
        **baseline_config,
        "embedding": {
            "method": EMBEDDING_MODEL_ID,
            "metadata": _model_metadata(embedding, "embedding"),
        },
    }
    reranker_config = {
        **baseline_config,
        "reranker": {
            "model": RERANKER_MODEL_ID,
            "metadata": _model_metadata(reranker_model, "reranker"),
        },
    }

    artifact_entries: list[dict[str, Any]] = []
    baseline_arm = _run_arm(
        "baseline",
        baseline_config,
        _service(baseline_bundle),
        resolved_cases,
        baseline_bundle["chunk_to_source"],
        baseline_bundle["draft_chunks"],
        baseline_bundle["unresolved_offsets"],
        snapshot.anchors,
        artifact_dir=root,
    )
    artifact_entries.extend(_trace_entries(root, baseline_arm))
    baseline_candidates = baseline_arm.vector_candidate_ids

    chunking_bundle = _make_bundle(
        snapshot,
        embedding=hashed_embedding,
        chunking_arm=True,
    )
    chunking_arm = _run_arm(
        "chunking",
        chunking_config,
        _service(chunking_bundle),
        resolved_cases,
        chunking_bundle["chunk_to_source"],
        chunking_bundle["draft_chunks"],
        chunking_bundle["unresolved_offsets"],
        snapshot.anchors,
        artifact_dir=root,
    )
    artifact_entries.extend(_trace_entries(root, chunking_arm))

    embedding_bundle = _make_bundle(
        snapshot,
        embedding=embedding,
        chunking_arm=False,
    )
    embedding_arm = _run_arm(
        "embedding",
        embedding_config,
        _service(embedding_bundle),
        resolved_cases,
        embedding_bundle["chunk_to_source"],
        embedding_bundle["draft_chunks"],
        embedding_bundle["unresolved_offsets"],
        snapshot.anchors,
        artifact_dir=root,
    )
    artifact_entries.extend(_trace_entries(root, embedding_arm))

    audited_reranker = _AuditedReranker(reranker_model)
    reranker_arm = _run_arm(
        "reranker",
        reranker_config,
        _service(baseline_bundle, rerankers=(audited_reranker,)),
        resolved_cases,
        baseline_bundle["chunk_to_source"],
        baseline_bundle["draft_chunks"],
        baseline_bundle["unresolved_offsets"],
        snapshot.anchors,
        artifact_dir=root,
        baseline_candidates=baseline_candidates,
        audited_reranker=audited_reranker,
    )
    artifact_entries.extend(_trace_entries(root, reranker_arm))

    arms = {
        "baseline": baseline_arm,
        "chunking": chunking_arm,
        "embedding": embedding_arm,
        "reranker": reranker_arm,
    }
    baseline_metrics = {
        name: getattr(baseline_arm.metrics, name) for name in _METRIC_FIELDS
    }
    metric_deltas = {
        arm_name: {
            metric_name: (
                None
                if getattr(arm.metrics, metric_name) is None
                or baseline_metrics[metric_name] is None
                else getattr(arm.metrics, metric_name) - baseline_metrics[metric_name]
            )
            for metric_name in _METRIC_FIELDS
        }
        for arm_name, arm in arms.items()
    }
    wrong_scope_numerators = {
        arm_name: sum(query.wrong_scope_numerator for query in arm.metrics.per_query)
        for arm_name, arm in arms.items()
    }
    wrong_scope_denominators = {
        arm_name: sum(query.wrong_scope_denominator for query in arm.metrics.per_query)
        for arm_name, arm in arms.items()
    }
    report = ExperimentReport(
        k=K,
        candidate_k=CANDIDATE_K,
        query_count=len(resolved_cases),
        snapshot_fingerprint=snapshot.fingerprint,
        arms=arms,
        metric_baseline=baseline_metrics,
        metric_deltas=metric_deltas,
        wrong_scope_numerators=wrong_scope_numerators,
        wrong_scope_denominators=wrong_scope_denominators,
        artifact_manifest=tuple(artifact_entries),
    )
    report_path = root / "report.json"
    report_bytes = _canonical_json(_report_to_dict(report)) + b"\n"
    report_path.write_bytes(report_bytes)
    sidecar = {
        "schema_version": 1,
        "snapshot_fingerprint": snapshot.fingerprint,
        "arm_configurations": {
            name: {
                "component_config": dict(arm.component_config),
                "config_fingerprint": arm.config_fingerprint,
            }
            for name, arm in arms.items()
        },
        "artifacts": [
            *artifact_entries,
            {
                "path": "report.json",
                "sha256": hashlib.sha256(report_bytes).hexdigest(),
                "size_bytes": len(report_bytes),
                "kind": "experiment_report",
            },
        ],
    }
    (root / "artifact-manifest.json").write_bytes(_canonical_json(sidecar) + b"\n")
    return report


@contextmanager
def _staged_artifact_directory(destination: Path):
    """Publish a complete sibling directory with one atomic rename."""
    if destination.exists():
        raise FileExistsError(
            f"Experiment artifact destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        yield staging
        if destination.exists():
            raise FileExistsError(
                f"Experiment artifact destination appeared during the run: {destination}"
            )
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def run_local_experiment(
    snapshot: LocalSnapshot,
    *,
    model_paths: LocalModelPaths,
    artifact_dir: Path,
) -> ExperimentReport:
    """Run and atomically publish the four isolated retrieval experiment arms."""
    _verify_snapshot(snapshot)
    _validate_baseline_splitter(snapshot)
    if not isinstance(model_paths, LocalModelPaths):
        raise TypeError("model_paths must be LocalModelPaths")
    cases = snapshot.judgments.cases
    if not cases:
        raise ValueError("The reviewed snapshot has no judged queries")
    if any(case.judgment_level != "source" for case in cases):
        raise ValueError("The local experiment requires source-level judgments")
    unsupported_cases = [
        case
        for case in cases
        if case.path not in (None, "/")
        or (case.filters is not None and bool(case.filters))
    ]
    if unsupported_cases:
        case = unsupported_cases[0]
        raise ValueError(
            "The reviewed snapshot has no reviewed logical metadata for path or "
            f"entity filtering required by query {case.case_id!r}; use path=None "
            "or '/' and empty metadata filters, or add reviewed logical metadata "
            "to the snapshot schema."
        )
    _validate_model_provenance(model_paths)

    destination = Path(artifact_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"Experiment artifact destination already exists: {destination}"
        )
    embedding = _create_embedding(
        model_paths.embedding_model_dir,
        revision=model_paths.embedding_revision,
        weight_source=model_paths.embedding_weight_source,
    )
    reranker_model = _create_reranker(
        model_paths.reranker_model_dir,
        revision=model_paths.reranker_revision,
        weight_source=model_paths.reranker_weight_source,
    )
    with _staged_artifact_directory(destination) as staging:
        return _run_local_experiment_in_directory(
            snapshot,
            embedding=embedding,
            reranker_model=reranker_model,
            artifact_dir=staging,
        )


def _trace_entries(root: Path, arm: ArmReport) -> list[dict[str, Any]]:
    entries = []
    for relative_path in arm.trace_artifacts.values():
        path = root / relative_path
        payload = path.read_bytes()
        entries.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "kind": "retrieval_trace",
            }
        )
    return entries


__all__ = ["ArmReport", "ExperimentReport", "run_local_experiment"]
