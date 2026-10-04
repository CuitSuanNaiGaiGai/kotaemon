"""Read-only retrieval over one approved local knowledge snapshot."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping

from kotaemon.embeddings import BaseEmbeddings
from kotaemon.indices.knowledge.evaluation.local_cli import (
    _ensure_contained,
    _reject_symlink_components,
    _require_ignored_repo_local_root,
    _validate_snapshot_approval,
    resolve_model_paths,
)
from kotaemon.indices.knowledge.evaluation.local_experiment import (
    _create_embedding,
    _make_bundle,
    _service,
    _validate_baseline_splitter,
    _validate_model_provenance,
)
from kotaemon.indices.knowledge.evaluation.local_models import (
    _require_offline_inference_process,
)
from kotaemon.indices.knowledge.evaluation.local_snapshot import (
    LocalSnapshot,
    load_local_snapshot,
)
from kotaemon.indices.knowledge.evaluation.source_metrics import source_ranked_chunks


@dataclass(frozen=True)
class EvidenceCard:
    source_rank: int
    chunk_rank: int
    source_id: str
    source_label: str
    locator: Mapping[str, Any]
    score: float | None
    text: str


class LocalQA:
    def __init__(self, *, snapshot: LocalSnapshot, bundle: Mapping[str, Any], service):
        self._service = service
        self._bundle = bundle
        self._chunk_to_source = bundle["chunk_to_source"]
        self._sources_by_id = {
            source["source_id"]: source for source in snapshot.selected_sources
        }

    def _locator_for_chunk(self, chunk_id: str) -> Mapping[str, Any]:
        draft = self._bundle["draft_chunks"].get(chunk_id)
        if draft is not None:
            return draft.locator
        unresolved = self._bundle["unresolved_offsets"].get(chunk_id)
        if unresolved is None:
            raise ValueError(f"retrieved chunk {chunk_id!r} has no source locator")
        return unresolved.locator

    def retrieve(self, question: str) -> tuple[EvidenceCard, ...]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must not be empty")

        candidates = self._service.search(question.strip(), top_k=20)
        representatives = source_ranked_chunks(
            candidates, self._chunk_to_source, limit=5
        )
        source_ranks = {
            self._chunk_to_source[document.doc_id]: rank
            for rank, document in enumerate(representatives, start=1)
        }
        selected_sources = set(source_ranks)

        cards = []
        for chunk_rank, document in enumerate(candidates, start=1):
            source_id = self._chunk_to_source[document.doc_id]
            if source_id not in selected_sources:
                continue
            score = document.score
            if not isinstance(score, Real) or not math.isfinite(score):
                score = None
            cards.append(
                EvidenceCard(
                    source_rank=source_ranks[source_id],
                    chunk_rank=chunk_rank,
                    source_id=source_id,
                    source_label=self._sources_by_id[source_id]["relative_path"],
                    locator=dict(self._locator_for_chunk(document.doc_id)),
                    score=score,
                    text=document.text,
                )
            )
        return tuple(cards)


def open_playground(
    local_root: Path,
    snapshot_dir: Path,
    embedding_model_dir: Path,
    reranker_model_dir: Path,
    *,
    embedding_factory: Callable[..., BaseEmbeddings] | None = None,
) -> LocalQA:
    root = Path(local_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("local root must be an existing directory")
    _require_ignored_repo_local_root(root)

    snapshots_root = _ensure_contained(root / "snapshots", root, "snapshots")
    _reject_symlink_components(snapshots_root, stop=root)
    if not snapshots_root.is_dir():
        raise ValueError("local snapshots/ directory must already exist")

    snapshot_path = _ensure_contained(snapshot_dir, snapshots_root, "snapshot")
    _reject_symlink_components(snapshot_path, stop=snapshots_root)
    _validate_snapshot_approval(snapshot_path)
    snapshot = load_local_snapshot(snapshot_path, require_reviewed=True)
    _validate_baseline_splitter(snapshot)

    paths = resolve_model_paths(root, embedding_model_dir, reranker_model_dir)
    _validate_model_provenance(paths)
    _require_offline_inference_process()
    factory = embedding_factory or _create_embedding
    embedding = factory(
        paths.embedding_model_dir,
        revision=paths.embedding_revision,
        weight_source=paths.embedding_weight_source,
    )
    bundle = _make_bundle(snapshot, embedding=embedding, chunking_arm=False)
    service = _service(bundle, rerankers=())
    return LocalQA(snapshot=snapshot, bundle=bundle, service=service)
