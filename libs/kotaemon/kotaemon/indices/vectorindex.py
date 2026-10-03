from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional, Sequence, cast

from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from theflow.settings import settings as flowsettings

from kotaemon.base import BaseComponent, Document, RetrievedDocument
from kotaemon.embeddings import BaseEmbeddings
from kotaemon.storages import BaseDocumentStore, BaseVectorStore

from .base import BaseIndexing, BaseRetrieval
from .rankings import BaseReranking, LLMReranking

VECTOR_STORE_FNAME = "vectorstore"
DOC_STORE_FNAME = "docstore"
logger = logging.getLogger(__name__)


class VectorIndexing(BaseIndexing):
    """Ingest the document, run through the embedding, and store the embedding in a
    vector store.

    This pipeline supports the following set of inputs:
        - List of documents
        - List of texts
    """

    cache_dir: Optional[str] = getattr(flowsettings, "KH_CHUNKS_OUTPUT_DIR", None)
    vector_store: BaseVectorStore
    doc_store: Optional[BaseDocumentStore] = None
    embedding: BaseEmbeddings
    count_: int = 0

    def to_retrieval_pipeline(self, *args, **kwargs):
        """Convert the indexing pipeline to a retrieval pipeline"""
        return VectorRetrieval(
            vector_store=self.vector_store,
            doc_store=self.doc_store,
            embedding=self.embedding,
            **kwargs,
        )

    def write_chunk_to_file(self, docs: list[Document]):
        # save the chunks content into markdown format
        if self.cache_dir:
            file_name = docs[0].metadata.get("file_name")
            if not file_name:
                return

            file_name = Path(file_name)
            for i in range(len(docs)):
                markdown_content = ""
                if "page_label" in docs[i].metadata:
                    page_label = str(docs[i].metadata["page_label"])
                    markdown_content += f"Page label: {page_label}"
                if "file_name" in docs[i].metadata:
                    filename = docs[i].metadata["file_name"]
                    markdown_content += f"\nFile name: {filename}"
                if "section" in docs[i].metadata:
                    section = docs[i].metadata["section"]
                    markdown_content += f"\nSection: {section}"
                if "type" in docs[i].metadata:
                    if docs[i].metadata["type"] == "image":
                        image_origin = docs[i].metadata["image_origin"]
                        image_origin = f'<p><img src="{image_origin}"></p>'
                        markdown_content += f"\nImage origin: {image_origin}"
                if docs[i].text:
                    markdown_content += f"\ntext:\n{docs[i].text}"

                with open(
                    Path(self.cache_dir) / f"{file_name.stem}_{self.count_+i}.md",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(markdown_content)

    def add_to_docstore(self, docs: list[Document]):
        if self.doc_store:
            print("Adding documents to doc store")
            self.doc_store.add(docs)

    def add_to_vectorstore(self, docs: list[Document]):
        # in case we want to skip embedding
        if self.vector_store:
            print(f"Getting embeddings for {len(docs)} nodes")
            embeddings = self.embedding(docs)
            print("Adding embeddings to vector store")
            self.vector_store.add(
                embeddings=embeddings,
                metadatas=[doc.metadata or {} for doc in docs],
                ids=[t.doc_id for t in docs],
            )

    def run(self, text: str | list[str] | Document | list[Document]):
        input_: list[Document] = []
        if not isinstance(text, list):
            text = [text]

        for item in cast(list, text):
            if isinstance(item, str):
                input_.append(Document(text=item, id_=str(uuid.uuid4())))
            elif isinstance(item, Document):
                input_.append(item)
            else:
                raise ValueError(
                    f"Invalid input type {type(item)}, should be str or Document"
                )

        self.add_to_vectorstore(input_)
        self.add_to_docstore(input_)
        self.write_chunk_to_file(input_)
        self.count_ += len(input_)


class VectorRetrieval(BaseRetrieval):
    """Retrieve list of documents from vector store"""

    vector_store: BaseVectorStore
    doc_store: Optional[BaseDocumentStore] = None
    embedding: BaseEmbeddings
    rerankers: Sequence[BaseReranking] = []
    top_k: int = 5
    first_round_top_k_mult: int = 10
    retrieval_mode: str = "hybrid"  # vector, text, hybrid

    def _filter_docs(
        self, documents: list[RetrievedDocument], top_k: int | None = None
    ):
        if top_k:
            documents = documents[:top_k]
        return documents

    @staticmethod
    def _normalize_ids(ids: Sequence[str] | None) -> list[str] | None:
        if ids is None:
            return None
        if isinstance(ids, str):
            return [ids]
        return list(dict.fromkeys(str(doc_id) for doc_id in ids if doc_id is not None))

    @staticmethod
    def _deduplicate_docs(
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        unique: list[RetrievedDocument] = []
        seen: set[str] = set()
        for document in documents:
            doc_id = document.doc_id
            if doc_id is not None and doc_id in seen:
                continue
            if doc_id is not None:
                seen.add(doc_id)
            unique.append(document)
        return unique

    @staticmethod
    def _filter_to_scope(
        documents: list[RetrievedDocument], scope: list[str] | None
    ) -> list[RetrievedDocument]:
        if scope is None:
            return documents
        allowed = set(scope)
        return [document for document in documents if document.doc_id in allowed]

    @staticmethod
    def _metadata_predicate(
        filters: Any = None, where: Any = None
    ) -> Callable[[Mapping[str, Any]], bool] | None:
        """Build a fail-closed predicate for the supported metadata-filter subset."""

        def unsupported(detail: str) -> ValueError:
            return ValueError(f"Unsupported metadata filter: {detail}")

        def combine(
            predicates: list[Callable[[Mapping[str, Any]], bool]], condition: Any
        ) -> Callable[[Mapping[str, Any]], bool]:
            if not predicates:
                raise unsupported("empty filter group")
            if condition == FilterCondition.AND:
                return lambda metadata: all(
                    predicate(metadata) for predicate in predicates
                )
            if condition == FilterCondition.OR:
                return lambda metadata: any(
                    predicate(metadata) for predicate in predicates
                )
            raise unsupported(f"unknown condition {condition!r}")

        def parse_llama_filters(
            node: Any,
        ) -> Callable[[Mapping[str, Any]], bool]:
            if isinstance(node, MetadataFilter):
                key = node.key
                operator = node.operator
                expected = node.value
                if not isinstance(key, str) or not key:
                    raise unsupported("metadata key must be a non-empty string")
                if operator == FilterOperator.EQ:
                    return (
                        lambda metadata: key in metadata and metadata[key] == expected
                    )
                if operator == FilterOperator.IN:
                    if not isinstance(expected, (list, tuple)):
                        raise unsupported("IN requires a list of values")
                    return lambda metadata: key in metadata and any(
                        metadata[key] == value for value in expected
                    )
                raise unsupported(f"operator {operator!r}")

            if isinstance(node, MetadataFilters):
                return combine(
                    [parse_llama_filters(child) for child in node.filters],
                    node.condition,
                )

            raise unsupported(f"expected MetadataFilter(s), got {type(node).__name__}")

        def parse_chroma_where(
            node: Any,
        ) -> Callable[[Mapping[str, Any]], bool]:
            if not isinstance(node, Mapping) or not node:
                raise unsupported("where must be a non-empty mapping")

            if len(node) == 1:
                operator, children = next(iter(node.items()))
                if operator in {"$and", "$or"}:
                    if not isinstance(children, (list, tuple)):
                        raise unsupported(f"{operator} requires a list of filters")
                    return combine(
                        [parse_chroma_where(child) for child in children],
                        (
                            FilterCondition.AND
                            if operator == "$and"
                            else FilterCondition.OR
                        ),
                    )

            if len(node) != 1:
                raise unsupported("where clauses must use one explicit operator")
            key, constraint = next(iter(node.items()))
            if not isinstance(key, str) or not key or key.startswith("$"):
                raise unsupported("metadata key must be a non-empty string")
            if not isinstance(constraint, Mapping) or len(constraint) != 1:
                raise unsupported(f"where clause for {key!r} must use one operator")
            operator, expected = next(iter(constraint.items()))
            if operator == "$eq":
                return lambda metadata: key in metadata and metadata[key] == expected
            if operator == "$in":
                if not isinstance(expected, (list, tuple)):
                    raise unsupported("$in requires a list of values")
                return lambda metadata: key in metadata and any(
                    metadata[key] == value for value in expected
                )
            raise unsupported(f"where operator {operator!r}")

        predicates = []
        if filters is not None:
            predicates.append(parse_llama_filters(filters))
        if where is not None:
            predicates.append(parse_chroma_where(where))
        if not predicates:
            return None
        return combine(predicates, FilterCondition.AND)

    @staticmethod
    def _filter_to_metadata(
        documents: list[RetrievedDocument],
        predicate: Callable[[Mapping[str, Any]], bool] | None,
    ) -> list[RetrievedDocument]:
        if predicate is None:
            return documents
        return [
            document for document in documents if predicate(document.metadata or {})
        ]

    @staticmethod
    def _lexical_available(doc_store: BaseDocumentStore) -> bool:
        """Honor an explicit backend capability flag and retain legacy adapters."""
        capability = getattr(doc_store, "supports_lexical_search", None)
        if capability is not None:
            return bool(capability)
        query_method = getattr(type(doc_store), "query", None)
        return callable(query_method) and query_method is not BaseDocumentStore.query

    @staticmethod
    def _trace_update(trace: dict[str, Any] | None, **fields):
        if trace is None:
            return
        try:
            trace.update(fields)
        except Exception:
            logger.exception("Could not update optional retrieval trace")

    def _vector_candidates(
        self,
        embedding: list[float],
        candidate_k: int,
        scope: list[str] | None,
        query_kwargs: dict[str, Any],
    ) -> list[RetrievedDocument]:
        assert self.doc_store is not None
        _, scores, ids = self.vector_store.query(
            embedding=embedding,
            top_k=candidate_k,
            ids=scope,
            **query_kwargs,
        )

        # Store adapters may return documents in a different order than requested.
        # Associate every score with the vector result ID before loading documents.
        score_by_id: dict[str, float] = {}
        unique_ids: list[str] = []
        seen: set[str] = set()
        for index, doc_id in enumerate(ids):
            if doc_id is None or doc_id in seen:
                continue
            if scope is not None and doc_id not in scope:
                continue
            seen.add(doc_id)
            unique_ids.append(doc_id)
            score_by_id[doc_id] = scores[index] if index < len(scores) else -1.0

        if not unique_ids:
            return []

        docs_by_id = {doc.doc_id: doc for doc in self.doc_store.get(unique_ids)}
        return [
            RetrievedDocument(**docs_by_id[doc_id].to_dict(), score=score_by_id[doc_id])
            for doc_id in unique_ids
            if doc_id in docs_by_id
        ]

    def run(
        self,
        text: str | Document,
        top_k: Optional[int] = None,
        scope: Sequence[str] | None = None,
        fallback_scope: Sequence[str] | None = None,
        trace: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[RetrievedDocument]:
        """Retrieve a list of documents from vector store

        Args:
            text: the text to retrieve similar documents
            top_k: number of top similar documents to return

        Returns:
            list[RetrievedDocument]: list of retrieved documents
        """
        if top_k is None:
            top_k = self.top_k

        do_extend = kwargs.pop("do_extend", False)
        thumbnail_count = kwargs.pop("thumbnail_count", 3)
        legacy_doc_ids = kwargs.pop("doc_ids", None)
        if scope is None and legacy_doc_ids is not None:
            scope = legacy_doc_ids

        metadata_filter = self._metadata_predicate(
            filters=kwargs.get("filters"), where=kwargs.get("where")
        )
        if kwargs.get("where_document") is not None:
            raise ValueError(
                "Unsupported metadata filter: where_document is not supported"
            )

        # ``scope`` is the planner's exact chunk scope. ``fallback_scope`` is the
        # caller-visible chunk allowlist. Even a global/ambiguous plan is restricted
        # to that allowlist when one was provided.
        requested_scope = self._normalize_ids(scope)
        visible_scope = self._normalize_ids(fallback_scope)
        if requested_scope is not None and visible_scope is not None:
            visible_ids = set(visible_scope)
            requested_scope = [
                doc_id for doc_id in requested_scope if doc_id in visible_ids
            ]
        elif requested_scope is None:
            requested_scope = visible_scope

        if requested_scope == []:
            self._trace_update(
                trace,
                scope_status="empty",
                lexical_status="not_queried",
                scope_fallback=False,
            )
            return []

        if do_extend:
            top_k_first_round = top_k * self.first_round_top_k_mult
        else:
            top_k_first_round = top_k

        # Scoped calls need spare candidates so post-filtering remains effective
        # when a backend ignores IDs or metadata filters.
        candidate_k = top_k_first_round
        if requested_scope is not None:
            candidate_k = max(candidate_k, top_k * self.first_round_top_k_mult)

        if self.doc_store is None:
            raise ValueError(
                "doc_store is not provided. Please provide a doc_store to "
                "retrieve the documents"
            )

        query = text.text if isinstance(text, Document) else text
        lexical_available = self._lexical_available(self.doc_store)
        lexical_status = "not_used"
        branch_errors: dict[str, Exception] = {}
        query_embedding = None
        if self.retrieval_mode in {"vector", "hybrid"}:
            try:
                query_embedding = self.embedding(text)[0].embedding
            except Exception as exc:
                self._trace_update(
                    trace, branch_errors={"vector": f"{type(exc).__name__}: {exc}"}
                )
                raise

        def retrieve_candidates(
            query_scope: list[str] | None,
        ) -> tuple[list[RetrievedDocument], str, dict[str, Exception], int]:
            errors: dict[str, Exception] = {}
            vector_docs: list[RetrievedDocument] = []
            lexical_docs: list[RetrievedDocument] = []
            status = "not_used"

            if self.retrieval_mode == "vector":
                try:
                    vector_docs = self._vector_candidates(
                        query_embedding, candidate_k, query_scope, kwargs
                    )
                except Exception as exc:
                    errors["vector"] = exc
                return (
                    self._filter_to_metadata(
                        self._filter_to_scope(vector_docs, query_scope),
                        metadata_filter,
                    ),
                    status,
                    errors,
                    0,
                )

            if self.retrieval_mode == "text":
                if not lexical_available:
                    return [], "unavailable", errors, 0
                try:
                    docs = self.doc_store.query(
                        query, top_k=candidate_k, doc_ids=query_scope
                    )
                    lexical_docs = [
                        RetrievedDocument(**doc.to_dict(), score=-1.0) for doc in docs
                    ]
                except Exception as exc:
                    errors["lexical"] = exc
                    status = "error"
                lexical_docs = self._filter_to_scope(lexical_docs, query_scope)
                lexical_docs = self._filter_to_metadata(lexical_docs, metadata_filter)
                if status != "error":
                    status = "available" if lexical_docs else "empty"
                lexical_docs = self._deduplicate_docs(lexical_docs)
                return lexical_docs, status, errors, len(lexical_docs)

            if self.retrieval_mode != "hybrid":
                raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

            if not lexical_available:
                status = "unavailable"

            def query_vectorstore():
                nonlocal vector_docs
                try:
                    vector_docs = self._vector_candidates(
                        query_embedding, candidate_k, query_scope, kwargs
                    )
                except Exception as exc:
                    errors["vector"] = exc

            def query_docstore():
                nonlocal lexical_docs
                nonlocal status
                if not lexical_available:
                    return
                try:
                    docs = self.doc_store.query(
                        query, top_k=candidate_k, doc_ids=query_scope
                    )
                    lexical_docs = [
                        RetrievedDocument(**doc.to_dict(), score=-1.0) for doc in docs
                    ]
                except Exception as exc:
                    errors["lexical"] = exc
                    status = "error"

            vector_thread = threading.Thread(target=query_vectorstore)
            vector_thread.start()
            lexical_thread = None
            if lexical_available:
                lexical_thread = threading.Thread(target=query_docstore)
                lexical_thread.start()

            vector_thread.join()
            if lexical_thread is not None:
                lexical_thread.join()

            lexical_docs = self._filter_to_scope(lexical_docs, query_scope)
            lexical_docs = self._filter_to_metadata(lexical_docs, metadata_filter)
            if lexical_available and "lexical" not in errors:
                status = "available" if lexical_docs else "empty"
            vector_docs = self._filter_to_scope(vector_docs, query_scope)
            vector_docs = self._filter_to_metadata(vector_docs, metadata_filter)
            merged = self._deduplicate_docs(lexical_docs + vector_docs)
            return merged, status, errors, len(lexical_docs)

        result, lexical_status, branch_errors, lexical_result_count = (
            retrieve_candidates(requested_scope)
        )
        result = self._deduplicate_docs(self._filter_to_scope(result, requested_scope))
        result = self._filter_to_metadata(result, metadata_filter)

        # A missing caller allowlist means the caller has not supplied an
        # additional visibility boundary, so a zero-hit planned scope may retry
        # globally. An explicit empty allowlist returned above and never reaches
        # this point.
        fallback_ids = visible_scope
        should_retry = (
            scope is not None
            and not result
            and not branch_errors
            and (fallback_ids is None or bool(fallback_ids))
            and set(requested_scope or ()) != set(fallback_ids or ())
        )
        self._trace_update(
            trace,
            scope_status="scoped" if requested_scope is not None else "global",
            scope_ids=requested_scope,
            candidate_k=candidate_k,
            lexical_status=lexical_status,
            lexical_result_count=lexical_result_count,
            scope_fallback=False,
        )
        if should_retry:
            self._trace_update(
                trace,
                scope_fallback=True,
                scope_fallback_reason="zero_scoped_hits",
            )
            result, lexical_status, branch_errors, lexical_result_count = (
                retrieve_candidates(fallback_ids)
            )
            result = self._deduplicate_docs(self._filter_to_scope(result, fallback_ids))
            result = self._filter_to_metadata(result, metadata_filter)
            requested_scope = fallback_ids
            self._trace_update(
                trace,
                scope_status="fallback",
                scope_ids=fallback_ids,
                lexical_status=lexical_status,
                lexical_result_count=lexical_result_count,
            )

        if branch_errors:
            error_summary = {
                branch: f"{type(error).__name__}: {error}"
                for branch, error in branch_errors.items()
            }
            self._trace_update(trace, branch_errors=error_summary)
            if not result:
                if self.retrieval_mode == "hybrid":
                    details = "; ".join(
                        f"{branch}: {message}"
                        for branch, message in error_summary.items()
                    )
                    raise RuntimeError(
                        f"retrieval branches failed: {details}"
                    ) from next(iter(branch_errors.values()))
                raise next(iter(branch_errors.values()))

        if lexical_status == "unavailable":
            self._trace_update(trace, lexical_status="unavailable")
        elif lexical_status == "empty":
            self._trace_update(trace, lexical_status="empty")

        # use additional reranker to re-order the document list
        if self.rerankers and text:
            for reranker in self.rerankers:
                # if reranker is LLMReranking, limit the document with top_k items only
                if isinstance(reranker, LLMReranking):
                    result = self._filter_docs(result, top_k=top_k)
                result = reranker.run(documents=result, query=text)
                result = self._deduplicate_docs(
                    self._filter_to_scope(result, requested_scope)
                )
                result = self._filter_to_metadata(result, metadata_filter)

        result = self._filter_docs(result, top_k=top_k)
        print(f"Got raw {len(result)} retrieved documents")

        # add page thumbnails to the result if exists
        thumbnail_doc_ids: set[str] = set()
        # we should copy the text from retrieved text chunk
        # to the thumbnail to get relevant LLM score correctly
        text_thumbnail_docs: dict[str, RetrievedDocument] = {}
        allowed_thumbnail_ids = (
            set(requested_scope) if requested_scope is not None else None
        )

        non_thumbnail_docs = []
        raw_thumbnail_docs = []
        for doc in result:
            if doc.metadata.get("type") == "thumbnail":
                # change type to image to display on UI
                doc.metadata["type"] = "image"
                raw_thumbnail_docs.append(doc)
                continue
            non_thumbnail_docs.append(doc)
            thumbnail_id = doc.metadata.get("thumbnail_doc_id")
            if not isinstance(thumbnail_id, str) or not thumbnail_id:
                continue
            if (
                allowed_thumbnail_ids is not None
                and thumbnail_id not in allowed_thumbnail_ids
            ):
                continue
            if len(thumbnail_doc_ids) < thumbnail_count:
                thumbnail_doc_ids.add(thumbnail_id)
                text_thumbnail_docs[thumbnail_id] = doc

        linked_thumbnail_docs = (
            self.doc_store.get(list(thumbnail_doc_ids)) if thumbnail_doc_ids else []
        )
        print(
            "thumbnail docs",
            len(linked_thumbnail_docs),
            "non-thumbnail docs",
            len(non_thumbnail_docs),
            "raw-thumbnail docs",
            len(raw_thumbnail_docs),
        )
        additional_docs = []
        replaced_text_doc_ids = set()

        for thumbnail_doc in linked_thumbnail_docs:
            text_doc = text_thumbnail_docs.get(thumbnail_doc.doc_id)
            if text_doc is None:
                continue

            text_file_id = text_doc.metadata.get("file_id")
            thumbnail_file_id = thumbnail_doc.metadata.get("file_id")
            if (
                "file_id" in text_doc.metadata
                and "file_id" in thumbnail_doc.metadata
                and text_file_id != thumbnail_file_id
            ):
                continue

            doc_dict = thumbnail_doc.to_dict()
            doc_dict["id_"] = text_doc.doc_id
            doc_dict["content"] = text_doc.content
            doc_dict["metadata"]["type"] = "image"
            for key in text_doc.metadata:
                if key not in doc_dict["metadata"]:
                    doc_dict["metadata"][key] = text_doc.metadata[key]

            image_doc = RetrievedDocument(**doc_dict, score=text_doc.score)
            if metadata_filter is not None and not metadata_filter(
                image_doc.metadata or {}
            ):
                continue
            additional_docs.append(image_doc)
            replaced_text_doc_ids.add(text_doc.doc_id)

        result = additional_docs + [
            doc for doc in non_thumbnail_docs if doc.doc_id not in replaced_text_doc_ids
        ]

        if not result:
            # return output from raw retrieved thumbnails
            result = self._filter_docs(raw_thumbnail_docs, top_k=thumbnail_count)

        return result


class TextVectorQA(BaseComponent):
    retrieving_pipeline: BaseRetrieval
    qa_pipeline: BaseComponent

    def run(self, question, **kwargs):
        retrieved_documents = self.retrieving_pipeline(question, **kwargs)
        return self.qa_pipeline(question, retrieved_documents, **kwargs)
