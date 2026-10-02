from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence, cast

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
        ) -> tuple[list[RetrievedDocument], str, dict[str, Exception]]:
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
                    self._filter_to_scope(vector_docs, query_scope),
                    status,
                    errors,
                )

            if self.retrieval_mode == "text":
                if not lexical_available:
                    return [], "unavailable", errors
                try:
                    docs = self.doc_store.query(
                        query, top_k=candidate_k, doc_ids=query_scope
                    )
                    lexical_docs = [
                        RetrievedDocument(**doc.to_dict(), score=-1.0) for doc in docs
                    ]
                    status = "available" if lexical_docs else "empty"
                except Exception as exc:
                    errors["lexical"] = exc
                    status = "error"
                lexical_docs = self._filter_to_scope(lexical_docs, query_scope)
                return self._deduplicate_docs(lexical_docs), status, errors

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
                    status = "available" if lexical_docs else "empty"
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
            vector_docs = self._filter_to_scope(vector_docs, query_scope)
            merged = self._deduplicate_docs(lexical_docs + vector_docs)
            return merged, status, errors

        result, lexical_status, branch_errors = retrieve_candidates(requested_scope)
        result = self._deduplicate_docs(self._filter_to_scope(result, requested_scope))

        fallback_ids: list[str] | None = None
        if visible_scope is not None:
            fallback_ids = visible_scope
        should_retry = (
            scope is not None
            and not result
            and not branch_errors
            and fallback_ids is not None
            and bool(fallback_ids)
            and set(requested_scope or ()) != set(fallback_ids)
        )
        self._trace_update(
            trace,
            scope_status="scoped" if requested_scope is not None else "global",
            scope_ids=requested_scope,
            candidate_k=candidate_k,
            lexical_status=lexical_status,
            lexical_result_count=sum(1 for item in result if item.score == -1.0),
            scope_fallback=False,
        )
        if should_retry:
            self._trace_update(
                trace,
                scope_fallback=True,
                scope_fallback_reason="zero_scoped_hits",
            )
            result, lexical_status, branch_errors = retrieve_candidates(fallback_ids)
            result = self._deduplicate_docs(self._filter_to_scope(result, fallback_ids))
            requested_scope = fallback_ids
            self._trace_update(
                trace,
                scope_status="fallback",
                scope_ids=fallback_ids,
                lexical_status=lexical_status,
                lexical_result_count=sum(1 for item in result if item.score == -1.0),
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

        result = self._filter_docs(result, top_k=top_k)
        print(f"Got raw {len(result)} retrieved documents")

        # add page thumbnails to the result if exists
        thumbnail_doc_ids: set[str] = set()
        # we should copy the text from retrieved text chunk
        # to the thumbnail to get relevant LLM score correctly
        text_thumbnail_docs: dict[str, RetrievedDocument] = {}

        non_thumbnail_docs = []
        raw_thumbnail_docs = []
        for doc in result:
            if doc.metadata.get("type") == "thumbnail":
                # change type to image to display on UI
                doc.metadata["type"] = "image"
                raw_thumbnail_docs.append(doc)
                continue
            if (
                "thumbnail_doc_id" in doc.metadata
                and len(thumbnail_doc_ids) < thumbnail_count
            ):
                thumbnail_id = doc.metadata["thumbnail_doc_id"]
                thumbnail_doc_ids.add(thumbnail_id)
                text_thumbnail_docs[thumbnail_id] = doc
            else:
                non_thumbnail_docs.append(doc)

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

        for thumbnail_doc in linked_thumbnail_docs:
            text_doc = text_thumbnail_docs[thumbnail_doc.doc_id]
            doc_dict = thumbnail_doc.to_dict()
            doc_dict["_id"] = text_doc.doc_id
            doc_dict["content"] = text_doc.content
            doc_dict["metadata"]["type"] = "image"
            for key in text_doc.metadata:
                if key not in doc_dict["metadata"]:
                    doc_dict["metadata"][key] = text_doc.metadata[key]

            additional_docs.append(RetrievedDocument(**doc_dict, score=text_doc.score))

        result = additional_docs + non_thumbnail_docs

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
