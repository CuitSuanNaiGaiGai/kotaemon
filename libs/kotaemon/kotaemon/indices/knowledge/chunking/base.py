"""Source-aware chunking interfaces and semantic-unit construction."""

from typing import Protocol

from kotaemon.base import Document

from ..metadata import normalize_knowledge_metadata


class ChunkStrategy(Protocol):
    def split(self, document: Document) -> list[Document]:
        """Split a source document into independent knowledge units."""
        ...


def semantic_unit(document: Document, text: str, section_path: list[str]) -> Document:
    """Copy source metadata onto a unit with a distinct semantic identity."""
    unit = Document(
        text=text,
        metadata=normalize_knowledge_metadata(document),
        source=document.source,
        channel=document.channel,
    )
    unit.metadata.update(section_path=list(section_path), parent_id=unit.doc_id)
    return unit
