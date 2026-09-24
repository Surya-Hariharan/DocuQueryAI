"""
The contract every vector-storage backend must implement.

Keeping this interface thin and identical across backends is what lets the
rest of the app (ingestion, retrieval) stay backend-agnostic — no caller
should need to know or care whether chunks live in pgvector, FAISS, or
anything else added later. A backend that can't satisfy this contract
doesn't get to sit alongside the others as a second, divergent code path.

Document identity (get_or_create_document) is part of this contract, not a
separate concern bolted on: every chunk a backend stores must be traceable
to a stable document_id, or retrieval scoping and citations can never be
implemented correctly on top of it.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.ingestion.document_model import Chunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def get_or_create_document(
        self,
        source_uri: str,
        tenant_id: str,
        source_type: str = "url",
        checksum: Optional[str] = None
    ) -> str:
        """Resolve a source URI to a stable document_id, creating a document record if needed."""
        raise NotImplementedError

    @abstractmethod
    def upsert_chunks(
        self,
        chunks: List[Chunk],
        document_id: str,
        tenant_id: str,
        batch_size: int = 100
    ) -> None:
        """Embed and store chunks against an existing document_id, deduplicating identical text within that document."""
        raise NotImplementedError

    @abstractmethod
    def query_top_k(
        self,
        query: str,
        k: int,
        document_id: Optional[str] = None,
        tenant_id: str = None
    ) -> List[RetrievedChunk]:
        """Return the top-k most relevant chunks (with source metadata), optionally scoped to one document."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str, tenant_id: str = None) -> None:
        """Remove every chunk (and the document record) for a given document_id."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Total number of stored chunks."""
        raise NotImplementedError
