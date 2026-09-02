"""Vector stores behind one interface: in-memory (default, offline) and Qdrant.

The in-memory store keeps the whole pipeline runnable and testable without
infrastructure; Qdrant is a drop-in for real deployments (docker-compose.yml).
"""

import math
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .chunking import Chunk
from .config import Settings


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def delete_doc(self, doc_id: str) -> None: ...

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def count(self) -> int: ...


def _chunk_from_payload(payload: dict[str, Any] | None) -> Chunk:
    """Rebuild a Chunk from a Qdrant point payload."""

    if payload is None:
        raise ValueError("Qdrant point is missing its payload")
    return Chunk(
        payload["doc_id"], payload["chunk_id"], payload["text"], payload["position"]
    )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryStore:
    """Thread-safe in-process store.

    FastAPI runs synchronous endpoints in a worker thread pool, so /ingest and
    /ask can touch this store at the same time. Without the lock, a read that
    iterates the mapping while a write inserts into it raises
    "dictionary changed size during iteration" mid-request.
    """

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}
        self._lock = threading.RLock()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        with self._lock:
            for chunk, vec in zip(chunks, vectors, strict=True):
                self._items[chunk.chunk_id] = (chunk, vec)

    def delete_doc(self, doc_id: str) -> None:
        with self._lock:
            self._items = {
                chunk_id: item
                for chunk_id, item in self._items.items()
                if item[0].doc_id != doc_id
            }

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        with self._lock:
            snapshot = list(self._items.values())
        scored = [ScoredChunk(c, _cosine(vector, v)) for c, v in snapshot]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return [c for c, _ in self._items.values()]

    def count(self) -> int:
        with self._lock:
            return len(self._items)


class QdrantStore:
    """Thin adapter over qdrant-client. Import is lazy so the dependency is optional."""

    def __init__(self, url: str, collection: str, dim: int):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                vector=v,
                payload={
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "position": c.position,
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points)

    def delete_doc(self, doc_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
        )

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        hits = self.client.query_points(self.collection, query=vector, limit=top_k).points
        out = []
        for h in hits:
            out.append(ScoredChunk(_chunk_from_payload(h.payload), h.score))
        return out

    def all_chunks(self) -> list[Chunk]:
        chunks, offset = [], None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=True
            )
            for pt in points:
                chunks.append(_chunk_from_payload(pt.payload))
            if offset is None:
                return chunks

    def count(self) -> int:
        return self.client.count(self.collection).count


def build_store(settings: Settings) -> VectorStore:
    if settings.store_backend == "qdrant":
        return QdrantStore(settings.qdrant_url, settings.collection, settings.emb_dim)
    return InMemoryStore()
