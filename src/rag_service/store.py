"""Vector stores behind one interface: in-memory (default, offline) and Qdrant.

The in-memory store keeps the whole pipeline runnable and testable without
infrastructure; Qdrant is a drop-in for real deployments (docker-compose.yml).
"""

import math
import uuid
from dataclasses import dataclass
from typing import Protocol

from .chunking import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def count(self) -> int: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryStore:
    def __init__(self):
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for chunk, vec in zip(chunks, vectors):
            self._items[chunk.chunk_id] = (chunk, vec)

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        scored = [ScoredChunk(c, _cosine(vector, v)) for c, v in self._items.values()]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def all_chunks(self) -> list[Chunk]:
        return [c for c, _ in self._items.values()]

    def count(self) -> int:
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
                payload={"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text, "position": c.position},
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(self.collection, points)

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        hits = self.client.query_points(self.collection, query=vector, limit=top_k).points
        out = []
        for h in hits:
            p = h.payload
            out.append(ScoredChunk(Chunk(p["doc_id"], p["chunk_id"], p["text"], p["position"]), h.score))
        return out

    def all_chunks(self) -> list[Chunk]:
        chunks, offset = [], None
        while True:
            points, offset = self.client.scroll(self.collection, limit=256, offset=offset, with_payload=True)
            for pt in points:
                p = pt.payload
                chunks.append(Chunk(p["doc_id"], p["chunk_id"], p["text"], p["position"]))
            if offset is None:
                return chunks

    def count(self) -> int:
        return self.client.count(self.collection).count


def build_store(settings):
    if settings.store_backend == "qdrant":
        return QdrantStore(settings.qdrant_url, settings.collection, settings.emb_dim)
    return InMemoryStore()
