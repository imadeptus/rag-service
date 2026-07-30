"""Hybrid retrieval: BM25 (lexical) + vector search, fused with Reciprocal Rank Fusion.

Lexical search catches exact terms (IDs, error codes, names) that embeddings
blur; vectors catch paraphrases that BM25 misses. RRF combines both rankings
without score calibration: score(d) = sum over rankers of 1 / (K + rank_d).

BM25 (Okapi variant) is implemented here directly — ~40 lines, zero
dependencies, and the scoring stays inspectable.
"""

import math
from collections import Counter
from dataclasses import dataclass

from .chunking import Chunk
from .embeddings import Embedder
from .store import VectorStore
from .tokenization import tokenize

RRF_K = 60


class BM25:
    """Okapi BM25 with standard parameters k1=1.5, b=0.75."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in corpus]
        self.avg_len = (sum(self.doc_lens) / len(corpus)) if corpus else 0.0
        df: Counter = Counter()
        for doc in corpus:
            df.update(set(doc))
        n = len(corpus)
        self.idf = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for freqs, dlen in zip(self.doc_freqs, self.doc_lens):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * dlen / (self.avg_len or 1.0))
            for term in query:
                tf = freqs.get(term, 0)
                if tf:
                    score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / (tf + norm)
            scores.append(score)
        return scores


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    sources: tuple[str, ...]  # which rankers surfaced it: "bm25", "vector"


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self._bm25: BM25 | None = None
        self._bm25_chunks: list[Chunk] = []

    def refresh_lexical_index(self) -> None:
        self._bm25_chunks = self.store.all_chunks()
        corpus = [tokenize(c.text) for c in self._bm25_chunks]
        self._bm25 = BM25(corpus) if corpus else None

    def _bm25_ranking(self, query: str, top_k: int) -> list[Chunk]:
        if self._bm25 is None:
            self.refresh_lexical_index()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._bm25_chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, s in ranked[:top_k] if s > 0]

    def _vector_ranking(self, query: str, top_k: int) -> list[Chunk]:
        qvec = self.embedder.embed([query])[0]
        return [sc.chunk for sc in self.store.search(qvec, top_k)]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        pool = max(top_k * 2, 10)
        rankings = {
            "bm25": self._bm25_ranking(query, pool),
            "vector": self._vector_ranking(query, pool),
        }
        fused: dict[str, dict] = {}
        for name, ranking in rankings.items():
            for rank, chunk in enumerate(ranking):
                entry = fused.setdefault(
                    chunk.chunk_id, {"chunk": chunk, "score": 0.0, "sources": set()}
                )
                entry["score"] += 1.0 / (RRF_K + rank + 1)
                entry["sources"].add(name)
        result = [
            RetrievedChunk(e["chunk"], e["score"], tuple(sorted(e["sources"])))
            for e in fused.values()
        ]
        result.sort(key=lambda r: r.score, reverse=True)
        return result[:top_k]
