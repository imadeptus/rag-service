"""End-to-end RAG pipeline: ingest -> retrieve -> answer with citations."""

from dataclasses import dataclass

from .chunking import chunk_document
from .config import Settings
from .embeddings import Embedder, build_embedder
from .llm import LLM, CostTracker, Usage, build_llm
from .retrieval import HybridRetriever, RetrievedChunk
from .store import VectorStore, build_store

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions strictly from the provided context. "
    "Rules: answer ONLY from the context; if the context does not contain the answer, say "
    "\"I don't know based on the provided documents.\"; cite sources inline as [chunk_id] "
    "after each claim; never invent numbers."
)


@dataclass
class Answer:
    text: str
    citations: list[str]
    chunks: list[RetrievedChunk]
    usage: Usage


class RagPipeline:
    def __init__(self, settings: Settings, store: VectorStore | None = None,
                 embedder: Embedder | None = None, llm: LLM | None = None,
                 tracker: CostTracker | None = None):
        self.settings = settings
        self.tracker = tracker or CostTracker()
        self.embedder = embedder or build_embedder(settings)
        self.store = store or build_store(settings)
        self.llm = llm or build_llm(settings, self.tracker)
        self.retriever = HybridRetriever(self.store, self.embedder)

    def ingest(self, doc_id: str, text: str) -> int:
        chunks = chunk_document(doc_id, text, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.upsert(chunks, vectors)
        self.retriever.refresh_lexical_index()
        return len(chunks)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        k = top_k or self.settings.top_k
        retrieved = self.retriever.retrieve(question, k)
        context = "\n\n".join(f"[{r.chunk.chunk_id}] {r.chunk.text}" for r in retrieved)
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        result = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        cited = [r.chunk.chunk_id for r in retrieved if f"[{r.chunk.chunk_id}]" in result.text]
        return Answer(text=result.text, citations=cited, chunks=retrieved, usage=result.usage)
