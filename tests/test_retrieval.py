from rag_service.chunking import Chunk
from rag_service.embeddings import FakeEmbedder
from rag_service.retrieval import HybridRetriever
from rag_service.store import InMemoryStore


def make_retriever(docs: dict[str, str]):
    store = InMemoryStore()
    embedder = FakeEmbedder(dim=128)
    chunks = [Chunk(doc_id, f"{doc_id}#0", text, 0) for doc_id, text in docs.items()]
    vectors = embedder.embed([c.text for c in chunks])
    store.upsert(chunks, vectors)
    retriever = HybridRetriever(store, embedder)
    retriever.refresh_lexical_index()
    return retriever


DOCS = {
    "tokens": "api token rotation happens every 90 days in the partner portal",
    "sla": "incident response acknowledgement sla is 15 minutes for p1",
    "deploy": "production deploys require approval from the service owner",
}


def test_exact_term_match_ranks_first():
    retriever = make_retriever(DOCS)
    result = retriever.retrieve("token rotation", top_k=2)
    assert result[0].chunk.doc_id == "tokens"


def test_rrf_merges_both_rankers():
    retriever = make_retriever(DOCS)
    result = retriever.retrieve("api token rotation", top_k=3)
    top = result[0]
    assert "bm25" in top.sources and "vector" in top.sources


def test_top_k_respected():
    retriever = make_retriever(DOCS)
    assert len(retriever.retrieve("sla", top_k=1)) == 1


def test_empty_index_returns_nothing():
    retriever = make_retriever({})
    assert retriever.retrieve("anything", top_k=5) == []


def test_fake_embedder_is_deterministic():
    e = FakeEmbedder(dim=64)
    v1 = e.embed(["same text"])[0]
    v2 = e.embed(["same text"])[0]
    assert v1 == v2
