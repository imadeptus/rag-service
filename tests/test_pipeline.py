from rag_service.config import Settings
from rag_service.llm import LLMResult, Usage
from rag_service.pipeline import RagPipeline


def make_pipeline() -> RagPipeline:
    return RagPipeline(Settings())  # fake providers, in-memory store


class RecordingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> LLMResult:
        self.calls += 1
        return LLMResult("unexpected", Usage(1, 1, 1.0))


def test_ingest_returns_chunk_count():
    pipe = make_pipeline()
    n = pipe.ingest("doc1", "First paragraph.\n\nSecond paragraph about tokens.")
    assert n >= 1
    assert pipe.store.count() == n


def test_ask_returns_answer_with_retrieved_context():
    pipe = make_pipeline()
    pipe.ingest("runbook", "The staging database listens on port 5433.")
    answer = pipe.ask("which port does staging listen on?")
    assert answer.text
    assert answer.chunks
    assert answer.chunks[0].chunk.doc_id == "runbook"


def test_usage_is_tracked_per_request_and_globally():
    pipe = make_pipeline()
    pipe.ingest("d", "some content here")
    a1 = pipe.ask("some content")
    a2 = pipe.ask("some content")
    assert a1.usage.prompt_tokens > 0
    assert pipe.tracker.requests == 2
    assert pipe.tracker.total.prompt_tokens >= a1.usage.prompt_tokens + a2.usage.prompt_tokens


def test_reingest_same_doc_does_not_duplicate():
    pipe = make_pipeline()
    n1 = pipe.ingest("doc1", "same text")
    n2 = pipe.ingest("doc1", "same text")
    assert n1 == n2
    assert pipe.store.count() == n1


def test_reingest_replaces_stale_chunks_and_content():
    pipe = make_pipeline()
    assert pipe.ingest("doc1", "obsolete " * 1000) >= 10

    new_count = pipe.ingest("doc1", "current content")

    chunks = [chunk for chunk in pipe.store.all_chunks() if chunk.doc_id == "doc1"]
    assert pipe.store.count() == new_count == 1
    assert chunks[0].text == "current content"
    assert all("obsolete" not in chunk.text for chunk in chunks)
    retrieved = pipe.retriever.retrieve("obsolete", top_k=20)
    assert all("obsolete" not in result.chunk.text for result in retrieved)


def test_reingest_with_empty_document_removes_stale_lexical_results():
    pipe = make_pipeline()
    pipe.ingest("doc1", "obsolete content")

    assert pipe.ingest("doc1", "") == 0

    assert pipe.store.count() == 0
    assert pipe.retriever.retrieve("obsolete", top_k=5) == []


def test_irrelevant_question_abstains_without_llm_call():
    llm = RecordingLLM()
    pipe = RagPipeline(Settings(), llm=llm)
    pipe.ingest("known", "alpha beta gamma")

    answer = pipe.ask("offcorpus987654321")

    assert answer.text == "I don't know based on the provided documents."
    assert answer.chunks == []
    assert answer.citations == []
    assert answer.usage.prompt_tokens == 0
    assert answer.usage.completion_tokens == 0
    assert answer.usage.cost_usd == 0.0
    assert llm.calls == 0
