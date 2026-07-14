from rag_service.config import Settings
from rag_service.pipeline import RagPipeline


def make_pipeline() -> RagPipeline:
    return RagPipeline(Settings())  # fake providers, in-memory store


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
