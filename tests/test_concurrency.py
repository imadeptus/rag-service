"""Concurrent ingest and ask.

FastAPI runs synchronous endpoints in a worker thread pool, so /ingest and
/ask genuinely overlap. Two failure modes were reachable:

  * the store's mapping was iterated while another thread inserted into it.
    A growing dict raises "dictionary changed size during iteration" mid-read;
  * the lexical index was published as two attributes, so a reader could pick
    up a new chunk list beside the previous BM25 model and rank the wrong
    documents.

Both are now a single reference swap behind a lock. These tests are written to
fail if either regresses: the first raises the store's real exception under a
tightened thread-switch interval, the second forces the exact interleaving
rather than hoping to hit it.
"""

import sys
import threading

import pytest

from rag_service.chunking import Chunk, chunk_document
from rag_service.config import Settings
from rag_service.embeddings import FakeEmbedder
from rag_service.pipeline import RagPipeline
from rag_service.retrieval import HybridRetriever
from rag_service.store import InMemoryStore

DOCUMENT = (
    "Staging database listens on port 5433 and is rebuilt nightly.\n\n"
    "Production replicas answer on port 5432 behind the connection pooler.\n\n"
    "Escalate paging failures to the on-call platform engineer."
)


@pytest.fixture
def fast_thread_switching():
    """Force frequent thread switches so an unsynchronised read is actually hit."""

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(previous)


def offline_pipeline() -> RagPipeline:
    return RagPipeline(
        Settings(llm_provider="fake", emb_provider="fake", store_backend="memory")
    )


def populate(store: InMemoryStore, doc_id: str, embedder: FakeEmbedder) -> None:
    chunks = chunk_document(doc_id, DOCUMENT, chunk_size=80, overlap=10)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))


# --------------------------------------------------------------------------
# store: iteration versus insertion
# --------------------------------------------------------------------------


def test_store_reads_survive_a_growing_store(fast_thread_switching):
    """all_chunks() must not blow up while another thread indexes new documents.

    The writer only ever adds documents. A stable-size store hides this bug —
    the RuntimeError is raised when the mapping *grows* mid-iteration.
    """

    store = InMemoryStore()
    embedder = FakeEmbedder(dim=32)
    for i in range(60):
        populate(store, f"seed-{i}", embedder)
    assert store.count() > 100, "need a store large enough for a read to span a write"

    errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        i = 0
        try:
            while not stop.is_set() and i < 4000:
                populate(store, f"grow-{i}", embedder)
                i += 1
        except BaseException as exc:
            errors.append(exc)
        finally:
            stop.set()

    def reader() -> None:
        # Each call is internally consistent; two calls are not a snapshot of
        # the same instant, so only the single read is asserted here.
        try:
            while not stop.is_set():
                chunks = store.all_chunks()
                assert all(c.chunk_id for c in chunks)
        except BaseException as exc:
            errors.append(exc)
            stop.set()

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, f"unsynchronised store access raised: {errors[:2]}"


# --------------------------------------------------------------------------
# lexical index: publish-in-one-step
# --------------------------------------------------------------------------


class PausingStore(InMemoryStore):
    """Blocks inside all_chunks() so a reader can run mid-refresh.

    This pins the interleaving that used to corrupt the ranking: the refresh
    has already read the chunk list but has not published the new index yet.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reading = threading.Event()
        self.may_continue = threading.Event()
        self.pause_next = False

    def all_chunks(self) -> list[Chunk]:
        chunks = super().all_chunks()
        if self.pause_next:
            self.pause_next = False
            self.reading.set()
            self.may_continue.wait(timeout=10)
        return chunks


def test_reader_sees_a_whole_index_while_a_refresh_is_in_flight():
    store = PausingStore()
    embedder = FakeEmbedder(dim=64)
    populate(store, "first", embedder)

    retriever = HybridRetriever(store, embedder)
    retriever.refresh_lexical_index()
    before = retriever._index
    assert before is not None

    populate(store, "second", embedder)
    store.pause_next = True

    failures: list[BaseException] = []
    observed: list[object] = []

    def refresher() -> None:
        try:
            retriever.refresh_lexical_index()
        except BaseException as exc:
            failures.append(exc)

    def reader() -> None:
        try:
            store.reading.wait(timeout=10)
            # The refresh is parked between reading chunks and publishing them.
            index = retriever._index
            assert index is not None
            assert len(index.chunks) == len(index.bm25.doc_lens), "torn index"
            observed.append(index)
            retriever.retrieve("staging port", top_k=3)
        except BaseException as exc:
            failures.append(exc)
        finally:
            store.may_continue.set()

    threads = [threading.Thread(target=refresher), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert not failures, f"index was read while half-updated: {failures[:2]}"
    assert observed == [before], "reader must see the previous index, not a partial one"

    after = retriever._index
    assert after is not None and after is not before
    assert len(after.chunks) == len(after.bm25.doc_lens)


def test_refresh_publishes_a_new_index_object_each_time():
    pipe = offline_pipeline()
    pipe.ingest("a", DOCUMENT)
    retriever: HybridRetriever = pipe.retriever

    first = retriever._index
    assert first is not None
    assert len(first.chunks) == len(first.bm25.doc_lens)

    pipe.ingest("b", DOCUMENT)
    second = retriever._index
    assert second is not None and second is not first
    assert len(second.chunks) == len(second.bm25.doc_lens)
    assert len(second.chunks) > len(first.chunks)


def test_ingest_and_ask_survive_each_other(fast_thread_switching):
    pipe = offline_pipeline()
    pipe.ingest("seed", DOCUMENT)

    errors: list[BaseException] = []

    def worker(action) -> None:
        try:
            for i in range(60):
                action(i)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(lambda i: pipe.ingest(f"doc-{i}", DOCUMENT),)),
        threading.Thread(target=worker, args=(lambda i: pipe.ask("staging port?"),)),
        threading.Thread(target=worker, args=(lambda i: pipe.ask("who is on call?"),)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, f"concurrent pipeline access raised: {errors[:2]}"
