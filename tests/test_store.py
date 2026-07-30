from rag_service.chunking import Chunk
from rag_service.store import InMemoryStore


def test_in_memory_delete_doc_removes_only_target_document():
    store = InMemoryStore()
    chunks = [
        Chunk("a", "a#0", "old", 0),
        Chunk("b", "b#0", "keep", 0),
    ]
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])

    store.delete_doc("a")

    assert [chunk.doc_id for chunk in store.all_chunks()] == ["b"]
