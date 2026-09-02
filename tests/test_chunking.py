from itertools import pairwise

from rag_service.chunking import chunk_document


def test_empty_document_produces_no_chunks():
    assert chunk_document("d", "") == []


def test_small_document_single_chunk():
    chunks = chunk_document("d", "hello world", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "d"
    assert chunks[0].text == "hello world"


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"paragraph {i} " + "word " * 30 for i in range(20))
    chunks = chunk_document("d", text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


def test_oversized_paragraph_is_hard_split():
    text = "x" * 2000
    chunks = chunk_document("d", text, chunk_size=500, overlap=100)
    assert all(len(c.text) <= 500 for c in chunks)
    assert sum(len(c.text) for c in chunks) >= 2000  # overlap duplicates some content


def test_chunk_ids_are_unique_and_positional():
    text = "\n\n".join("para " + "w " * 100 for _ in range(10))
    chunks = chunk_document("doc1", text, chunk_size=200, overlap=40)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_chunk_size():
    try:
        chunk_document("d", "text", chunk_size=100, overlap=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= chunk_size")


def test_adjacent_chunks_preserve_maximum_overlap_for_eight_paragraphs():
    paragraphs = [f"paragraph-{index} " + chr(97 + index) * 115 for index in range(8)]
    chunks = chunk_document("doc", "\n\n".join(paragraphs), chunk_size=140, overlap=30)

    assert len(chunks) >= 4
    for previous, current in pairwise(chunks):
        max_tail = min(30, 140 - len(current.text.split("\n\n")[-1]) - 2)
        expected_tail = previous.text[-max_tail:] if max_tail > 0 else ""
        assert expected_tail
        assert current.text.startswith(expected_tail)
        assert len(current.text) <= 140
