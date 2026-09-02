"""Paragraph-aware chunking with overlap.

Splits on blank lines first, then packs paragraphs into chunks of at most
`chunk_size` characters with `overlap` characters carried between chunks.
Oversized single paragraphs are hard-split. Every chunk keeps a reference
to its source document and position for citations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    position: int


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in parts if p]


def chunk_document(
    doc_id: str, text: str, chunk_size: int = 700, overlap: int = 120
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = _split_paragraphs(text)
    # Hard-split paragraphs that alone exceed chunk_size
    units: list[str] = []
    for p in paragraphs:
        while len(p) > chunk_size:
            units.append(p[:chunk_size])
            p = p[chunk_size - overlap:]
        if p:
            units.append(p)

    chunks: list[Chunk] = []
    buf = ""
    for unit in units:
        candidate = f"{buf}\n\n{unit}" if buf else unit
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        if buf:
            chunks.append(Chunk(doc_id, f"{doc_id}#{len(chunks)}", buf, len(chunks)))
            separator = "\n\n"
            available = max(0, chunk_size - len(unit) - len(separator))
            tail_length = min(overlap, available)
            tail = buf[-tail_length:] if tail_length else ""
            buf = f"{tail}{separator}{unit}" if tail else unit
        else:
            buf = unit
    if buf:
        chunks.append(Chunk(doc_id, f"{doc_id}#{len(chunks)}", buf, len(chunks)))
    return chunks
