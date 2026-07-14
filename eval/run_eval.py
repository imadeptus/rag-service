"""Retrieval evaluation harness: hit@k and MRR over a golden dataset.

Golden dataset format (eval/golden.jsonl), one JSON object per line:
    {"question": "...", "relevant_doc_ids": ["doc-a", "doc-b"]}

A question is a "hit" if any retrieved chunk belongs to a relevant document.
MRR uses the rank of the first relevant chunk.

Usage:
    python eval/run_eval.py --docs sample_docs --golden eval/golden.jsonl --k 5

Runs fully offline with the fake embedder; set EMB_PROVIDER=openai-compatible
(+ keys) to evaluate real embeddings. Compare configurations by running twice
and diffing the numbers — that is the whole point of the harness.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from rag_service.config import load_settings  # noqa: E402
from rag_service.pipeline import RagPipeline  # noqa: E402


def load_golden(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="sample_docs")
    parser.add_argument("--golden", default="eval/golden.jsonl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    pipe = RagPipeline(load_settings())
    docs_dir = pathlib.Path(args.docs)
    n_docs = 0
    for f in sorted(docs_dir.glob("*.md")):
        pipe.ingest(f.stem, f.read_text(encoding="utf-8"))
        n_docs += 1

    golden = load_golden(pathlib.Path(args.golden))
    hits, rr_sum = 0, 0.0
    for row in golden:
        retrieved = pipe.retriever.retrieve(row["question"], args.k)
        relevant = set(row["relevant_doc_ids"])
        first_rank = next(
            (i + 1 for i, r in enumerate(retrieved) if r.chunk.doc_id in relevant), None
        )
        if first_rank is not None:
            hits += 1
            rr_sum += 1.0 / first_rank

    n = len(golden)
    report = {
        "documents": n_docs,
        "chunks": pipe.store.count(),
        "questions": n,
        "k": args.k,
        f"hit@{args.k}": round(hits / n, 3) if n else None,
        "mrr": round(rr_sum / n, 3) if n else None,
        "embedder": type(pipe.embedder).__name__,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
