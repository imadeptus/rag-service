"""Retrieval evaluation harness: hit@k and MRR over a golden dataset.

Golden dataset format (eval/golden.jsonl), one JSON object per line:
    {"question": "...", "relevant_doc_ids": ["doc-a", "doc-b"]}

A question is a "hit" if any retrieved chunk belongs to a relevant document.
MRR uses the rank of the first relevant chunk.

Usage:
    python eval/run_eval.py --k 3 --min-hit 0.85 --min-mrr 0.7

Runs fully offline with fake providers, independently of provider environment
variables. Compare configurations by running twice and diffing the reports.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from rag_service.config import Settings  # noqa: E402
from rag_service.pipeline import RagPipeline  # noqa: E402


def load_golden(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate(
    docs_dir: pathlib.Path,
    golden_path: pathlib.Path,
    k: int,
) -> dict[str, int | float | str | None]:
    pipe = RagPipeline(
        Settings(llm_provider="fake", emb_provider="fake", store_backend="memory")
    )
    n_docs = 0
    for file_path in sorted(docs_dir.glob("*.md")):
        pipe.ingest(file_path.stem, file_path.read_text(encoding="utf-8"))
        n_docs += 1

    golden = load_golden(golden_path)
    hits, rr_sum = 0, 0.0
    for row in golden:
        retrieved = pipe.retriever.retrieve(row["question"], k)
        relevant = set(row["relevant_doc_ids"])
        first_rank = next(
            (i + 1 for i, result in enumerate(retrieved) if result.chunk.doc_id in relevant),
            None,
        )
        if first_rank is not None:
            hits += 1
            rr_sum += 1.0 / first_rank

    question_count = len(golden)
    hit_rate = hits / question_count if question_count else 0.0
    mrr = rr_sum / question_count if question_count else 0.0
    return {
        "documents": n_docs,
        "chunks": pipe.store.count(),
        "questions": question_count,
        "k": k,
        f"hit@{k}": round(hit_rate, 3),
        "mrr": round(mrr, 3),
        "embedder": type(pipe.embedder).__name__,
        "status": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="sample_docs")
    parser.add_argument("--golden", default="eval/golden.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-hit", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    args = parser.parse_args(argv)

    report = evaluate(pathlib.Path(args.docs), pathlib.Path(args.golden), args.k)
    passed = (
        float(report[f"hit@{args.k}"]) >= args.min_hit
        and float(report["mrr"]) >= args.min_mrr
    )
    report["status"] = "PASS" if passed else "FAIL"
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
