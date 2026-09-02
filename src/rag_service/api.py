"""FastAPI application: /ingest, /ask, /health, /stats."""

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import load_settings
from .pipeline import RagPipeline


class IngestRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(title="rag-service", version="0.1.0")
    pipe = pipeline or RagPipeline(load_settings())

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "chunks": pipe.store.count()}

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        n = pipe.ingest(req.doc_id, req.text)
        return {"doc_id": req.doc_id, "chunks_indexed": n}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        answer = pipe.ask(req.question, req.top_k)
        return {
            "answer": answer.text,
            "citations": answer.citations,
            "retrieved": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "score": round(r.score, 5),
                    "sources": list(r.sources),
                }
                for r in answer.chunks
            ],
            "usage": {
                "prompt_tokens": answer.usage.prompt_tokens,
                "completion_tokens": answer.usage.completion_tokens,
                "cost_usd": round(answer.usage.cost_usd, 6),
            },
        }

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        t = pipe.tracker
        return {
            "requests": t.requests,
            "prompt_tokens": t.total.prompt_tokens,
            "completion_tokens": t.total.completion_tokens,
            "total_cost_usd": round(t.total.cost_usd, 6),
        }

    return app


app = create_app()
