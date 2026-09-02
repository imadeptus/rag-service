from fastapi.testclient import TestClient

from rag_service.api import create_app
from rag_service.config import Settings
from rag_service.pipeline import RagPipeline


def make_client() -> TestClient:
    return TestClient(create_app(RagPipeline(Settings())))


def test_health():
    client = make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_then_ask_flow():
    client = make_client()
    resp = client.post(
        "/ingest", json={"doc_id": "runbook", "text": "Staging listens on port 5433."}
    )
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] >= 1

    resp = client.post("/ask", json={"question": "staging port?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["retrieved"]
    assert body["usage"]["prompt_tokens"] > 0


def test_validation_rejects_empty_question():
    client = make_client()
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_stats_accumulate():
    client = make_client()
    client.post("/ingest", json={"doc_id": "d", "text": "content"})
    client.post("/ask", json={"question": "content?"})
    stats = client.get("/stats").json()
    assert stats["requests"] == 1
