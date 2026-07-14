# rag-service

Provider-agnostic RAG service: hybrid retrieval (BM25 + vectors, RRF fusion), pluggable LLM
providers (DeepSeek / GigaChat / any OpenAI-compatible endpoint), citations, retrieval evals,
and per-request cost tracking. Fully testable offline — CI needs zero API keys.

## Why these design choices

- **Hybrid retrieval.** Embeddings blur exact terms (IDs, error codes, port numbers); BM25 misses
  paraphrases. Reciprocal Rank Fusion combines both rankings without score calibration.
- **Provider adapters, not a framework.** One `LLM` protocol, three implementations:
  OpenAI-compatible (covers DeepSeek, vLLM, Ollama, OpenAI), GigaChat (OAuth flow, for
  Russian closed-loop deployments), and a deterministic fake for tests. Swapping providers
  is an env var, not a rewrite.
- **Offline-first testing.** `FakeEmbedder` / `FakeLLM` are deterministic, so the entire
  pipeline — chunking, indexing, fusion, prompting, cost math, HTTP layer — is covered by
  fast tests that run anywhere. Real providers are configuration.
- **Cost is a first-class output.** Every `/ask` response reports tokens and estimated USD;
  `/stats` aggregates. You cannot manage LLM spend you don't measure.
- **Answers cite sources.** The prompt forces `[chunk_id]` citations; uncited answers are
  visible immediately in the response payload.

## Architecture

```
            ingest                                 ask
  docs ──► chunking ──► embeddings ──► store   query ──► BM25 ─┐
             (paragraph-aware,          (in-memory │            ├─► RRF ─► top-k ─► LLM ─► answer
              overlap, citations)        or Qdrant)└──► vector ─┘         context   (+usage) (+citations)
```

## Quickstart (offline, no keys)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                  # full offline test suite
python eval/run_eval.py                    # retrieval eval on the sample corpus
uvicorn rag_service.api:app --app-dir src  # API on :8000
```

```bash
curl -X POST localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"doc_id": "runbook", "text": "Staging DB listens on port 5433."}'
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "which port does staging listen on?"}'
```

## Real providers

```bash
# DeepSeek (or any OpenAI-compatible endpoint: vLLM, Ollama, OpenAI)
export LLM_PROVIDER=openai-compatible
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_API_KEY=sk-...
export LLM_MODEL=deepseek-chat

# GigaChat (Sber, OAuth)
export LLM_PROVIDER=gigachat
export GIGACHAT_AUTH_KEY=...        # base64 client_id:client_secret
export GIGACHAT_SCOPE=GIGACHAT_API_PERS

# Real embeddings
export EMB_PROVIDER=openai-compatible
export EMB_BASE_URL=... EMB_API_KEY=... EMB_MODEL=... EMB_DIM=1536

# Qdrant instead of in-memory store
export STORE_BACKEND=qdrant QDRANT_URL=http://localhost:6333
docker compose up -d qdrant
```

## Evals

`eval/run_eval.py` measures retrieval quality (hit@k, MRR) against a golden dataset
(`eval/golden.jsonl`). Run it after every retrieval change; compare configurations
(chunk size, overlap, k, embedder) by diffing the JSON reports. The eval is part of CI.

## Layout

```
src/rag_service/   config · chunking · embeddings · llm (+cost) · store · retrieval · pipeline · api
eval/              golden.jsonl + run_eval.py (hit@k, MRR)
sample_docs/       tiny corpus used by evals and the quickstart
tests/             offline suite: chunking, fusion, pipeline, cost, API
```

## Roadmap

- Faithfulness eval (LLM-as-judge over golden answers)
- Reranker stage behind the same protocol
- Async ingestion for large corpora
