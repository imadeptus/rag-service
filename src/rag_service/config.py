"""Configuration via environment variables. No secrets in code."""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # LLM provider: "openai-compatible" (OpenAI / DeepSeek / vLLM / Ollama), "gigachat", "fake"
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "fake"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "deepseek-chat"))

    # Embeddings provider: "openai-compatible", "fake"
    emb_provider: str = field(default_factory=lambda: _env("EMB_PROVIDER", "fake"))
    emb_base_url: str = field(default_factory=lambda: _env("EMB_BASE_URL", ""))
    emb_api_key: str = field(default_factory=lambda: _env("EMB_API_KEY", ""))
    emb_model: str = field(default_factory=lambda: _env("EMB_MODEL", "text-embedding-3-small"))
    emb_dim: int = field(default_factory=lambda: int(_env("EMB_DIM", "256")))

    # Vector store: "memory" or "qdrant"
    store_backend: str = field(default_factory=lambda: _env("STORE_BACKEND", "memory"))
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    collection: str = field(default_factory=lambda: _env("COLLECTION", "docs"))

    # Retrieval
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "700")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "120")))

    # GigaChat (OAuth flow)
    gigachat_auth_key: str = field(default_factory=lambda: _env("GIGACHAT_AUTH_KEY", ""))
    gigachat_scope: str = field(default_factory=lambda: _env("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"))


def load_settings() -> Settings:
    return Settings()
