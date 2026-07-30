"""Embedding providers behind one interface.

- OpenAICompatibleEmbedder: works with OpenAI, DeepSeek-hosted, vLLM, Ollama,
  or any /v1/embeddings endpoint.
- FakeEmbedder: deterministic, offline. Used in tests and CI so the whole
  pipeline is verifiable without API keys. Not semantically meaningful,
  but stable: identical text -> identical vector, shared tokens -> shared
  vector components.
"""

import hashlib
import math
from typing import Protocol

from .tokenization import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic bag-of-token-hashes embedding. Offline, for tests/CI."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _token_index(self, token: str) -> int:
        h = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big") % self.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in tokenize(text):
                vec[self._token_index(token)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 1536, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx  # lazy: only real providers need HTTP

        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


def build_embedder(settings) -> Embedder:
    if settings.emb_provider == "openai-compatible":
        return OpenAICompatibleEmbedder(
            settings.emb_base_url, settings.emb_api_key, settings.emb_model, settings.emb_dim
        )
    return FakeEmbedder(settings.emb_dim)
