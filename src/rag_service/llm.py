"""LLM providers behind one interface, with token usage and cost tracking.

- OpenAICompatibleLLM: OpenAI / DeepSeek / vLLM / Ollama (any /v1/chat/completions).
- GigaChatLLM: Sber GigaChat (OAuth token flow, russian closed-loop deployments).
- FakeLLM: offline deterministic answers for tests/CI.

Cost is estimated from a per-model price table (USD per 1M tokens) and
accumulated by CostTracker — so every /ask response reports what it cost.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

# USD per 1M tokens (input, output). Extend freely.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "gpt-4o-mini": (0.15, 0.60),
    "GigaChat": (0.0, 0.0),  # priced in RUB by Sber; track tokens, not USD
    "fake": (0.0, 0.0),
}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CostTracker:
    requests: int = 0
    total: Usage = field(default_factory=Usage)

    def add(self, model: str, prompt_tokens: int, completion_tokens: int) -> Usage:
        p_in, p_out = PRICE_TABLE.get(model, (0.0, 0.0))
        cost = (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000
        self.requests += 1
        self.total.prompt_tokens += prompt_tokens
        self.total.completion_tokens += completion_tokens
        self.total.cost_usd += cost
        return Usage(prompt_tokens, completion_tokens, cost)


@dataclass
class LLMResult:
    text: str
    usage: Usage


class LLM(Protocol):
    def complete(self, system: str, user: str) -> LLMResult: ...


class FakeLLM:
    """Echoes a deterministic answer built from the prompt. Offline."""

    def __init__(self, tracker: CostTracker):
        self.tracker = tracker

    def complete(self, system: str, user: str) -> LLMResult:
        # Deterministic: quote the first context line as the "answer".
        first_ctx = next((ln for ln in user.splitlines() if ln.startswith("[")), "")
        text = f"(fake) Based on the context: {first_ctx[:200]}"
        usage = self.tracker.add("fake", len((system + user).split()), len(text.split()))
        return LLMResult(text=text, usage=usage)


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: str, model: str, tracker: CostTracker, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.tracker = tracker
        self.timeout = timeout

    def complete(self, system: str, user: str) -> LLMResult:
        import httpx  # lazy: only real providers need HTTP

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        u = data.get("usage", {})
        usage = self.tracker.add(self.model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return LLMResult(text=text, usage=usage)


class GigaChatLLM:
    """Sber GigaChat: OAuth access token (30 min TTL), then chat/completions."""

    AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    API_URL = "https://gigachat.devices.sberbank.ru/api/v1"

    def __init__(self, auth_key: str, scope: str, tracker: CostTracker,
                 model: str = "GigaChat", verify_ssl: bool = True, timeout: float = 60.0):
        self.auth_key = auth_key
        self.scope = scope
        self.model = model
        self.tracker = tracker
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def _get_token(self) -> str:
        import httpx  # lazy

        if self._token and time.time() < self._token_expires_at - 60:
            return self._token
        resp = httpx.post(
            self.AUTH_URL,
            headers={
                "Authorization": f"Basic {self.auth_key}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"scope": self.scope},
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires_at = data.get("expires_at", time.time() * 1000 + 25 * 60 * 1000) / 1000
        return self._token

    def complete(self, system: str, user: str) -> LLMResult:
        import httpx  # lazy

        token = self._get_token()
        resp = httpx.post(
            f"{self.API_URL}/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.2,
            },
            verify=self.verify_ssl,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        u = data.get("usage", {})
        usage = self.tracker.add(self.model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return LLMResult(text=text, usage=usage)


def build_llm(settings, tracker: CostTracker) -> LLM:
    if settings.llm_provider == "openai-compatible":
        return OpenAICompatibleLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model, tracker)
    if settings.llm_provider == "gigachat":
        return GigaChatLLM(settings.gigachat_auth_key, settings.gigachat_scope, tracker)
    return FakeLLM(tracker)
