"""Real-provider adapters against a mocked transport.

These cover the code paths that only run when the service talks to an actual
endpoint: request shape, auth headers, response parsing, cost accounting and
the GigaChat token lifecycle. httpx.MockTransport keeps them fully offline.
"""

import httpx
import pytest

from rag_service.embeddings import OpenAICompatibleEmbedder
from rag_service.llm import CostTracker, GigaChatLLM, OpenAICompatibleLLM


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------
# OpenAI-compatible chat completions
# --------------------------------------------------------------------------


def test_openai_llm_sends_expected_request_and_parses_usage():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "42 is the answer"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            },
        )

    tracker = CostTracker()
    llm = OpenAICompatibleLLM(
        "https://api.deepseek.com/v1/",  # trailing slash must be normalised
        "secret-key",
        "deepseek-chat",
        tracker,
        client=client_for(handler),
    )
    result = llm.complete("system prompt", "user prompt")

    assert result.text == "42 is the answer"
    assert result.usage.prompt_tokens == 1000
    assert result.usage.completion_tokens == 500

    request = seen[0]
    assert str(request.url) == "https://api.deepseek.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret-key"

    import json

    body = json.loads(request.content)
    assert body["model"] == "deepseek-chat"
    assert body["temperature"] == 0.2
    assert body["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]


def test_openai_llm_prices_tokens_from_the_table():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
            },
        )

    tracker = CostTracker()
    llm = OpenAICompatibleLLM(
        "https://x/v1", "k", "deepseek-chat", tracker, client=client_for(handler)
    )
    usage = llm.complete("s", "u").usage

    # deepseek-chat is priced at 0.27 in / 1.10 out per 1M tokens.
    assert usage.cost_usd == pytest.approx(1.37)
    assert tracker.requests == 1
    assert tracker.total.cost_usd == pytest.approx(1.37)


def test_unknown_model_is_tracked_but_costs_nothing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    tracker = CostTracker()
    llm = OpenAICompatibleLLM(
        "https://x/v1", "k", "some-new-model", tracker, client=client_for(handler)
    )
    usage = llm.complete("s", "u").usage

    assert usage.cost_usd == 0.0
    assert tracker.total.prompt_tokens == 10


def test_missing_usage_block_defaults_to_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = OpenAICompatibleLLM(
        "https://x/v1", "k", "deepseek-chat", CostTracker(), client=client_for(handler)
    )
    usage = llm.complete("s", "u").usage

    assert (usage.prompt_tokens, usage.completion_tokens, usage.cost_usd) == (0, 0, 0.0)


def test_provider_error_surfaces_and_is_not_counted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "upstream unavailable"})

    tracker = CostTracker()
    llm = OpenAICompatibleLLM(
        "https://x/v1", "k", "deepseek-chat", tracker, client=client_for(handler)
    )

    with pytest.raises(httpx.HTTPStatusError):
        llm.complete("s", "u")
    assert tracker.requests == 0


# --------------------------------------------------------------------------
# OpenAI-compatible embeddings
# --------------------------------------------------------------------------


def test_embedder_restores_provider_order():
    def handler(request: httpx.Request) -> httpx.Response:
        # Providers may return embeddings out of order; "index" is authoritative.
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                ]
            },
        )

    embedder = OpenAICompatibleEmbedder(
        "https://x/v1", "k", "text-embedding-3-small", dim=3, client=client_for(handler)
    )
    assert embedder.embed(["first", "second"]) == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_embedder_posts_all_texts_in_one_request():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": [0.0]} for i in range(3)]}
        )

    embedder = OpenAICompatibleEmbedder(
        "https://x/v1", "key", "m", dim=1, client=client_for(handler)
    )
    embedder.embed(["a", "b", "c"])

    import json

    assert len(seen) == 1
    assert json.loads(seen[0].content)["input"] == ["a", "b", "c"]
    assert seen[0].headers["Authorization"] == "Bearer key"


# --------------------------------------------------------------------------
# GigaChat OAuth lifecycle
# --------------------------------------------------------------------------


def _gigachat_handler(calls: list[str], expires_at_ms: float):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth"):
            calls.append("auth")
            return httpx.Response(
                200, json={"access_token": "tok-1", "expires_at": expires_at_ms}
            )
        calls.append("chat")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ответ"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    return handler


def test_gigachat_fetches_a_token_then_reuses_it():
    calls: list[str] = []
    far_future_ms = (10**12) * 10  # comfortably beyond the 60s refresh margin
    llm = GigaChatLLM(
        "base64-auth", "GIGACHAT_API_PERS", CostTracker(),
        client=client_for(_gigachat_handler(calls, far_future_ms)),
    )

    assert llm.complete("s", "u").text == "ответ"
    assert llm.complete("s", "u").text == "ответ"

    # One OAuth round trip, two completions: the token was cached.
    assert calls == ["auth", "chat", "chat"]


def test_gigachat_refetches_an_expired_token():
    calls: list[str] = []
    llm = GigaChatLLM(
        "base64-auth", "GIGACHAT_API_PERS", CostTracker(),
        client=client_for(_gigachat_handler(calls, 0)),  # already expired
    )

    llm.complete("s", "u")
    llm.complete("s", "u")

    assert calls == ["auth", "chat", "auth", "chat"]


def test_gigachat_sends_auth_headers_and_tracks_tokens():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/oauth"):
            return httpx.Response(
                200, json={"access_token": "tok-1", "expires_at": (10**12) * 10}
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ответ"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    tracker = CostTracker()
    llm = GigaChatLLM(
        "base64-auth", "GIGACHAT_API_PERS", tracker, client=client_for(handler)
    )
    usage = llm.complete("s", "u").usage

    auth, chat = seen
    assert auth.headers["Authorization"] == "Basic base64-auth"
    assert auth.headers["RqUID"]  # required by the provider, one per request
    assert chat.headers["Authorization"] == "Bearer tok-1"

    # GigaChat is billed in roubles, so tokens are tracked and USD stays zero.
    assert (usage.prompt_tokens, usage.completion_tokens) == (7, 3)
    assert usage.cost_usd == 0.0
    assert tracker.total.prompt_tokens == 7
