from rag_service.llm import PRICE_TABLE, CostTracker


def test_known_model_cost_is_computed():
    tracker = CostTracker()
    usage = tracker.add("deepseek-chat", 1_000_000, 1_000_000)
    p_in, p_out = PRICE_TABLE["deepseek-chat"]
    assert usage.cost_usd == p_in + p_out


def test_unknown_model_costs_zero_but_tracks_tokens():
    tracker = CostTracker()
    usage = tracker.add("mystery-model", 500, 100)
    assert usage.cost_usd == 0.0
    assert tracker.total.prompt_tokens == 500
    assert tracker.total.completion_tokens == 100


def test_totals_accumulate():
    tracker = CostTracker()
    tracker.add("deepseek-chat", 100, 50)
    tracker.add("deepseek-chat", 200, 25)
    assert tracker.requests == 2
    assert tracker.total.prompt_tokens == 300
    assert tracker.total.completion_tokens == 75
    assert tracker.total.cost_usd > 0
