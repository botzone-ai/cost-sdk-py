"""Unit tests for botzone_cost.wrap (no real LLM clients required)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from botzone_cost import wrap, flush


def _fake_anthropic(input_tokens=100, output_tokens=50, cached=80):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        model="claude-sonnet-4-6",
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cached,
            cache_creation_input_tokens=0,
        ),
    )
    return client


def _fake_openai(prompt_tokens=200, completion_tokens=75, cached=50):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )
    return client


def _fake_gemini(prompt=300, completion=100, cached=0):
    model = MagicMock()
    model.model_name = "gemini-2.5-flash"
    model.generate_content.return_value = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=completion,
            cached_content_token_count=cached,
        ),
    )
    return model


def _captured_events(monkeypatch, body_calls):
    """Returns a stub IngestionQueue.enqueue capturing events into body_calls."""
    def fake_enqueue(self, event):
        body_calls.append(event)
    return fake_enqueue


def _setup(monkeypatch, body_calls):
    monkeypatch.setenv("COST_API_KEY", "cost_sk_test")
    monkeypatch.setenv("COST_ENDPOINT", "http://localhost:3001")
    from botzone_cost import _wrap as wrap_mod
    wrap_mod._queues.clear()
    from botzone_cost.queue import IngestionQueue
    monkeypatch.setattr(IngestionQueue, "enqueue", _captured_events(monkeypatch, body_calls))


def test_anthropic(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = wrap(_fake_anthropic(), route="test", provider="anthropic")
    client.messages.create(model="claude-sonnet-4-6", messages=[])
    assert len(body_calls) == 1
    ev = body_calls[0]
    assert ev["provider"] == "anthropic"
    assert ev["promptTokens"] == 100
    assert ev["cachedTokens"] == 80
    assert ev["route"] == "test"


def test_openai(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = wrap(_fake_openai(), route="summarise", provider="openai")
    client.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert len(body_calls) == 1
    ev = body_calls[0]
    assert ev["provider"] == "openai"
    assert ev["promptTokens"] == 200
    assert ev["cachedTokens"] == 50


def test_gemini(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    model = wrap(_fake_gemini(), route="classify", provider="gemini")
    model.generate_content("hello")
    assert len(body_calls) == 1
    ev = body_calls[0]
    assert ev["provider"] == "gemini"
    assert ev["model"] == "gemini-2.5-flash"
    assert ev["promptTokens"] == 300


def test_disabled(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = wrap(_fake_anthropic(), enabled=False, provider="anthropic")
    client.messages.create(model="claude-sonnet-4-6", messages=[])
    assert body_calls == []


def test_passthrough_returns_result(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = wrap(_fake_anthropic(), provider="anthropic")
    result = client.messages.create(model="claude-sonnet-4-6", messages=[])
    assert result.model == "claude-sonnet-4-6"
