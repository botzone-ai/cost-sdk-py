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


# --- regression tests for the ingestion payload contract (fixed in 0.1.1) ---

import re

from botzone_cost.queue import IngestionQueue


def test_occurred_at_is_z_suffixed(monkeypatch):
    """occurredAt must be RFC3339 with a 'Z' suffix; a '+00:00' offset 400s."""
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = wrap(_fake_anthropic(), route="t", provider="anthropic")
    client.messages.create(model="claude-sonnet-4-6", messages=[])
    occurred = body_calls[0]["occurredAt"]
    assert occurred.endswith("Z")
    assert "+" not in occurred
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", occurred)


def test_user_id_hash_omitted_when_absent(monkeypatch):
    """No user_id -> field absent, NOT null (the server rejects an explicit null)."""
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = wrap(_fake_anthropic(), route="t", provider="anthropic")
    client.messages.create(model="claude-sonnet-4-6", messages=[])
    assert "userIdHash" not in body_calls[0]


def test_user_id_hash_present_and_hashed_when_given(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = wrap(_fake_anthropic(), route="t", user_id="user-42", provider="anthropic")
    client.messages.create(model="claude-sonnet-4-6", messages=[])
    h = body_calls[0]["userIdHash"]
    assert isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)  # sha256 hex


def test_route_and_feature_tag_omitted_when_none(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = wrap(_fake_anthropic(), provider="anthropic")  # no route / feature_tag
    client.messages.create(model="claude-sonnet-4-6", messages=[])
    ev = body_calls[0]
    assert "route" not in ev and "featureTag" not in ev


class _FakeResp:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class _FakeHttp:
    def __init__(self, status_code):
        self.status_code = status_code
        self.calls = 0

    def post(self, *a, **k):
        self.calls += 1
        return _FakeResp(self.status_code, "boom")


def _queue(http):
    return IngestionQueue(api_key="cost_sk_test", endpoint="http://localhost:3001", http=http)


def test_send_records_rejection_on_4xx(monkeypatch):
    """A 4xx must be surfaced (failed_count), not silently swallowed, and not retried."""
    monkeypatch.setattr("botzone_cost.queue.time.sleep", lambda *_: None)
    http = _FakeHttp(400)
    q = _queue(http)
    q._send([{"x": 1}])
    assert q.failed_count() == 1
    assert http.calls == 1


def test_send_retries_then_fails_on_5xx(monkeypatch):
    monkeypatch.setattr("botzone_cost.queue.time.sleep", lambda *_: None)
    http = _FakeHttp(503)
    q = _queue(http)
    q._send([{"x": 1}])
    assert http.calls == 4  # initial + 3 retries
    assert q.failed_count() == 1


def test_send_success_no_failure():
    http = _FakeHttp(202)
    q = _queue(http)
    q._send([{"x": 1}])
    assert q.failed_count() == 0
    assert http.calls == 1
