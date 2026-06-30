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


# --- body capture (0.1.2): rawRequest / rawResponse for verify-downgrade ---
#
# These shapes are read verbatim by the runtime worker:
#   replay()              -> rawRequest      (Anthropic/OpenAI kwargs; Gemini {contents})
#   extractBaseline()     -> rawResponse     (Anthropic .content[]; OpenAI .choices[];
#                                             Gemini {text})
#   extractPromptForJudge -> rawRequest.messages / rawRequest.contents
# Capture is OPT-IN (capture_bodies=True) and must NEVER break the wrapped call.


class _Pydanticish:
    """Mimics a pydantic v2 response: attribute access plus model_dump(mode=...)."""

    def __init__(self, attrs, dump):
        self._dump = dump
        for k, v in attrs.items():
            setattr(self, k, v)

    def model_dump(self, mode="python"):
        return self._dump


def _fake_anthropic_pydantic():
    client = MagicMock()
    client.messages.create.return_value = _Pydanticish(
        attrs={
            "model": "claude-sonnet-4-6",
            "usage": SimpleNamespace(
                input_tokens=100, output_tokens=50,
                cache_read_input_tokens=0, cache_creation_input_tokens=0,
            ),
        },
        dump={
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "Hi there"}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    )
    return client


def _fake_openai_pydantic():
    client = MagicMock()
    client.chat.completions.create.return_value = _Pydanticish(
        attrs={
            "model": "gpt-4o-mini",
            "usage": SimpleNamespace(
                prompt_tokens=200, completion_tokens=75,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        },
        dump={
            "model": "gpt-4o-mini",
            "choices": [{"message": {"role": "assistant", "content": "42"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 75},
        },
    )
    return client


def _fake_gemini_capture():
    model = MagicMock()
    model.model_name = "gemini-2.5-flash"
    model.generate_content.return_value = SimpleNamespace(
        text="The capital is Paris.",
        usage_metadata=SimpleNamespace(
            prompt_token_count=10, candidates_token_count=5, cached_content_token_count=0,
        ),
    )
    return model


def test_capture_bodies_default_off(monkeypatch):
    """Default is opt-OUT: no bodies leave the process unless explicitly enabled."""
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = wrap(_fake_anthropic(), route="t", provider="anthropic")
    client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}])
    ev = body_calls[0]
    assert "rawRequest" not in ev
    assert "rawResponse" not in ev


def test_capture_bodies_anthropic_shapes(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = wrap(_fake_anthropic_pydantic(), route="t", provider="anthropic", capture_bodies=True)
    client.messages.create(
        model="claude-sonnet-4-6", max_tokens=256,
        messages=[{"role": "user", "content": "hi"}],
    )
    ev = body_calls[0]
    # replay() reads rawRequest verbatim and needs messages + max_tokens
    assert ev["rawRequest"]["messages"] == [{"role": "user", "content": "hi"}]
    assert ev["rawRequest"]["max_tokens"] == 256
    # extractBaseline() reads rawResponse.content[].text
    assert ev["rawResponse"]["content"][0]["text"] == "Hi there"


def test_capture_bodies_openai_shapes(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = wrap(_fake_openai_pydantic(), route="t", provider="openai", capture_bodies=True)
    client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "2+2?"}],
    )
    ev = body_calls[0]
    assert ev["rawRequest"]["messages"] == [{"role": "user", "content": "2+2?"}]
    # extractBaseline() reads rawResponse.choices[0].message.content
    assert ev["rawResponse"]["choices"][0]["message"]["content"] == "42"


def test_capture_bodies_gemini_normalises_string(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    model = wrap(_fake_gemini_capture(), route="t", provider="gemini", capture_bodies=True)
    model.generate_content("What is the capital of France?")
    ev = body_calls[0]
    # replay() feeds rawRequest to generateContent: needs Content[] under .contents
    assert ev["rawRequest"] == {
        "contents": [{"role": "user", "parts": [{"text": "What is the capital of France?"}]}],
    }
    # extractBaseline() Gemini path reads the decoded {text}
    assert ev["rawResponse"] == {"text": "The capital is Paris."}


def test_capture_bodies_gemini_passes_through_content_list(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    model = wrap(_fake_gemini_capture(), route="t", provider="gemini", capture_bodies=True)
    contents = [{"role": "user", "parts": [{"text": "hello"}]}]
    model.generate_content(contents)
    assert body_calls[0]["rawRequest"] == {"contents": contents}


class _BoomResponse:
    """A response whose serialisation blows up - must not break the call."""

    model = "claude-sonnet-4-6"
    usage = SimpleNamespace(
        input_tokens=1, output_tokens=1,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )

    def model_dump(self, mode="python"):
        raise RuntimeError("cannot serialise this object")


def test_capture_failure_never_breaks_call(monkeypatch):
    body_calls: list = []
    _setup(monkeypatch, body_calls)
    client = MagicMock()
    client.messages.create.return_value = _BoomResponse()
    wrapped = wrap(client, provider="anthropic", capture_bodies=True)
    result = wrapped.messages.create(model="claude-sonnet-4-6", messages=[])
    assert isinstance(result, _BoomResponse)   # the real call still returns
    assert len(body_calls) == 1                # the metadata event still emits
    assert "rawResponse" not in body_calls[0]  # the unserialisable body is dropped, not raised
