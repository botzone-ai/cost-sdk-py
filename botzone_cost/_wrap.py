"""wrap() entry point. Auto-detects Anthropic / OpenAI / Gemini clients."""
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from .queue import IngestionQueue

_queues: dict[str, IngestionQueue] = {}


def _get_queue(api_key: Optional[str], endpoint: Optional[str], enabled: bool) -> Optional[IngestionQueue]:
    if not enabled:
        return None
    api_key = api_key or os.environ.get("COST_API_KEY")
    if not api_key:
        return None
    endpoint = endpoint or os.environ.get("COST_ENDPOINT", "https://cost.botzone.ai")
    key = f"{api_key}|{endpoint}"
    q = _queues.get(key)
    if not q:
        q = IngestionQueue(api_key=api_key, endpoint=endpoint)
        _queues[key] = q
    return q


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wrap(
    client: Any,
    *,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    route: Optional[str] = None,
    user_id: Optional[str] = None,
    feature_tag: Optional[str] = None,
    enabled: bool = True,
    capture_bodies: bool = False,
    provider: Optional[str] = None,
) -> Any:
    """Wrap an LLM client to capture cost-tracking events.

    Returns the same client (mutated) for chaining. Raises ValueError if
    the provider can't be detected.

    The Python SDK is metadata-only today: it ships token counts, model,
    route, latency, hashed user id, and feature tag. It does not send raw
    request or response bodies. The ``capture_bodies`` keyword is reserved
    for future parity with the TypeScript SDK and has no effect today.
    """
    queue = _get_queue(api_key, endpoint, enabled)
    user_hash = _sha256(user_id) if user_id else None

    detected = provider or _detect(client)
    if detected == "anthropic":
        return _wrap_anthropic(client, queue, route, user_hash, feature_tag, capture_bodies)
    if detected == "openai":
        return _wrap_openai(client, queue, route, user_hash, feature_tag, capture_bodies)
    if detected == "gemini":
        return _wrap_gemini(client, queue, route, user_hash, feature_tag, capture_bodies)
    raise ValueError(
        "[botzone-cost] could not detect provider: pass provider='anthropic'|'openai'|'gemini'"
    )


def _detect(client: Any) -> Optional[str]:
    # Prefer module-path detection so auto-mocks don't all look like Anthropic.
    module = getattr(type(client), "__module__", "") or ""
    if module.startswith("anthropic"):
        return "anthropic"
    if module.startswith("openai"):
        return "openai"
    if module.startswith("google.generativeai") or module.startswith("google.genai"):
        return "gemini"
    # Fallback duck-typing for thin wrappers that don't carry the module name.
    if hasattr(client, "messages") and hasattr(client.messages, "create"):
        return "anthropic"
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return "openai"
    if hasattr(client, "generate_content"):
        return "gemini"
    return None


def _emit(queue: Optional[IngestionQueue], event: dict) -> None:
    if queue is not None:
        queue.enqueue(event)


def _wrap_anthropic(client: Any, queue, route, user_hash, feature_tag, capture_bodies):
    original = client.messages.create

    def wrapped(*args, **kwargs):
        start = time.time()
        result = original(*args, **kwargs)
        latency_ms = int((time.time() - start) * 1000)
        if queue is not None:
            usage = getattr(result, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            cached = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            _emit(queue, {
                "provider": "anthropic",
                "model": kwargs.get("model") or getattr(result, "model", "unknown"),
                "promptTokens": input_tokens,
                "completionTokens": output_tokens,
                "cachedTokens": cached,
                "cacheCreationTokens": cache_creation,
                "latencyMs": latency_ms,
                "route": route,
                "userIdHash": user_hash,
                "featureTag": feature_tag,
                "occurredAt": _now(),
            })
        return result

    client.messages.create = wrapped
    return client


def _wrap_openai(client: Any, queue, route, user_hash, feature_tag, capture_bodies):
    original = client.chat.completions.create

    def wrapped(*args, **kwargs):
        start = time.time()
        result = original(*args, **kwargs)
        latency_ms = int((time.time() - start) * 1000)
        if queue is not None:
            usage = getattr(result, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            _emit(queue, {
                "provider": "openai",
                "model": kwargs.get("model") or getattr(result, "model", "unknown"),
                "promptTokens": prompt_tokens,
                "completionTokens": completion_tokens,
                "cachedTokens": cached or 0,
                "cacheCreationTokens": 0,
                "latencyMs": latency_ms,
                "route": route,
                "userIdHash": user_hash,
                "featureTag": feature_tag,
                "occurredAt": _now(),
            })
        return result

    client.chat.completions.create = wrapped
    return client


def _wrap_gemini(model: Any, queue, route, user_hash, feature_tag, capture_bodies):
    """Wraps a `google.generativeai.GenerativeModel` instance directly."""
    original = model.generate_content
    model_name = getattr(model, "model_name", None) or getattr(model, "_model_name", "unknown")
    if isinstance(model_name, str) and model_name.startswith("models/"):
        model_name = model_name[len("models/"):]

    def wrapped(*args, **kwargs):
        start = time.time()
        result = original(*args, **kwargs)
        latency_ms = int((time.time() - start) * 1000)
        if queue is not None:
            usage = getattr(result, "usage_metadata", None)
            prompt = getattr(usage, "prompt_token_count", 0) if usage else 0
            completion = getattr(usage, "candidates_token_count", 0) if usage else 0
            cached = getattr(usage, "cached_content_token_count", 0) if usage else 0
            _emit(queue, {
                "provider": "gemini",
                "model": model_name,
                "promptTokens": prompt or 0,
                "completionTokens": completion or 0,
                "cachedTokens": cached or 0,
                "cacheCreationTokens": 0,
                "latencyMs": latency_ms,
                "route": route,
                "userIdHash": user_hash,
                "featureTag": feature_tag,
                "occurredAt": _now(),
            })
        return result

    model.generate_content = wrapped
    return model


def flush() -> None:
    """Block until all pending events are sent. Useful in scripts before exit."""
    for q in _queues.values():
        q.flush()
