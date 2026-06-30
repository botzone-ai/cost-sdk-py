"""wrap() entry point. Auto-detects Anthropic / OpenAI / Gemini clients."""
from __future__ import annotations

import hashlib
import json
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
    # RFC3339 with millisecond precision and a 'Z' suffix. The ingestion API's
    # datetime validator rejects a numeric offset such as '+00:00'.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


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

    By default the SDK is metadata-only: it ships token counts, model,
    route, latency, hashed user id, and feature tag - never prompt or
    response content.

    Set ``capture_bodies=True`` to also send the raw request and response
    bodies (``rawRequest`` / ``rawResponse``). This is what powers the
    eval-gated verify-downgrade feature, which replays captured requests
    through a cheaper model and judges the result. It is OPT-IN because the
    bodies contain prompt content; it mirrors the TypeScript SDK, whose
    ``captureBodies`` is likewise off by default. Capture is best-effort and
    fail-safe: if a body can't be serialised it is dropped, never raised.
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


def _json_safe(obj: Any) -> Optional[Any]:
    """Best-effort JSON-serialisable copy of ``obj``, or None if it can't be made
    safe. Never raises - body capture must not break the wrapped call."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass
    # Anthropic / OpenAI responses are pydantic v2 models.
    try:
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            d = dump(mode="json")
            json.dumps(d)
            return d
    except Exception:
        pass
    return None


def _normalize_gemini_contents(contents: Any) -> Optional[Any]:
    """Coerce a google-generativeai ``contents`` argument into the Content[] shape
    the runtime's replay() feeds to generateContent. None if it can't."""
    if isinstance(contents, str):
        return [{"role": "user", "parts": [{"text": contents}]}]
    if isinstance(contents, list):
        if contents and all(isinstance(c, str) for c in contents):
            return [{"role": "user", "parts": [{"text": c} for c in contents]}]
        safe = _json_safe(contents)
        if not isinstance(safe, list):
            return safe
        if safe and all(isinstance(c, dict) and "parts" in c for c in safe):
            return safe  # already Content[]
        if safe and all(isinstance(c, dict) and "text" in c for c in safe):
            return [{"role": "user", "parts": safe}]  # Part[] -> one user turn
        return safe
    safe = _json_safe(contents)
    if isinstance(safe, dict) and "contents" in safe:
        return safe["contents"]
    if isinstance(safe, dict) and "parts" in safe:
        return [safe]
    return safe


def _gemini_request_body(args, kwargs) -> Optional[dict]:
    contents = args[0] if args else kwargs.get("contents")
    if contents is None:
        return None
    norm = _normalize_gemini_contents(contents)
    if norm is None:
        return None
    return {"contents": norm}


def _gemini_text(result: Any) -> Optional[str]:
    # ``.text`` is a convenience property that raises when there is no candidate.
    try:
        t = getattr(result, "text", None)
        if isinstance(t, str) and t:
            return t
    except Exception:
        pass
    try:
        out = []
        for cand in (getattr(result, "candidates", None) or []):
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                txt = getattr(part, "text", None)
                if txt:
                    out.append(txt)
        if out:
            return "".join(out)
    except Exception:
        pass
    return None


def _gemini_response_body(result: Any) -> Optional[dict]:
    text = _gemini_text(result)
    return {"text": text} if text is not None else None


def _event(provider, model, prompt_tokens, completion_tokens, cached,
           cache_creation, latency_ms, route, user_hash, feature_tag,
           raw_request=None, raw_response=None):
    """Build one ingestion event.

    Optional fields that are None are OMITTED, not sent as null: the ingestion
    API expects a string-or-absent for ``userIdHash`` / ``route`` / ``featureTag``
    and rejects an explicit null (HTTP 400). ``rawRequest`` / ``rawResponse`` are
    likewise included only when body capture is on and serialisation succeeded.
    """
    event = {
        "provider": provider,
        "model": model,
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "cachedTokens": cached,
        "cacheCreationTokens": cache_creation,
        "latencyMs": latency_ms,
        "occurredAt": _now(),
    }
    if route is not None:
        event["route"] = route
    if user_hash is not None:
        event["userIdHash"] = user_hash
    if feature_tag is not None:
        event["featureTag"] = feature_tag
    if raw_request is not None:
        event["rawRequest"] = raw_request
    if raw_response is not None:
        event["rawResponse"] = raw_response
    return event


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
            raw_req = raw_res = None
            if capture_bodies:
                try:
                    raw_req = _json_safe(dict(kwargs))
                    raw_res = _json_safe(result)
                except Exception:
                    raw_req = raw_res = None
            _emit(queue, _event(
                "anthropic",
                kwargs.get("model") or getattr(result, "model", "unknown"),
                input_tokens, output_tokens, cached, cache_creation,
                latency_ms, route, user_hash, feature_tag,
                raw_req, raw_res,
            ))
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
            raw_req = raw_res = None
            if capture_bodies:
                try:
                    raw_req = _json_safe(dict(kwargs))
                    raw_res = _json_safe(result)
                except Exception:
                    raw_req = raw_res = None
            _emit(queue, _event(
                "openai",
                kwargs.get("model") or getattr(result, "model", "unknown"),
                prompt_tokens, completion_tokens, cached or 0, 0,
                latency_ms, route, user_hash, feature_tag,
                raw_req, raw_res,
            ))
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
            raw_req = raw_res = None
            if capture_bodies:
                try:
                    raw_req = _gemini_request_body(args, kwargs)
                    raw_res = _gemini_response_body(result)
                except Exception:
                    raw_req = raw_res = None
            _emit(queue, _event(
                "gemini",
                model_name,
                prompt or 0, completion or 0, cached or 0, 0,
                latency_ms, route, user_hash, feature_tag,
                raw_req, raw_res,
            ))
        return result

    model.generate_content = wrapped
    return model


def flush() -> None:
    """Block until all pending events are sent. Useful in scripts before exit."""
    for q in _queues.values():
        q.flush()
