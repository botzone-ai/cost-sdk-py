"""Background ingestion queue.

Fire-and-forget: enqueue() returns immediately. A daemon thread flushes the
buffer every 2 seconds (and on process exit). Drops events past a 1000-item
cap rather than leak memory.
"""
from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger("botzone_cost")


class IngestionQueue:
    def __init__(self, api_key: str, endpoint: str, *, http: Optional[httpx.Client] = None):
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._dropped = 0
        self._failed = 0
        self._http = http or httpx.Client(timeout=5.0)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        atexit.register(self._on_exit)

    def enqueue(self, event: dict) -> None:
        with self._lock:
            if len(self._buf) >= 1000:
                self._dropped += 1
                return
            self._buf.append(event)

    def flush(self) -> None:
        with self._lock:
            batch, self._buf = self._buf, []
        if not batch:
            return
        for chunk_start in range(0, len(batch), 50):
            chunk = batch[chunk_start : chunk_start + 50]
            self._send(chunk)

    def dropped_count(self) -> int:
        """Events discarded locally because the buffer was full."""
        return self._dropped

    def failed_count(self) -> int:
        """Events the server rejected, or that exhausted network retries."""
        return self._failed

    # Retry only when the condition is transient.
    _RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

    def _send(self, batch: list[dict], attempt: int = 0) -> None:
        try:
            resp = self._http.post(
                f"{self._endpoint}/api/v1/events",
                headers={"x-api-key": self._api_key, "content-type": "application/json"},
                content=json.dumps({"events": batch}),
            )
        except Exception as exc:  # transport error: DNS, connect, read timeout
            if attempt < 3:
                time.sleep(0.2 * (2 ** attempt))
                self._send(batch, attempt + 1)
            else:
                self._failed += len(batch)
                logger.warning(
                    "botzone-cost: dropping %d event(s) after network errors: %s",
                    len(batch), exc,
                )
            return

        if resp.status_code < 300:
            return  # accepted

        if resp.status_code in self._RETRY_STATUS and attempt < 3:
            time.sleep(0.2 * (2 ** attempt))
            self._send(batch, attempt + 1)
            return

        # Permanent rejection (bad key, invalid payload, ...). Surface it rather
        # than silently dropping - it is almost always a misconfiguration.
        self._failed += len(batch)
        body = (resp.text or "")[:300]
        logger.warning(
            "botzone-cost: server rejected %d event(s): HTTP %d %s",
            len(batch), resp.status_code, body,
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(2.0)
            try:
                self.flush()
            except Exception:
                pass

    def _on_exit(self) -> None:
        self._stop.set()
        try:
            self.flush()
        except Exception:
            pass
