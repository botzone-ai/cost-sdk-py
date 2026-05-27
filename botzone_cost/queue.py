"""Background ingestion queue.

Fire-and-forget: enqueue() returns immediately. A daemon thread flushes the
buffer every 2 seconds (and on process exit). Drops events past a 1000-item
cap rather than leak memory.
"""
from __future__ import annotations

import atexit
import json
import threading
import time
from typing import Optional

import httpx


class IngestionQueue:
    def __init__(self, api_key: str, endpoint: str, *, http: Optional[httpx.Client] = None):
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._dropped = 0
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
        return self._dropped

    def _send(self, batch: list[dict], attempt: int = 0) -> None:
        try:
            self._http.post(
                f"{self._endpoint}/api/v1/events",
                headers={"x-api-key": self._api_key, "content-type": "application/json"},
                content=json.dumps({"events": batch}),
            )
        except Exception:
            if attempt < 3:
                time.sleep(0.2 * (2 ** attempt))
                self._send(batch, attempt + 1)

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
