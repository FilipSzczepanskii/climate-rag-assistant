"""Logging configuration and a minimal in-process metrics counter.

Kept deliberately small: structured stdout logging and a thread-safe counter
of queries, errors and latency. Enough to see how the service behaves in the
Space logs without pulling in a full metrics backend.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from threading import Lock


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


@dataclass
class Metrics:
    """Cumulative request metrics, safe for concurrent updates."""

    queries: int = 0
    errors: int = 0
    total_latency_ms: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_query(self, latency_ms: int) -> None:
        with self._lock:
            self.queries += 1
            self.total_latency_ms += latency_ms

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg = self.total_latency_ms // self.queries if self.queries else 0
            return {"queries": self.queries, "errors": self.errors, "avg_latency_ms": avg}


metrics = Metrics()
