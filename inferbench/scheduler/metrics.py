"""Serving metrics: queue wait, service time, TTFT, latency, throughput, counters.

``ServingMetrics`` is deliberately independent of the scheduler. The scheduler
feeds it automatically, and the benchmark report feeds it by hand for the
sequential and batched-only configurations, so every configuration is measured
with exactly the same definitions:

* queue wait      time from enqueue until the request is handed to the executor
* service time    time from hand-off until the request finishes
* end-to-end      queue wait + service time (completed requests only)
* TTFT            engine-reported time to first token (prefill), service side
* TTFT incl. wait queue wait + engine TTFT (what the caller actually experiences)
* inter-token     engine-reported mean decode time per generated token
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from inferbench.scheduler.types import RequestState


def summarize(values: list[float]) -> dict[str, Any] | None:
    """mean / p50 / p95 / max of a list, or None when there is nothing to summarise."""
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


@dataclass
class _Record:
    enqueued: float
    dispatched: float | None = None
    finished: float | None = None
    state: RequestState | None = None


class ServingMetrics:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._records: dict[str, _Record] = {}
        self._rejected: Counter[str] = Counter()
        self._outcomes: Counter[str] = Counter()
        self._queue_wait_ms: list[float] = []
        self._service_ms: list[float] = []
        self._e2e_ms: list[float] = []
        self._ttft_ms: list[float] = []
        self._ttft_incl_queue_ms: list[float] = []
        self._inter_token_ms: list[float] = []
        self._generated_tokens = 0
        self._prompt_tokens = 0
        self._queued = 0
        self._in_flight = 0
        self._peak_queued = 0
        self._peak_in_flight = 0
        self._first_enqueue: float | None = None
        self._last_finish: float | None = None
        self._batch_sizes: list[int] = []
        self._padding_ratios: list[float] = []
        self._formation_ms: list[float] = []

    # -- events ---------------------------------------------------------

    def on_enqueue(self, request_id: str) -> None:
        now = self._clock()
        with self._lock:
            self._records[request_id] = _Record(enqueued=now)
            if self._first_enqueue is None:
                self._first_enqueue = now
            self._queued += 1
            self._peak_queued = max(self._peak_queued, self._queued)

    def on_dispatch(self, request_id: str) -> None:
        now = self._clock()
        with self._lock:
            record = self._records.get(request_id)
            if record is None or record.dispatched is not None or record.finished is not None:
                return
            record.dispatched = now
            self._queued -= 1
            self._in_flight += 1
            self._peak_in_flight = max(self._peak_in_flight, self._in_flight)

    def on_finish(self, request_id: str, state: RequestState, result: Any = None) -> None:
        """Record the end of a request. Later calls for the same request are ignored."""
        now = self._clock()
        with self._lock:
            record = self._records.get(request_id)
            if record is None or record.finished is not None:
                return
            record.finished = now
            record.state = state
            if record.dispatched is None:
                self._queued -= 1
            else:
                self._in_flight -= 1
            self._last_finish = now
            self._outcomes[state.value] += 1

            started = record.dispatched if record.dispatched is not None else now
            queue_wait_ms = (started - record.enqueued) * 1000.0
            self._queue_wait_ms.append(queue_wait_ms)
            if record.dispatched is not None:
                self._service_ms.append((now - record.dispatched) * 1000.0)
            if state is RequestState.COMPLETED:
                self._e2e_ms.append((now - record.enqueued) * 1000.0)
                if result is not None:
                    ttft = float(getattr(result, "time_to_first_token_ms", 0.0))
                    self._ttft_ms.append(ttft)
                    self._ttft_incl_queue_ms.append(queue_wait_ms + ttft)
                    self._inter_token_ms.append(
                        float(getattr(result, "inter_token_latency_ms", 0.0))
                    )
                    self._generated_tokens += int(getattr(result, "generated_token_count", 0))
                    self._prompt_tokens += int(getattr(result, "prompt_token_count", 0))

    def on_reject(self, reason: str) -> None:
        with self._lock:
            self._rejected[reason] += 1

    def on_batch(
        self,
        effective_batch_size: int,
        padding_overhead_ratio: float = 0.0,
        batch_formation_time_ms: float = 0.0,
    ) -> None:
        with self._lock:
            self._batch_sizes.append(int(effective_batch_size))
            self._padding_ratios.append(float(padding_overhead_ratio))
            self._formation_ms.append(float(batch_formation_time_ms))

    # -- reporting ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = 0.0
            if self._first_enqueue is not None and self._last_finish is not None:
                elapsed = max(0.0, self._last_finish - self._first_enqueue)
            completed = self._outcomes[RequestState.COMPLETED.value]
            batches = len(self._batch_sizes)
            return {
                "counts": {
                    "submitted": len(self._records),
                    "completed": completed,
                    "failed": self._outcomes[RequestState.FAILED.value],
                    "cancelled": self._outcomes[RequestState.CANCELLED.value],
                    "timed_out": self._outcomes[RequestState.TIMED_OUT.value],
                    "rejected": sum(self._rejected.values()),
                    "rejected_by_reason": dict(self._rejected),
                },
                "active": {
                    "queued": self._queued,
                    "in_flight": self._in_flight,
                    "peak_queued": self._peak_queued,
                    "peak_in_flight": self._peak_in_flight,
                },
                "queue_wait_ms": summarize(self._queue_wait_ms),
                "service_time_ms": summarize(self._service_ms),
                "end_to_end_ms": summarize(self._e2e_ms),
                "ttft_ms": summarize(self._ttft_ms),
                "ttft_incl_queue_ms": summarize(self._ttft_incl_queue_ms),
                "inter_token_latency_ms": summarize(self._inter_token_ms),
                "elapsed_s": elapsed,
                "generated_tokens": self._generated_tokens,
                "prompt_tokens": self._prompt_tokens,
                "requests_per_second": completed / elapsed if elapsed > 0 else 0.0,
                "tokens_per_second": self._generated_tokens / elapsed if elapsed > 0 else 0.0,
                "batching": {
                    "batches": batches,
                    "mean_effective_batch_size": (
                        float(np.mean(self._batch_sizes)) if batches else None
                    ),
                    "max_effective_batch_size": max(self._batch_sizes) if batches else None,
                    "mean_padding_overhead_ratio": (
                        float(np.mean(self._padding_ratios)) if batches else None
                    ),
                    "mean_batch_formation_time_ms": (
                        float(np.mean(self._formation_ms)) if batches else None
                    ),
                },
            }