"""Asynchronous request scheduler in front of a batch executor.

The scheduler decides WHICH requests run next and WHEN; Person 3's batching code
(``OnnxTextGenerator.generate_batch`` + ``BatchConfig``) decides how the chosen
requests are grouped into padded buckets and Person 2's generator runs them.

    scheduler = RequestScheduler(
        lambda reqs: generator.generate_batch(reqs, batch_config),
        token_counter=lambda r: generator.count_prompt_tokens(r.prompt),
        config=SchedulerConfig(max_queue_length=32, default_timeout_s=60),
        batch_metrics_source=lambda: generator.last_batch_metrics,
    )
    async with scheduler:
        result = await scheduler.submit(GenerationRequest("Hello", priority=1))

Policies
--------
* FIFO by default. A higher ``priority`` is served first; ties are FIFO.
* Aging: each ``aging_interval_s`` of waiting adds one priority level, so a
  low-priority request cannot be starved forever by a stream of urgent ones.
* Overload: a full queue, an over-long prompt, or an over-long generation
  request raises immediately (``QueueFullError`` etc.) instead of hanging.
* Timeout: ``request.deadline_s`` (seconds from submission) or
  ``config.default_timeout_s`` bounds queue wait + execution. A request that
  expires in the queue never runs.
* Cancellation: ``cancel(request_id)``. A queued request is removed. A request
  already inside a batch cannot be interrupted (the executor call is blocking),
  so the caller is released immediately and the late result is discarded.

Limitation: the executor is opaque, so DECODING is only recorded if the
executor reports progress through ``report_progress``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import itertools
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

from inferbench.scheduler.metrics import ServingMetrics
from inferbench.scheduler.types import (
    DeadlineExpiredError,
    DuplicateRequestError,
    GenerationTooLongError,
    PromptTooLongError,
    QueueFullError,
    RequestCancelledError,
    RequestRejectedError,
    RequestState,
    RequestTimeoutError,
    SchedulerClosedError,
    SchedulerConfig,
)

if TYPE_CHECKING:
    from models.generation import GenerationRequest, GenerationResult

BatchExecutor = Callable[
    [list["GenerationRequest"]],
    "Sequence[GenerationResult] | Awaitable[Sequence[GenerationResult]]",
]
TokenCounter = Callable[["GenerationRequest"], int]


class RequestHandle:
    """Tracks one submitted request through its lifecycle."""

    def __init__(
        self,
        request: GenerationRequest,
        prompt_tokens: int,
        seq: int,
        enqueued_at: float,
        future: asyncio.Future[Any],
        timeout_s: float | None,
    ) -> None:
        self.request = request
        self.request_id: str = request.request_id  # type: ignore[assignment]
        self.prompt_tokens = prompt_tokens
        self.seq = seq
        self.enqueued_at = enqueued_at
        self.future = future
        self.timeout_s = timeout_s
        self.state = RequestState.QUEUED
        self.history: list[tuple[RequestState, float]] = [(RequestState.QUEUED, enqueued_at)]
        self._timer: asyncio.TimerHandle | None = None

    async def result(self) -> GenerationResult:
        return await asyncio.shield(self.future)


class RequestScheduler:
    def __init__(
        self,
        batch_executor: BatchExecutor,
        token_counter: TokenCounter | None = None,
        config: SchedulerConfig | None = None,
        metrics: ServingMetrics | None = None,
        batch_metrics_source: Callable[[], Sequence[Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or SchedulerConfig()
        self.metrics = metrics or ServingMetrics()
        self._executor = batch_executor
        # Fallback token counter: whitespace words. Pass a real tokenizer-based
        # counter (generator.count_prompt_tokens) for accurate limits.
        self._token_counter = token_counter or (lambda request: len(request.prompt.split()))
        self._batch_metrics_source = batch_metrics_source
        self._clock = clock
        self._queue: list[RequestHandle] = []
        self._handles: dict[str, RequestHandle] = {}
        self._ids = itertools.count(1)
        self._seq = itertools.count()
        self._wakeup: asyncio.Event | None = None
        self._full: asyncio.Event | None = None
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    @property
    def in_flight(self) -> int:
        return sum(
            1
            for handle in self._handles.values()
            if handle.state in (RequestState.BATCHED, RequestState.PREFILLING, RequestState.DECODING)
        )

    def state(self, request_id: str) -> RequestState | None:
        handle = self._handles.get(request_id)
        return handle.state if handle else None

    def handle(self, request_id: str) -> RequestHandle | None:
        return self._handles.get(request_id)

    def enqueue(self, request: GenerationRequest) -> RequestHandle:
        """Validate and queue a request. Must be called from the running event loop.

        Raises a ``RequestRejectedError`` subclass instead of queueing when the
        request breaks a limit.
        """
        try:
            return self._enqueue(request)
        except RequestRejectedError as exc:
            self.metrics.on_reject(exc.reason)
            raise

    async def submit(self, request: GenerationRequest) -> GenerationResult:
        """Queue a request and wait for its own result."""
        handle = self.enqueue(request)
        try:
            return await asyncio.shield(handle.future)
        except asyncio.CancelledError:
            # The caller gave up (task cancelled): stop working on its behalf.
            self.cancel(handle.request_id)
            raise

    def cancel(self, request_id: str) -> bool:
        """Cancel a queued or in-flight request. Returns False if it is already finished."""
        handle = self._handles.get(request_id)
        if handle is None or handle.state.terminal:
            return False
        self._finish(
            handle,
            RequestState.CANCELLED,
            exception=RequestCancelledError(f"request {request_id!r} was cancelled"),
        )
        return True

    def report_progress(self, request_ids: Sequence[str], state: RequestState) -> None:
        """Let an executor report PREFILLING / DECODING for requests it is running."""
        if state not in (RequestState.PREFILLING, RequestState.DECODING):
            raise ValueError("progress state must be PREFILLING or DECODING")
        for request_id in request_ids:
            handle = self._handles.get(request_id)
            if handle is not None and handle.state in (
                RequestState.BATCHED,
                RequestState.PREFILLING,
                RequestState.DECODING,
            ):
                self._set_state(handle, state)

    def snapshot(self) -> dict[str, Any]:
        data = self.metrics.snapshot()
        data["queue_length"] = self.queue_length
        data["states"] = dict(Counter(h.state.value for h in self._handles.values()))
        return data

    async def close(self, drain: bool = True) -> None:
        """Stop accepting requests. ``drain=True`` finishes queued work first;
        ``drain=False`` cancels everything still waiting in the queue."""
        self._closed = True
        if not drain:
            for handle in list(self._queue):
                self.cancel(handle.request_id)
        if self._worker is not None:
            self._ensure_events()
            self._wakeup.set()  # type: ignore[union-attr]
            await self._worker

    async def __aenter__(self) -> RequestScheduler:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close(drain=True)

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def _ensure_events(self) -> None:
        if self._wakeup is None:
            self._wakeup = asyncio.Event()
            self._full = asyncio.Event()

    def _enqueue(self, request: GenerationRequest) -> RequestHandle:
        if self._closed:
            raise SchedulerClosedError("scheduler is closed")
        loop = asyncio.get_running_loop()
        self._ensure_events()
        cfg = self.config

        if request.request_id is None:
            request = dataclasses.replace(request, request_id=f"req-{next(self._ids)}")
        request_id = request.request_id
        existing = self._handles.get(request_id)  # type: ignore[arg-type]
        if existing is not None and not existing.state.terminal:
            raise DuplicateRequestError(f"request id {request_id!r} is already active")

        try:
            prompt_tokens = int(self._token_counter(request))
        except Exception as exc:
            raise RequestRejectedError(f"could not measure prompt length: {exc}") from exc
        if cfg.max_prompt_tokens is not None and prompt_tokens > cfg.max_prompt_tokens:
            raise PromptTooLongError(
                f"prompt has {prompt_tokens} tokens, limit is {cfg.max_prompt_tokens}"
            )
        if prompt_tokens > cfg.max_batch_tokens:
            raise PromptTooLongError(
                f"prompt has {prompt_tokens} tokens, exceeding max_batch_tokens "
                f"({cfg.max_batch_tokens})"
            )
        requested = request.config.max_new_tokens
        if cfg.max_new_tokens is not None and requested > cfg.max_new_tokens:
            raise GenerationTooLongError(
                f"request asks for {requested} new tokens, limit is {cfg.max_new_tokens}"
            )
        timeout = request.deadline_s if request.deadline_s is not None else cfg.default_timeout_s
        if timeout is not None and timeout <= 0:
            raise DeadlineExpiredError("request deadline is already expired")
        if len(self._queue) >= cfg.max_queue_length:
            raise QueueFullError(f"queue is full ({cfg.max_queue_length} waiting requests)")

        future: asyncio.Future[Any] = loop.create_future()
        # Mark exceptions as retrieved so abandoned requests do not log warnings.
        future.add_done_callback(lambda f: f.cancelled() or f.exception())
        handle = RequestHandle(
            request, prompt_tokens, next(self._seq), self._clock(), future, timeout
        )
        self._queue.append(handle)
        self._handles[request_id] = handle  # type: ignore[index]
        self.metrics.on_enqueue(request_id)  # type: ignore[arg-type]
        if timeout is not None:
            handle._timer = loop.call_later(timeout, self._expire, handle)
        if len(self._queue) >= cfg.max_batch_size:
            self._full.set()  # type: ignore[union-attr]
        if self._worker is None:
            self._worker = loop.create_task(self._run())
        self._wakeup.set()  # type: ignore[union-attr]
        return handle

    # ------------------------------------------------------------------
    # Selection policy
    # ------------------------------------------------------------------

    def _effective_priority(self, handle: RequestHandle, now: float) -> float:
        aging = self.config.aging_interval_s
        waited = max(0.0, now - handle.enqueued_at)
        bonus = waited / aging if aging else 0.0
        return handle.request.priority + bonus

    def _pick_batch(self, now: float | None = None) -> list[RequestHandle]:
        """Choose the next batch: best effective priority first, FIFO on ties,
        stopping at the batch size / token limits (never skipping ahead of a
        better-ranked request that does not fit)."""
        now = self._clock() if now is None else now
        ranked = sorted(
            self._queue,
            key=lambda h: (-self._effective_priority(h, now), h.seq),
        )
        chosen: list[RequestHandle] = []
        tokens = 0
        for handle in ranked:
            if len(chosen) >= self.config.max_batch_size:
                break
            if chosen and tokens + handle.prompt_tokens > self.config.max_batch_tokens:
                break
            chosen.append(handle)
            tokens += handle.prompt_tokens
        for handle in chosen:
            self._queue.remove(handle)
        return chosen

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        wakeup, full = self._wakeup, self._full
        assert wakeup is not None and full is not None
        try:
            while True:
                await wakeup.wait()
                if not self._queue:
                    if self._closed:
                        return
                    wakeup.clear()
                    continue
                window = self.config.batching_window_ms / 1000.0
                if (
                    self.config.max_batch_size > 1
                    and window > 0
                    and len(self._queue) < self.config.max_batch_size
                    and not self._closed
                ):
                    try:
                        await asyncio.wait_for(full.wait(), window)
                    except TimeoutError:
                        pass
                full.clear()
                batch = self._pick_batch()
                if batch:
                    await self._execute(batch)
                if not self._queue and not self._closed:
                    wakeup.clear()
        except Exception as exc:  # never leave callers hanging on a scheduler bug
            for handle in list(self._queue):
                self._finish(handle, RequestState.FAILED, exception=exc)
            raise

    async def _execute(self, batch: list[RequestHandle]) -> None:
        for handle in batch:
            self._set_state(handle, RequestState.BATCHED)
        for handle in batch:
            self._set_state(handle, RequestState.PREFILLING)
            self.metrics.on_dispatch(handle.request_id)
        requests = [handle.request for handle in batch]
        try:
            if inspect.iscoroutinefunction(self._executor):
                results = await self._executor(requests)
            else:
                results = await asyncio.to_thread(self._executor, requests)
                if inspect.isawaitable(results):
                    results = await results
            if len(results) != len(batch):
                raise RuntimeError(
                    f"batch executor returned {len(results)} results for {len(batch)} requests"
                )
        except Exception as exc:
            for handle in batch:
                self._finish(handle, RequestState.FAILED, exception=exc)
            return
        self._record_batch_metrics(len(batch))
        for handle, result in zip(batch, results):
            rid = getattr(result, "request_id", None)
            if rid is not None and rid != handle.request_id:
                self._finish(
                    handle,
                    RequestState.FAILED,
                    exception=RuntimeError(
                        f"executor returned the result of request {rid!r} "
                        f"for request {handle.request_id!r}"
                    ),
                )
            else:
                self._finish(handle, RequestState.COMPLETED, result=result)

    def _record_batch_metrics(self, batch_size: int) -> None:
        recorded = False
        if self._batch_metrics_source is not None:
            try:
                for item in self._batch_metrics_source() or ():
                    self.metrics.on_batch(
                        item.effective_batch_size,
                        getattr(item, "padding_overhead_ratio", 0.0),
                        getattr(item, "batch_formation_time_ms", 0.0),
                    )
                    recorded = True
            except Exception:
                recorded = False
        if not recorded:
            self.metrics.on_batch(batch_size)

    # ------------------------------------------------------------------
    # State changes
    # ------------------------------------------------------------------

    def _set_state(self, handle: RequestHandle, state: RequestState) -> None:
        if handle.state.terminal or handle.state is state:
            return
        handle.state = state
        handle.history.append((state, self._clock()))

    def _expire(self, handle: RequestHandle) -> None:
        if handle.state.terminal:
            return
        self._finish(
            handle,
            RequestState.TIMED_OUT,
            exception=RequestTimeoutError(
                f"request {handle.request_id!r} exceeded its {handle.timeout_s}s limit"
            ),
        )

    def _finish(
        self,
        handle: RequestHandle,
        state: RequestState,
        result: Any = None,
        exception: BaseException | None = None,
    ) -> None:
        """Move a request to a terminal state exactly once. Late results for a
        request that already timed out or was cancelled are dropped here."""
        if handle.state.terminal:
            return
        if handle in self._queue:
            self._queue.remove(handle)
        if handle._timer is not None:
            handle._timer.cancel()
            handle._timer = None
        self._set_state(handle, state)
        self.metrics.on_finish(handle.request_id, state, result)
        if not handle.future.done():
            if exception is not None:
                handle.future.set_exception(exception)
            else:
                handle.future.set_result(result)