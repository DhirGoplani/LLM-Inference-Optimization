"""Batch formation and asynchronous dynamic request collection."""

from __future__ import annotations

import asyncio
import inspect
import math
import statistics
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from models.generation import GenerationRequest, GenerationResult


class BatchGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def generate_batch(
        self,
        requests: list[GenerationRequest],
        batch_config: BatchConfig | None = None,
    ) -> list[GenerationResult]: ...

    def count_prompt_tokens(self, prompt: str) -> int: ...


@dataclass(frozen=True)
class BatchConfig:
    max_batch_size: int = 8
    max_total_batch_tokens: int = 8192
    batching_window_ms: float = 5.0
    max_padding_ratio: float = 0.5
    prompt_length_buckets: tuple[int, ...] | None = (128, 512, 2048, 4096)

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_total_batch_tokens <= 0:
            raise ValueError("max_total_batch_tokens must be positive")
        if self.batching_window_ms < 0:
            raise ValueError("batching_window_ms must be non-negative")
        if self.max_padding_ratio < 0:
            raise ValueError("max_padding_ratio must be non-negative")
        if self.prompt_length_buckets is not None:
            if not self.prompt_length_buckets:
                raise ValueError("prompt_length_buckets must not be empty")
            if any(boundary <= 0 for boundary in self.prompt_length_buckets):
                raise ValueError("prompt length bucket boundaries must be positive")
            if (
                tuple(sorted(set(self.prompt_length_buckets)))
                != self.prompt_length_buckets
            ):
                raise ValueError(
                    "prompt length bucket boundaries must be strictly increasing"
                )

    def bucket_index(self, prompt_length: int) -> int:
        if prompt_length <= 0:
            raise ValueError("prompt length must be positive")
        if self.prompt_length_buckets is None:
            if prompt_length > self.max_total_batch_tokens:
                raise ValueError(
                    f"Prompt has {prompt_length} tokens, exceeding "
                    f"max_total_batch_tokens ({self.max_total_batch_tokens})"
                )
            return 0
        for index, boundary in enumerate(self.prompt_length_buckets):
            if prompt_length <= boundary:
                return index
        raise ValueError(
            f"Prompt has {prompt_length} tokens, exceeding the largest configured "
            f"prompt bucket ({self.prompt_length_buckets[-1]})"
        )


@dataclass(frozen=True)
class BatchMetrics:
    effective_batch_size: int
    prompt_tokens: int
    padded_prompt_tokens: int
    padding_overhead_ratio: float
    batch_formation_time_ms: float
    execution_time_ms: float
    generated_tokens: int
    aggregate_tokens_per_second: float


@dataclass(frozen=True)
class BatchingBenchmarkResult:
    mode: str
    concurrency: int
    request_count: int
    elapsed_ms: float
    requests_per_second: float
    generated_tokens_per_second: float
    mean_latency_ms: float
    p95_latency_ms: float


@dataclass(frozen=True)
class BatchRequest:
    """A formed, same-bucket group and its already-tokenized prompts."""

    requests: tuple["GenerationRequest", ...]
    prompt_token_ids: tuple[tuple[int, ...], ...]
    bucket_index: int
    request_indices: tuple[int, ...]
    batch_formation_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("a batch request must contain at least one request")
        if len(self.requests) != len(self.prompt_token_ids):
            raise ValueError("requests and prompt_token_ids must have matching lengths")
        if len(self.requests) != len(self.request_indices):
            raise ValueError("requests and request_indices must have matching lengths")


def form_batch_indices(
    prompt_lengths: Sequence[int],
    config: BatchConfig,
    *,
    max_batch_size: int | None = None,
) -> list[tuple[int, ...]]:
    """Group prompt indices while respecting bucket, padding, and token limits."""
    limit = min(
        config.max_batch_size,
        config.max_batch_size if max_batch_size is None else max_batch_size,
    )
    buckets: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()
    for index, length in enumerate(prompt_lengths):
        bucket = config.bucket_index(length)
        if length > config.max_total_batch_tokens:
            raise ValueError(
                f"Prompt has {length} tokens, exceeding max_total_batch_tokens "
                f"({config.max_total_batch_tokens})"
            )
        buckets.setdefault(bucket, []).append((index, length))

    groups: list[tuple[int, ...]] = []
    for entries in buckets.values():
        current: list[tuple[int, int]] = []
        for entry in entries:
            candidate = [*current, entry]
            padded_tokens = max(length for _, length in candidate) * len(candidate)
            real_tokens = sum(length for _, length in candidate)
            padding_ratio = (padded_tokens - real_tokens) / real_tokens
            exceeds_limits = (
                len(candidate) > limit
                or padded_tokens > config.max_total_batch_tokens
                or padding_ratio > config.max_padding_ratio
            )
            if current and exceeds_limits:
                groups.append(tuple(index for index, _ in current))
                current = [entry]
            else:
                current = candidate
        if current:
            groups.append(tuple(index for index, _ in current))
    return groups


BatchCallback = Callable[
    [Sequence["GenerationRequest"]],
    Sequence["GenerationResult"] | Awaitable[Sequence["GenerationResult"]],
]
TokenCounter = Callable[["GenerationRequest"], int]


class DynamicBatcher:
    """Collect concurrent submissions for a short window and execute formed batches."""

    def __init__(
        self,
        batch_callback: BatchCallback,
        token_counter: TokenCounter,
        config: BatchConfig | None = None,
    ) -> None:
        self.config = config or BatchConfig()
        self._batch_callback = batch_callback
        self._token_counter = token_counter
        self._queue: asyncio.Queue[
            tuple[
                GenerationRequest,
                int,
                float,
                asyncio.Future[GenerationResult],
            ]
            | None
        ] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self.last_batch_metrics: tuple[BatchMetrics, ...] = ()

    async def submit(self, request: GenerationRequest) -> GenerationResult:
        if self._closed:
            raise RuntimeError("DynamicBatcher is closed")
        prompt_length = self._token_counter(request)
        self.config.bucket_index(prompt_length)
        if prompt_length > self.config.max_total_batch_tokens:
            raise ValueError(
                f"Prompt has {prompt_length} tokens, exceeding max_total_batch_tokens "
                f"({self.config.max_total_batch_tokens})"
            )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[GenerationResult] = loop.create_future()
        await self._queue.put((request, prompt_length, time.perf_counter(), future))
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())
        return await future

    async def close(self) -> None:
        if self._closed:
            if self._worker is not None:
                await self._worker
            return
        self._closed = True
        if self._worker is not None:
            await self._queue.put(None)
            await self._worker

    async def _run(self) -> None:
        self.last_batch_metrics = ()
        stop_after_batch = False
        while not stop_after_batch:
            first = await self._queue.get()
            if first is None:
                return
            items = [first]
            deadline = (
                asyncio.get_running_loop().time()
                + self.config.batching_window_ms / 1000.0
            )
            while len(items) < self.config.max_batch_size:
                if self._closed and self._queue.empty():
                    break
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._queue.get(), remaining)
                    except TimeoutError:
                        break
                if item is None:
                    stop_after_batch = True
                    break
                items.append(item)
            await self._execute(items)

    async def _execute(
        self,
        items: list[
            tuple[
                GenerationRequest,
                int,
                float,
                asyncio.Future[GenerationResult],
            ]
        ],
    ) -> None:
        formation_ms = (
            time.perf_counter() - min(queued_at for _, _, queued_at, _ in items)
        ) * 1000.0
        indices = form_batch_indices(
            [length for _, length, _, _ in items],
            self.config,
        )
        metrics: list[BatchMetrics] = []
        for group in indices:
            group_items = [items[index] for index in group]
            requests = [request for request, _, _, _ in group_items]
            prompt_lengths = [length for _, length, _, _ in group_items]
            padded_prompt_tokens = max(prompt_lengths) * len(prompt_lengths)
            real_prompt_tokens = sum(prompt_lengths)
            execution_started = time.perf_counter()
            try:
                if inspect.iscoroutinefunction(self._batch_callback):
                    results = await self._batch_callback(requests)
                else:
                    results = await asyncio.to_thread(self._batch_callback, requests)
                    if inspect.isawaitable(results):
                        results = await results
                if len(results) != len(group_items):
                    raise RuntimeError(
                        "batch callback returned a different number of results "
                        "than requests"
                    )
            except Exception as exc:
                for _, _, _, future in group_items:
                    if not future.done():
                        future.set_exception(exc)
                continue
            execution_elapsed = time.perf_counter() - execution_started
            generated_tokens = sum(
                getattr(result, "generated_token_count", 0) for result in results
            )
            metrics.append(BatchMetrics(
                effective_batch_size=len(group_items),
                prompt_tokens=real_prompt_tokens,
                padded_prompt_tokens=padded_prompt_tokens,
                padding_overhead_ratio=(
                    (padded_prompt_tokens - real_prompt_tokens) / real_prompt_tokens
                ),
                batch_formation_time_ms=formation_ms,
                execution_time_ms=execution_elapsed * 1000.0,
                generated_tokens=generated_tokens,
                aggregate_tokens_per_second=(
                    generated_tokens / execution_elapsed
                    if execution_elapsed > 0
                    else 0.0
                ),
            ))
            for result, (_, _, _, future) in zip(results, group_items):
                if not future.done():
                    future.set_result(result)
        self.last_batch_metrics += tuple(metrics)


async def benchmark_batching(
    generator: BatchGenerator,
    requests: Sequence["GenerationRequest"],
    concurrency_levels: Sequence[int],
    config: BatchConfig | None = None,
) -> list[BatchingBenchmarkResult]:
    """Compare single, unbucketed, and bucketed execution at each concurrency."""
    if not requests:
        raise ValueError("requests must not be empty")
    if not concurrency_levels or any(level <= 0 for level in concurrency_levels):
        raise ValueError("concurrency_levels must contain positive integers")

    bucketed_config = config or BatchConfig()
    modes = (
        ("single", None),
        ("unbucketed", replace(bucketed_config, prompt_length_buckets=None)),
        ("bucketed", bucketed_config),
    )
    benchmark_results: list[BatchingBenchmarkResult] = []
    for mode, batch_config in modes:
        for concurrency in concurrency_levels:
            latencies_ms = [0.0] * len(requests)
            semaphore = asyncio.Semaphore(concurrency)
            batcher: DynamicBatcher | None = None
            if batch_config is not None:
                batcher = DynamicBatcher(
                    lambda pending: generator.generate_batch(
                        list(pending), batch_config
                    ),
                    token_counter=lambda request: generator.count_prompt_tokens(
                        request.prompt
                    ),
                    config=batch_config,
                )

            async def run_one(
                index: int, request: "GenerationRequest"
            ) -> "GenerationResult":
                async with semaphore:
                    started = time.perf_counter()
                    if batcher is None:
                        result = await asyncio.to_thread(generator.generate, request)
                    else:
                        result = await batcher.submit(request)
                    latencies_ms[index] = (time.perf_counter() - started) * 1000.0
                    return result

            started = time.perf_counter()
            try:
                results = await asyncio.gather(
                    *(run_one(index, request) for index, request in enumerate(requests))
                )
            finally:
                if batcher is not None:
                    await batcher.close()
            elapsed = time.perf_counter() - started
            generated_tokens = sum(
                result.generated_token_count for result in results
            )
            sorted_latencies = sorted(latencies_ms)
            p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
            benchmark_results.append(BatchingBenchmarkResult(
                mode=mode,
                concurrency=concurrency,
                request_count=len(requests),
                elapsed_ms=elapsed * 1000.0,
                requests_per_second=(
                    len(requests) / elapsed if elapsed > 0 else 0.0
                ),
                generated_tokens_per_second=(
                    generated_tokens / elapsed if elapsed > 0 else 0.0
                ),
                mean_latency_ms=statistics.fmean(latencies_ms),
                p95_latency_ms=sorted_latencies[p95_index],
            ))
    return benchmark_results
