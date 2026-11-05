"""Benchmark data structures for autoregressive LLM generation."""

from dataclasses import dataclass
from typing import Any

from inferbench.scheduler.metrics import summarize


@dataclass(frozen=True)
class GenerationBenchmarkMetrics:
    time_to_first_token_ms: float
    decode_latency_ms: float
    inter_token_latency_ms: float
    generated_token_count: int
    tokens_per_second: float
    # Serving-side numbers. They stay 0.0 for a plain single-request run with no queue.
    queue_wait_ms: float = 0.0
    end_to_end_ms: float = 0.0

    @classmethod
    def from_result(
        cls,
        result: Any,
        queue_wait_ms: float = 0.0,
        end_to_end_ms: float = 0.0,
    ) -> "GenerationBenchmarkMetrics":
        return cls(
            time_to_first_token_ms=result.time_to_first_token_ms,
            decode_latency_ms=result.decode_latency_ms,
            inter_token_latency_ms=result.inter_token_latency_ms,
            generated_token_count=result.generated_token_count,
            tokens_per_second=result.tokens_per_second,
            queue_wait_ms=queue_wait_ms,
            end_to_end_ms=end_to_end_ms,
        )


def summarize_generation_results(results: list[Any]) -> dict[str, Any]:
    """Mean / p50 / p95 / max of the per-request engine metrics of many results."""
    return {
        "requests": len(results),
        "generated_tokens": sum(int(r.generated_token_count) for r in results),
        "time_to_first_token_ms": summarize([float(r.time_to_first_token_ms) for r in results]),
        "inter_token_latency_ms": summarize([float(r.inter_token_latency_ms) for r in results]),
        "decode_latency_ms": summarize([float(r.decode_latency_ms) for r in results]),
        "tokens_per_second": summarize([float(r.tokens_per_second) for r in results]),
    }