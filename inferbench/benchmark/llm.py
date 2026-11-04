"""Benchmark data structures for autoregressive LLM generation."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationBenchmarkMetrics:
    time_to_first_token_ms: float
    decode_latency_ms: float
    inter_token_latency_ms: float
    generated_token_count: int
    tokens_per_second: float

    @classmethod
    def from_result(cls, result: Any) -> "GenerationBenchmarkMetrics":
        return cls(
            time_to_first_token_ms=result.time_to_first_token_ms,
            decode_latency_ms=result.decode_latency_ms,
            inter_token_latency_ms=result.inter_token_latency_ms,
            generated_token_count=result.generated_token_count,
            tokens_per_second=result.tokens_per_second,
        )
