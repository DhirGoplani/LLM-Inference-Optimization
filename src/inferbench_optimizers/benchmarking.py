"""Connect optimizer candidates to InferBench's benchmark runner."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Iterable, Literal

from inferbench.common.benchmarktypes import BenchmarkConfig, BenchmarkResult

from .base import StrategyCandidate

if TYPE_CHECKING:
    from inferbench.benchmark.runner import Baseline

_GRAPH_LEVELS = {
    "ORT_DISABLE_ALL": "disable",
    "ORT_ENABLE_BASIC": "basic",
    "ORT_ENABLE_EXTENDED": "extended",
    "ORT_ENABLE_ALL": "all",
}
_SUPPORTED_FIELDS = {
    "batch_size",
    "enable_mem_pattern",
    "execution_mode",
    "execution_provider",
    "execution_providers",
    "graph_optimization_level",
    "inter_op_num_threads",
    "num_threads",
}


def config_for_candidate(
    base_config: BenchmarkConfig, candidate: StrategyCandidate
) -> BenchmarkConfig:
    """Translate a strategy candidate into a validated benchmark config."""
    values = dict(candidate.config)
    aliases = {
        "graph_optimization": "graph_optimization_level",
        "intra_op_threads": "num_threads",
        "inter_op_threads": "inter_op_num_threads",
    }
    translated = {aliases.get(key, key): value for key, value in values.items()}
    unsupported = set(translated) - _SUPPORTED_FIELDS
    if unsupported:
        raise ValueError(
            f"Candidate {candidate.strategy!r} requires an artifact adapter for: "
            f"{', '.join(sorted(unsupported))}"
        )
    if "graph_optimization_level" in translated:
        level = str(translated["graph_optimization_level"]).upper()
        translated["graph_optimization_level"] = _GRAPH_LEVELS.get(
            level, str(translated["graph_optimization_level"]).lower()
        )
    if "execution_mode" in translated:
        mode = str(translated["execution_mode"]).upper()
        translated["execution_mode"] = {
            "ORT_SEQUENTIAL": "sequential",
            "ORT_PARALLEL": "parallel",
        }.get(mode, str(translated["execution_mode"]).lower())
    return replace(
        base_config,
        **translated,
        metadata={**base_config.metadata, "candidate": candidate.config},
    )


def benchmark_candidates(
    base_config: BenchmarkConfig,
    candidates: Iterable[StrategyCandidate],
    *,
    baseline: "Baseline | None" = None,
) -> list[BenchmarkResult]:
    """Evaluate every candidate against one reusable correctness baseline."""
    from inferbench.benchmark.runner import evaluate_config, make_baseline

    reference = baseline or make_baseline(base_config)
    results = []
    for candidate in candidates:
        try:
            config = config_for_candidate(base_config, candidate)
        except ValueError as exc:
            from inferbench.benchmark.runner import failed_result

            results.append(failed_result(base_config, exc))
            continue
        results.append(evaluate_config(config, reference))
    return results


def select_best(
    results: Iterable[BenchmarkResult],
    *,
    metric: Literal["latency_p50_ms", "throughput_per_sec"] = "latency_p50_ms",
) -> BenchmarkResult | None:
    """Return the best valid result, or None when no candidate is valid."""
    valid = [
        result for result in results
        if result.is_valid and getattr(result, metric) is not None
    ]
    if not valid:
        return None
    return (
        min(valid, key=lambda result: getattr(result, metric))
        if metric == "latency_p50_ms"
        else max(valid, key=lambda result: getattr(result, metric))
    )
