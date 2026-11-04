"""Adapter layer bridging optimization candidates to the InferBench benchmark engine."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Literal

from inferbench.benchmark.export import result_to_dict
from inferbench.benchmark.runner import Baseline, evaluate_config, failed_result, make_baseline
from inferbench.common.benchmarktypes import BenchmarkConfig, BenchmarkResult
from inferbench.optimizer.context import OptimizationContext, StrategyCandidate
from inferbench.optimizer.quantization import (
    load_samples,
    measure_quality,
    model_input_names,
    quantize_model,
    split_samples,
)
from inferbench.optimizer.registry import StrategyRegistry, get_tuning_registry

_GRAPH_LEVELS: dict[str, str] = {
    "ORT_DISABLE_ALL": "disable",
    "ORT_ENABLE_BASIC": "basic",
    "ORT_ENABLE_EXTENDED": "extended",
    "ORT_ENABLE_ALL": "all",
}
BASELINE_NUM_THREADS: int = 1
DEFAULT_ARTIFACT_DIR = Path("results") / "quantized"
FP32_STRATEGIES: tuple[str, ...] = ("graph_optimization", "cpu_thread_tuning")


def candidate_to_benchmark_config(
    base_config: BenchmarkConfig,
    candidate: StrategyCandidate,
    *,
    baseline_num_threads: int = BASELINE_NUM_THREADS,
) -> BenchmarkConfig:
    """Translate a strategy candidate into a validated BenchmarkConfig.

    Only maps fields supported by the benchmark runner. Safely ignores unsupported
    or experimental fields (e.g., inter_op_threads) without raising unexpected
    argument errors against BenchmarkConfig.

    Graph-optimization candidates use the baseline's thread count (one thread in
    the current AutoTune workflow) for consistent comparison with the baseline run,
    while CPU-thread-tuning candidates retain their explicitly generated thread counts.
    """
    config_dict = dict(candidate.config)

    graph_level = base_config.graph_optimization_level
    if "graph_optimization" in config_dict:
        raw_level = str(config_dict["graph_optimization"]).strip()
        graph_level = _GRAPH_LEVELS.get(raw_level.upper(), raw_level.lower())
    elif "graph_optimization_level" in config_dict:
        raw_level = str(config_dict["graph_optimization_level"]).strip()
        graph_level = _GRAPH_LEVELS.get(raw_level.upper(), raw_level.lower())

    if "intra_op_threads" in config_dict:
        num_threads = int(config_dict["intra_op_threads"])
    elif "num_threads" in config_dict:
        num_threads = int(config_dict["num_threads"])
    elif candidate.strategy == "graph_optimization":
        num_threads = baseline_num_threads
    else:
        num_threads = base_config.num_threads

    return replace(
        base_config,
        graph_optimization_level=graph_level,
        num_threads=num_threads,
    )


def quantized_candidate_to_benchmark_config(
    tuned_config: BenchmarkConfig,
    candidate: StrategyCandidate,
    artifact_path: Path | str,
) -> BenchmarkConfig:
    """Config that benchmarks the quantized file with the best FP32 runtime settings."""
    return replace(
        tuned_config,
        model_path=str(artifact_path),
        metadata={
            **tuned_config.metadata,
            "precision": "int8",
            "method": str(candidate.config.get("method", "static")),
            "source_model": tuned_config.model_path,
        },
    )


def benchmark_candidates(
    base_config: BenchmarkConfig,
    candidates: Iterable[StrategyCandidate],
    *,
    baseline: Baseline | None = None,
) -> list[BenchmarkResult]:
    """Evaluate candidate configurations against a single reusable baseline."""
    reference = baseline or make_baseline(base_config)
    results: list[BenchmarkResult] = []
    baseline_threads = getattr(reference.config, "num_threads", BASELINE_NUM_THREADS)

    for candidate in candidates:
        try:
            cfg = candidate_to_benchmark_config(
                base_config, candidate, baseline_num_threads=baseline_threads
            )
            result = evaluate_config(cfg, baseline=reference)
        except Exception as exc:
            result = failed_result(base_config, f"Candidate evaluation failed: {type(exc).__name__}: {exc}")
        results.append(result)

    return results


def benchmark_quantized_candidates(
    base_config: BenchmarkConfig,
    tuned_config: BenchmarkConfig,
    candidates: Iterable[StrategyCandidate],
    *,
    baseline: Baseline,
    feeds: list[dict],
    min_cosine: float,
    min_top1: float,
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
) -> list[BenchmarkResult]:
    """Quantize, accuracy-check and benchmark each INT8 candidate.

    Speed is measured with the best FP32 runtime settings (tuned_config), so the
    comparison isolates the effect of quantization itself.
    """
    calibration_feeds, eval_feeds = split_samples(feeds)
    results: list[BenchmarkResult] = []

    for candidate in candidates:
        method = str(candidate.config.get("method", "static"))
        try:
            artifact = quantize_model(
                base_config.model_path, artifact_dir, method, calibration_feeds=calibration_feeds
            )
            quality = measure_quality(
                base_config.model_path,
                artifact,
                eval_feeds,
                min_cosine=min_cosine,
                min_top1=min_top1,
            )
            cfg = quantized_candidate_to_benchmark_config(tuned_config, candidate, artifact)
            results.append(evaluate_config(cfg, baseline=baseline, correctness_override=quality))
        except Exception as exc:
            failed_cfg = replace(
                tuned_config,
                metadata={**tuned_config.metadata, "precision": "int8", "method": method},
            )
            results.append(
                failed_result(
                    failed_cfg,
                    f"INT8 {method} quantization failed: {type(exc).__name__}: {exc}",
                )
            )

    return results


def select_best(
    results: Iterable[BenchmarkResult],
    *,
    metric: Literal["latency_p50_ms", "throughput_per_sec"] = "latency_p50_ms",
) -> BenchmarkResult | None:
    """Select the best valid candidate based on the specified metric.

    Returns None if no candidate successfully ran with correct outputs (is_valid=True).
    """
    valid = [
        r for r in results
        if r.is_valid and getattr(r, metric, None) is not None
    ]
    if not valid:
        return None

    if metric == "latency_p50_ms":
        return min(valid, key=lambda r: getattr(r, metric))
    elif metric == "throughput_per_sec":
        return max(valid, key=lambda r: getattr(r, metric))
    else:
        # Default to latency_p50_ms
        return min(valid, key=lambda r: getattr(r, "latency_p50_ms", float("inf")))


def _run_quantization_stage(
    base_config: BenchmarkConfig,
    best_fp32: BenchmarkResult | None,
    baseline: Baseline,
    registry: StrategyRegistry,
    context: OptimizationContext,
    *,
    calibration_path: str | None,
    min_cosine: float,
    min_top1: float,
    artifact_dir: Path | str,
) -> tuple[list[BenchmarkResult], dict[str, str]]:
    """Stage 2: try INT8 variants using the best FP32 settings found in stage 1."""
    candidates, skipped = registry.generate_candidates(context, strategy_names=("quantization",))
    if not candidates:
        return [], skipped

    if not calibration_path:
        skipped["quantization"] = "No calibration samples file was provided."
        return [], skipped

    try:
        feeds = load_samples(calibration_path, model_input_names(base_config.model_path))
    except Exception as exc:
        skipped["quantization"] = f"Calibration samples unusable: {type(exc).__name__}: {exc}"
        return [], skipped

    tuned_config = best_fp32.config if best_fp32 is not None else base_config
    results = benchmark_quantized_candidates(
        base_config,
        tuned_config,
        candidates,
        baseline=baseline,
        feeds=feeds,
        min_cosine=min_cosine,
        min_top1=min_top1,
        artifact_dir=artifact_dir,
    )
    return results, skipped


def run_autotune(
    base_config: BenchmarkConfig,
    *,
    metric: Literal["latency_p50_ms", "throughput_per_sec"] = "latency_p50_ms",
    strategy_names: Iterable[str] | None = None,
    context: OptimizationContext | None = None,
    quantization: bool = False,
    calibration_path: str | None = None,
    min_cosine: float = 0.99,
    min_top1: float = 0.90,
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    """Execute complete AutoTune optimization workflow for a model.

    Stage 1 searches graph optimization levels and thread counts (FP32).
    Stage 2 (only when quantization=True) builds INT8 variants, keeps those whose
    accuracy stays close to FP32, and benchmarks them with the stage-1 winner's settings.
    """
    if context is None:
        context = OptimizationContext(
            execution_providers=(
                base_config.execution_provider,
            ),
        )
    registry = get_tuning_registry(include_quantization=quantization)
    strategies_to_run = strategy_names if strategy_names is not None else FP32_STRATEGIES
    candidates, skipped = registry.generate_candidates(context, strategy_names=strategies_to_run)

    try:
        baseline = make_baseline(base_config)
    except Exception as exc:
        base_failed = failed_result(base_config, f"Baseline run failed: {type(exc).__name__}: {exc}")
        return {
            "model_path": base_config.model_path,
            "metric": metric,
            "baseline": result_to_dict(base_failed),
            "best": None,
            "speedup": None,
            "throughput_change_percent": None,
            "candidates_evaluated": 0,
            "candidates_valid": 0,
            "candidates": [],
            "skipped_strategies": skipped,
            "error_message": f"Baseline run failed: {type(exc).__name__}: {exc}",
        }

    fp32_results = benchmark_candidates(base_config, candidates, baseline=baseline)
    best_fp32 = select_best(fp32_results, metric=metric)

    quantized_results: list[BenchmarkResult] = []
    if quantization:
        quantized_results, quant_skipped = _run_quantization_stage(
            base_config,
            best_fp32,
            baseline,
            registry,
            context,
            calibration_path=calibration_path,
            min_cosine=min_cosine,
            min_top1=min_top1,
            artifact_dir=artifact_dir,
        )
        skipped.update(quant_skipped)

    candidate_results = fp32_results + quantized_results
    best_candidate = select_best(candidate_results, metric=metric)

    # Compute latency speedup relative to baseline P50 latency.
    speedup: float | None = None
    if (
        best_candidate is not None
        and baseline.result.is_valid
        and baseline.result.latency_p50_ms is not None
        and best_candidate.latency_p50_ms is not None
        and best_candidate.latency_p50_ms > 0
    ):
        speedup = round(
            baseline.result.latency_p50_ms / best_candidate.latency_p50_ms,
            2,
        )

    # Compute throughput improvement relative to baseline.
    throughput_change_percent: float | None = None
    baseline_throughput = baseline.result.throughput_per_sec
    best_throughput = (
        best_candidate.throughput_per_sec
        if best_candidate is not None
        else None
    )

    if (
        best_candidate is not None
        and baseline.result.is_valid
        and baseline_throughput is not None
        and baseline_throughput > 0
        and best_throughput is not None
    ):
        throughput_change_percent = round(
            (best_throughput / baseline_throughput - 1) * 100,
            2,
        )

    # How much does INT8 add on top of the best FP32 settings? (above 1.0 = INT8 faster)
    quantization_speedup: float | None = None
    best_quantized = select_best(quantized_results, metric=metric)
    if (
        best_fp32 is not None
        and best_quantized is not None
        and best_fp32.latency_p50_ms
        and best_quantized.latency_p50_ms
        and best_quantized.latency_p50_ms > 0
    ):
        quantization_speedup = round(
            best_fp32.latency_p50_ms / best_quantized.latency_p50_ms, 2
        )

    return {
        "model_path": base_config.model_path,
        "metric": metric,
        "baseline": result_to_dict(baseline.result),
        "best": result_to_dict(best_candidate) if best_candidate is not None else None,
        "best_fp32": result_to_dict(best_fp32) if best_fp32 is not None else None,
        "speedup": speedup,
        "quantization_speedup_vs_best_fp32": quantization_speedup,
        "throughput_change_percent": throughput_change_percent,
        "candidates_evaluated": len(candidate_results),
        "candidates_valid": sum(1 for r in candidate_results if r.is_valid),
        "candidates": [result_to_dict(r) for r in candidate_results],
        "skipped_strategies": skipped,
    }