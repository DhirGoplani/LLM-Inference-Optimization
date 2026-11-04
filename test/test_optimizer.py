"""Unit tests for the InferBench optimizer adapter and strategy candidates."""

import pytest
from inferbench.common.benchmarktypes import BenchmarkConfig, BenchmarkResult
from inferbench.optimizer.adapter import (
    BASELINE_NUM_THREADS,
    benchmark_candidates,
    candidate_to_benchmark_config,
    select_best,
)
from inferbench.optimizer.context import OptimizationContext, StrategyCandidate
from inferbench.optimizer.registry import get_tuning_registry
from inferbench.optimizer.strategies import CPUThreadTuningStrategy, GraphOptimizationStrategy


def test_strategy_applicability():
    ctx = OptimizationContext(execution_providers=("CPUExecutionProvider",))
    graph_strat = GraphOptimizationStrategy()
    cpu_strat = CPUThreadTuningStrategy()

    applicable, reason = graph_strat.is_applicable(ctx)
    assert applicable is True
    assert reason is None

    applicable, reason = cpu_strat.is_applicable(ctx)
    assert applicable is True
    assert reason is None

    # Incompatible runtime
    bad_ctx = OptimizationContext(runtime="torch")
    applicable, reason = graph_strat.is_applicable(bad_ctx)
    assert applicable is False
    assert "supported for ONNX models" in reason


def test_candidate_generation():
    ctx = OptimizationContext(logical_cpu_count=4)
    registry = get_tuning_registry()
    candidates, skipped = registry.generate_candidates(ctx)

    assert len(skipped) == 0
    assert len(candidates) > 0

    # Ensure levels from graph optimization are present
    graph_candidates = [c for c in candidates if c.strategy == "graph_optimization"]
    levels = [c.config["graph_optimization"] for c in graph_candidates]
    assert "ORT_DISABLE_ALL" in levels
    assert "ORT_ENABLE_ALL" in levels

    # Ensure thread candidates are bounded by logical_cpu_count
    cpu_candidates = [c for c in candidates if c.strategy == "cpu_thread_tuning"]
    thread_counts = [c.config["intra_op_threads"] for c in cpu_candidates]
    assert 1 in thread_counts
    assert max(thread_counts) <= 4


def test_candidate_to_benchmark_config_translation():
    base = BenchmarkConfig(model_path="dummy.onnx", graph_optimization_level="all", num_threads=0)

    # Test graph optimization mapping: must use baseline thread count (1), not base_config's 0
    c_graph = StrategyCandidate(
        strategy="graph_optimization",
        config={"graph_optimization": "ORT_ENABLE_BASIC"},
        description="basic opt",
    )
    cfg1 = candidate_to_benchmark_config(base, c_graph)
    assert cfg1.graph_optimization_level == "basic"
    assert cfg1.num_threads == BASELINE_NUM_THREADS
    assert cfg1.num_threads == 1

    # Test thread count mapping: must retain requested thread count (4), not baseline 1
    c_thread = StrategyCandidate(
        strategy="cpu_thread_tuning",
        config={"intra_op_threads": 4, "inter_op_threads": 1},
        description="4 threads",
    )
    cfg2 = candidate_to_benchmark_config(base, c_thread)
    assert cfg2.num_threads == 4
    assert cfg2.graph_optimization_level == "all"
    # Ensure no extra attributes like metadata were forced onto BenchmarkConfig
    assert not hasattr(cfg2, "inter_op_threads")
    assert cfg2.inter_op_num_threads == 0


def test_graph_candidates_use_baseline_thread_count():
    """Graph optimization candidates must consistently use the baseline thread count."""
    base = BenchmarkConfig(model_path="dummy.onnx", graph_optimization_level="all", num_threads=0)
    strat = GraphOptimizationStrategy()
    ctx = OptimizationContext()
    candidates = strat.generate_candidates(ctx)

    expected_levels = {
        "ORT_DISABLE_ALL": "disable",
        "ORT_ENABLE_BASIC": "basic",
        "ORT_ENABLE_EXTENDED": "extended",
        "ORT_ENABLE_ALL": "all",
    }

    for c in candidates:
        cfg = candidate_to_benchmark_config(base, c)
        raw_level = c.config["graph_optimization"]
        assert cfg.graph_optimization_level == expected_levels[raw_level]
        # Must be 1 thread (baseline), not 0 from base
        assert cfg.num_threads == 1

    # If an explicit baseline thread count is provided, it should be respected
    cfg_custom = candidate_to_benchmark_config(base, candidates[0], baseline_num_threads=2)
    assert cfg_custom.num_threads == 2


def test_cpu_thread_candidates_retain_requested_thread_counts():
    """CPU thread candidates must retain their explicitly generated thread counts."""
    base = BenchmarkConfig(model_path="dummy.onnx", graph_optimization_level="extended", num_threads=0)
    for requested_threads in (1, 2, 4, 8):
        c = StrategyCandidate(
            strategy="cpu_thread_tuning",
            config={"intra_op_threads": requested_threads, "inter_op_threads": 1},
            description=f"{requested_threads} threads",
        )
        cfg = candidate_to_benchmark_config(base, c)
        assert cfg.num_threads == requested_threads
        assert cfg.graph_optimization_level == "extended"
        assert not hasattr(cfg, "inter_op_threads")


def test_graph_level_translation_and_unsupported_inter_op_threads():
    """Graph optimization levels translate properly while unsupported inter_op_threads is ignored."""
    base = BenchmarkConfig(model_path="dummy.onnx", graph_optimization_level="disable", num_threads=1)

    # Lowercase / direct level mapping
    c_level = StrategyCandidate(
        strategy="graph_optimization",
        config={"graph_optimization_level": "all"},
        description="all opt",
    )
    cfg_level = candidate_to_benchmark_config(base, c_level)
    assert cfg_level.graph_optimization_level == "all"
    assert cfg_level.num_threads == 1

    # Unsupported inter_op_threads ignored safely without attribute creation
    c_inter_op = StrategyCandidate(
        strategy="cpu_thread_tuning",
        config={"intra_op_threads": 2, "inter_op_threads": 4},
        description="2 intra, 4 inter",
    )
    cfg_inter = candidate_to_benchmark_config(base, c_inter_op)
    assert cfg_inter.num_threads == 2
    assert not hasattr(cfg_inter, "inter_op_threads")
    assert cfg_inter.inter_op_num_threads == 0


def test_benchmark_candidates_thread_consistency(monkeypatch):
    """benchmark_candidates evaluates graph candidates with 1 thread and thread candidates with their count."""
    from unittest.mock import MagicMock

    base = BenchmarkConfig(model_path="dummy.onnx", graph_optimization_level="all", num_threads=0)

    # Create dummy candidates
    graph_cand = StrategyCandidate(
        strategy="graph_optimization",
        config={"graph_optimization": "ORT_ENABLE_BASIC"},
        description="basic opt",
    )
    thread_cand = StrategyCandidate(
        strategy="cpu_thread_tuning",
        config={"intra_op_threads": 4, "inter_op_threads": 1},
        description="4 threads",
    )

    evaluated_configs = []

    def mock_eval(cfg, baseline=None):
        evaluated_configs.append(cfg)
        return BenchmarkResult(config=cfg, success=True, correct=True, latency_p50_ms=5.0)

    # Mock make_baseline to return a dummy baseline with num_threads=1
    dummy_base_cfg = BenchmarkConfig(model_path="dummy.onnx", graph_optimization_level="disable", num_threads=1)
    from inferbench.benchmark.runner import Baseline
    dummy_baseline = Baseline(
        config=dummy_base_cfg,
        result=BenchmarkResult(config=dummy_base_cfg, success=True, correct=True, latency_p50_ms=10.0),
        outputs=[],
        inputs={},
    )

    monkeypatch.setattr("inferbench.optimizer.adapter.evaluate_config", mock_eval)

    results = benchmark_candidates(base, [graph_cand, thread_cand], baseline=dummy_baseline)

    assert len(results) == 2
    assert len(evaluated_configs) == 2

    # Graph candidate evaluated with baseline thread count (1)
    assert evaluated_configs[0].graph_optimization_level == "basic"
    assert evaluated_configs[0].num_threads == 1

    # Thread candidate evaluated with explicit thread count (4)
    assert evaluated_configs[1].graph_optimization_level == "all"
    assert evaluated_configs[1].num_threads == 4



def test_select_best_by_latency():
    cfg = BenchmarkConfig(model_path="dummy.onnx")
    r1 = BenchmarkResult(config=cfg, success=True, correct=True, latency_p50_ms=10.0)
    r2 = BenchmarkResult(config=cfg, success=True, correct=True, latency_p50_ms=5.0)  # best
    r3 = BenchmarkResult(config=cfg, success=True, correct=False, latency_p50_ms=2.0)  # invalid output
    r4 = BenchmarkResult(config=cfg, success=False, correct=None, latency_p50_ms=1.0)  # crashed

    best = select_best([r1, r2, r3, r4], metric="latency_p50_ms")
    assert best is not None
    assert best.latency_p50_ms == 5.0


def test_select_best_by_throughput():
    cfg = BenchmarkConfig(model_path="dummy.onnx")
    r1 = BenchmarkResult(config=cfg, success=True, correct=True, throughput_per_sec=100.0)
    r2 = BenchmarkResult(config=cfg, success=True, correct=True, throughput_per_sec=250.0)  # best
    r3 = BenchmarkResult(config=cfg, success=True, correct=False, throughput_per_sec=500.0)  # invalid output

    best = select_best([r1, r2, r3], metric="throughput_per_sec")
    assert best is not None
    assert best.throughput_per_sec == 250.0


def test_select_best_no_valid():
    cfg = BenchmarkConfig(model_path="dummy.onnx")
    r_crashed = BenchmarkResult(config=cfg, success=False, error_message="fail")
    r_incorrect = BenchmarkResult(config=cfg, success=True, correct=False)

    assert select_best([r_crashed, r_incorrect]) is None
    assert select_best([]) is None
