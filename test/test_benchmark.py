"""Tests for inferbench.benchmark (metrics, correctness, runner).

Run from the project root with:  python -m pytest -v
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from inferbench.benchmark.correctness import compare_outputs
from inferbench.benchmark.metrics import latency_stats, memory_mb, throughput_per_sec
from inferbench.benchmark.runner import (
    compare_baseline_and_optimized,
    create_session,
    make_dummy_inputs,
    run_benchmark,
)


# ---------- a tiny model so tests are fast and need no downloads ----------

@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    weights = numpy_helper.from_array(
        np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32), "W"
    )
    node = helper.make_node("MatMul", ["x", "W"], ["y"])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 8])],
        [weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path_factory.mktemp("models") / "tiny.onnx"
    onnx.save(model, str(path))
    return path


FAST = SimpleNamespace(warmup_runs=2, num_runs=10, batch_size=1)


# ---------- metrics ----------

def test_latency_stats_values():
    stats = latency_stats([1, 2, 3, 4, 5])
    assert stats["mean_ms"] == 3
    assert stats["p50_ms"] == 3
    assert stats["min_ms"] == 1
    assert stats["max_ms"] == 5
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]


def test_latency_stats_empty_raises():
    with pytest.raises(ValueError):
        latency_stats([])


def test_throughput():
    # 100 runs in 2 seconds with batch 4 -> 200 samples per second
    assert throughput_per_sec(100, 2.0, batch_size=4) == 200
    assert throughput_per_sec(100, 0) == 0.0


def test_memory_is_positive():
    assert memory_mb() > 0


# ---------- correctness ----------

def test_identical_outputs_pass():
    a = [np.ones((2, 3), dtype=np.float32)]
    result = compare_outputs(a, [a[0].copy()])
    assert result["passed"] is True
    assert result["max_abs_diff"] == 0


def test_small_difference_within_tolerance_passes():
    a = [np.ones(5, dtype=np.float32)]
    b = [np.ones(5, dtype=np.float32) + 1e-6]
    assert compare_outputs(a, b)["passed"] is True


def test_big_difference_fails():
    a = [np.ones(5, dtype=np.float32)]
    b = [np.ones(5, dtype=np.float32) * 2]
    result = compare_outputs(a, b)
    assert result["passed"] is False
    assert result["max_abs_diff"] == pytest.approx(1.0)


def test_shape_mismatch_fails():
    result = compare_outputs([np.ones((2, 3))], [np.ones((3, 2))])
    assert result["passed"] is False


def test_output_count_mismatch_fails():
    result = compare_outputs([np.ones(3)], [np.ones(3), np.ones(3)])
    assert result["passed"] is False


def test_nan_mismatch_fails():
    a = [np.array([1.0, np.nan])]
    b = [np.array([1.0, 2.0])]
    assert compare_outputs(a, b)["passed"] is False


# ---------- runner ----------

def test_dummy_inputs_match_model(tiny_model):
    session = create_session(tiny_model)
    inputs = make_dummy_inputs(session, batch_size=3)
    assert inputs["x"].shape == (3, 16)
    assert inputs["x"].dtype == np.float32


def test_run_benchmark_returns_expected_keys(tiny_model):
    result, outputs, _ = run_benchmark(tiny_model, FAST)
    for key in ["mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_per_sec", "memory_mb", "environment"]:
        assert key in result
    assert result["num_runs"] == 10
    assert result["mean_ms"] > 0
    assert outputs[0].shape == (1, 8)


def test_same_input_gives_same_output(tiny_model):
    _, out1, inputs = run_benchmark(tiny_model, FAST)
    _, out2, _ = run_benchmark(tiny_model, FAST, inputs=inputs)
    assert compare_outputs(out1, out2)["passed"] is True


def test_compare_baseline_and_optimized(tiny_model):
    report = compare_baseline_and_optimized(
        tiny_model,
        tiny_model,
        SimpleNamespace(warmup_runs=2, num_runs=10, graph_optimization_level="disable"),
        SimpleNamespace(warmup_runs=2, num_runs=10, graph_optimization_level="all", num_threads=2),
    )
    assert report["correctness"]["passed"] is True
    assert report["speedup"] > 0


def test_missing_config_uses_defaults(tiny_model):
    # No config at all should still work (defaults: 10 warm-up, 100 runs)
    result, _, _ = run_benchmark(tiny_model, None)
    assert result["num_runs"] == 100


# ---------- optional: real model, skipped if it isn't there ----------

MOBILENET = Path(__file__).resolve().parent.parent / "models" / "mobilenet_v2.onnx"


@pytest.mark.skipif(not MOBILENET.exists(), reason="mobilenet_v2.onnx not found")
def test_mobilenet_benchmark():
    result, outputs, _ = run_benchmark(MOBILENET, SimpleNamespace(warmup_runs=1, num_runs=3))
    assert result["mean_ms"] > 0
    assert len(outputs) >= 1

# ---------- shared BenchmarkConfig is really used ----------

def test_shared_config_fields_are_used(tiny_model):
    from inferbench.common.benchmarktypes import BenchmarkConfig

    cfg = BenchmarkConfig(model_path=str(tiny_model), timed_runs=7, warmup_runs=1,
                          batch_size=3, execution_provider="CPUExecutionProvider")
    result, outputs, inputs = run_benchmark(tiny_model, cfg)
    assert result["num_runs"] == 7          # timed_runs is respected
    assert result["batch_size"] == 3
    assert inputs["x"].shape == (3, 16)
    assert result["environment"]["providers"] == ["CPUExecutionProvider"]


def test_seed_changes_inputs(tiny_model):
    from inferbench.common.benchmarktypes import BenchmarkConfig

    a = BenchmarkConfig(model_path=str(tiny_model), warmup_runs=1, timed_runs=2, seed=1)
    b = BenchmarkConfig(model_path=str(tiny_model), warmup_runs=1, timed_runs=2, seed=2)
    _, _, in_a = run_benchmark(tiny_model, a)
    _, _, in_b = run_benchmark(tiny_model, b)
    assert not np.array_equal(in_a["x"], in_b["x"])


def test_tolerances_come_from_config(tiny_model):
    from inferbench.common.benchmarktypes import BenchmarkConfig

    loose = BenchmarkConfig(model_path=str(tiny_model), warmup_runs=1, timed_runs=2, rel_tol=0.5)
    report = compare_baseline_and_optimized(tiny_model, tiny_model, loose, loose)
    assert report["correctness"]["passed"] is True