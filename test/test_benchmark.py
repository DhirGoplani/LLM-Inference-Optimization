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
    for key in ["mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "throughput_per_sec", "memory_mb", "peak_memory_mb", "environment"]:
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


# ---------- conversion to the shared BenchmarkResult ----------

def test_to_benchmark_result_fills_fields(tiny_model):
    from inferbench.benchmark.runner import to_benchmark_result
    from inferbench.common.benchmarktypes import BenchmarkConfig

    cfg = BenchmarkConfig(model_path=str(tiny_model), warmup_runs=1, timed_runs=5)
    data, out, _ = run_benchmark(tiny_model, cfg)
    check = compare_outputs(out, out)
    res = to_benchmark_result(data, cfg, check)
    assert res.success is True
    assert res.latency_mean_ms == data["mean_ms"]
    assert res.latency_p95_ms == data["p95_ms"]
    assert res.throughput_per_sec == data["throughput_per_sec"]
    assert res.peak_memory_mb == data["peak_memory_mb"]
    assert res.correct is True and res.is_valid
    assert res.to_dict()["config"]["timed_runs"] == 5


def test_to_benchmark_result_without_correctness_is_not_valid(tiny_model):
    from inferbench.benchmark.runner import to_benchmark_result

    data, _, _ = run_benchmark(tiny_model, FAST)
    res = to_benchmark_result(data)            # no config, no correctness
    assert res.correct is None
    assert res.is_valid is False               # never trusted without a check


def test_failed_result_is_marked_failed():
    from inferbench.benchmark.runner import failed_result
    from inferbench.common.benchmarktypes import BenchmarkConfig

    res = failed_result(BenchmarkConfig(model_path="x.onnx"), "boom")
    assert res.success is False and res.error_message == "boom"
    assert res.is_valid is False


# ---------- the correctness gate (evaluate_config) ----------

def _make_tiny(path, seed):
    weights = numpy_helper.from_array(
        np.random.default_rng(seed).standard_normal((16, 8)).astype(np.float32), "W")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["y"])], "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 8])], [weights])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _cfg_for(path, **kw):
    from inferbench.common.benchmarktypes import BenchmarkConfig
    return BenchmarkConfig(model_path=str(path), warmup_runs=1, timed_runs=3, **kw)


def test_good_config_is_valid(tiny_model):
    from inferbench.benchmark.runner import evaluate_config
    res = evaluate_config(_cfg_for(tiny_model, num_threads=2, graph_optimization_level="all"))
    assert res.success and res.correct is True and res.is_valid


def test_baseline_can_be_reused(tiny_model):
    from inferbench.benchmark.runner import evaluate_config, make_baseline
    base = make_baseline(_cfg_for(tiny_model))
    assert base.result.is_valid
    for threads in (1, 2):
        assert evaluate_config(_cfg_for(tiny_model, num_threads=threads), base).is_valid


def test_wrong_outputs_are_rejected(tiny_model, tmp_path):
    from inferbench.benchmark.runner import evaluate_config, make_baseline
    other = _make_tiny(tmp_path / "other.onnx", seed=99)   # same shape, different weights
    wrong_baseline = make_baseline(_cfg_for(other))
    res = evaluate_config(_cfg_for(tiny_model), wrong_baseline)
    assert res.success is True and res.correct is False
    assert res.is_valid is False                           # rejected
    assert res.max_abs_diff > 0


def test_crashing_config_does_not_raise(tmp_path):
    from inferbench.benchmark.runner import evaluate_config
    res = evaluate_config(_cfg_for(tmp_path / "missing.onnx"))
    assert res.success is False and res.error_message
    assert res.is_valid is False


# ---------- JSON export ----------

def _strict_loads(text):
    import json

    def refuse(name):
        raise ValueError(f"not valid JSON: {name}")
    return json.loads(text, parse_constant=refuse)   # rejects Infinity / NaN


def test_save_and_load_roundtrip(tiny_model, tmp_path):
    from inferbench.benchmark.export import load_results, save_results
    from inferbench.benchmark.runner import evaluate_config

    res = evaluate_config(_cfg_for(tiny_model))
    path = save_results(res, tmp_path / "out" / "run.json")   # folder is created
    doc = load_results(path)
    assert doc["schema_version"] == 1
    row = doc["results"][0]
    assert row["is_valid"] is True
    assert row["latency_mean_ms"] == res.latency_mean_ms
    assert row["config"]["timed_runs"] == 3
    assert "os" in row["environment"]


def test_many_results_in_one_file(tiny_model, tmp_path):
    from inferbench.benchmark.export import load_results, save_results
    from inferbench.benchmark.runner import evaluate_config, failed_result

    good = evaluate_config(_cfg_for(tiny_model))
    bad = failed_result(_cfg_for(tiny_model), "boom")
    doc = load_results(save_results([good, bad], tmp_path / "many.json"))
    assert [r["success"] for r in doc["results"]] == [True, False]
    assert doc["results"][1]["error_message"] == "boom"


def test_infinite_values_become_null(tiny_model, tmp_path):
    from inferbench.benchmark.export import results_to_json
    from inferbench.benchmark.runner import evaluate_config

    res = evaluate_config(_cfg_for(tiny_model))
    res.max_abs_diff = float("inf")
    row = _strict_loads(results_to_json(res))["results"][0]   # would raise on Infinity
    assert row["max_abs_diff"] is None


def test_default_filename_uses_model_name(tiny_model, tmp_path):
    from inferbench.benchmark.export import save_results
    from inferbench.benchmark.runner import evaluate_config

    path = save_results(evaluate_config(_cfg_for(tiny_model)), results_dir=tmp_path)
    assert path.parent == tmp_path and path.name.startswith("tiny_") and path.suffix == ".json"


# ---------- peak memory ----------

def test_monitor_catches_a_short_memory_spike():
    import time
    from inferbench.benchmark.metrics import PeakMemoryMonitor

    with PeakMemoryMonitor(interval=0.005) as mon:
        big = np.ones(25_000_000)          # ~200 MB, touched so it is really used
        time.sleep(0.1)
        del big
    assert mon.delta_mb > 100
    assert mon.peak_mb >= mon.start_mb


def test_monitor_without_spike_stays_close():
    from inferbench.benchmark.metrics import PeakMemoryMonitor

    with PeakMemoryMonitor() as mon:
        sum(range(1000))
    assert mon.peak_mb >= mon.start_mb
    assert mon.delta_mb < 50


def test_run_benchmark_reports_peak_memory(tiny_model):
    result, _, _ = run_benchmark(tiny_model, FAST)
    assert result["peak_memory_mb"] > 0
    assert result["peak_memory_mb"] >= result["peak_memory_delta_mb"] >= 0