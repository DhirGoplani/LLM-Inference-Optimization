"""Runs an ONNX model many times and measures how fast it is."""

import dataclasses
import platform
import time

import numpy as np
import onnxruntime as ort

from inferbench.benchmark.correctness import compare_outputs
from inferbench.benchmark.metrics import PeakMemoryMonitor, latency_stats, memory_mb, throughput_per_sec

# Map ONNX type strings to numpy types
_ONNX_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def _cfg(config, name, default):
    """Read a setting from the config, or use a default if the config doesn't have it."""
    value = getattr(config, name, None)
    return default if value is None else value


def _cfg_first(config, names, default):
    """Try several setting names in order (e.g. the new name, then the old one)."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def create_session(model_path, config=None):
    """Open an ONNX model with the settings from the config."""
    options = ort.SessionOptions()

    threads = _cfg(config, "num_threads", 0)  # 0 = let ONNX Runtime decide
    options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = int(_cfg(config, "inter_op_num_threads", 0))

    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    level_name = str(_cfg(config, "graph_optimization_level", "all")).lower()
    level_name = {
        "ort_disable_all": "disable",
        "ort_enable_basic": "basic",
        "ort_enable_extended": "extended",
        "ort_enable_all": "all",
    }.get(level_name, level_name)
    options.graph_optimization_level = levels.get(level_name, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

    mode = str(_cfg(config, "execution_mode", "sequential")).lower()
    modes = {
        "ort_sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
        "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
        "ort_parallel": ort.ExecutionMode.ORT_PARALLEL,
        "parallel": ort.ExecutionMode.ORT_PARALLEL,
    }
    options.execution_mode = modes.get(mode, ort.ExecutionMode.ORT_SEQUENTIAL)
    options.enable_mem_pattern = bool(_cfg(config, "enable_mem_pattern", True))

    configured_providers = getattr(config, "execution_providers", None)
    if configured_providers:
        providers = configured_providers
    else:
        providers = [_cfg(config, "execution_provider", "CPUExecutionProvider")]
    return ort.InferenceSession(str(model_path), sess_options=options, providers=list(providers))


def make_dummy_inputs(session, batch_size=1, seed=0):
    """Build fake input data that matches what the model expects.

    Unknown sizes (like a dynamic batch) are replaced by batch_size for the
    first dimension and 1 for the others. Integer inputs are filled with zeros
    (safe for things like token ids), float inputs get random numbers.
    """
    rng = np.random.default_rng(seed)
    inputs = {}
    for inp in session.get_inputs():
        shape = []
        for i, dim in enumerate(inp.shape):
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                shape.append(batch_size if i == 0 else 1)
        dtype = _ONNX_TO_NUMPY.get(inp.type)
        if dtype is None:
            raise ValueError(f"Unsupported input type: {inp.type}")
        if np.issubdtype(dtype, np.floating):
            inputs[inp.name] = rng.standard_normal(shape).astype(dtype)
        else:
            inputs[inp.name] = np.zeros(shape, dtype=dtype)
    return inputs


def time_session(session, inputs, warmup_runs=10, num_runs=100):
    """Warm up, then time num_runs runs. Returns (latencies_ms, total_seconds, last_outputs)."""
    for _ in range(warmup_runs):  # warm-up runs are not timed
        session.run(None, inputs)

    latencies_ms = []
    outputs = None
    start_all = time.perf_counter()
    for _ in range(num_runs):
        start = time.perf_counter()
        outputs = session.run(None, inputs)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    total_seconds = time.perf_counter() - start_all
    return latencies_ms, total_seconds, outputs


def run_benchmark(model_path, config=None, inputs=None):
    """Benchmark one model. Returns a dict with all measurements."""
    warmup = int(_cfg(config, "warmup_runs", 10))
    runs = int(_cfg_first(config, ["timed_runs", "num_runs"], 100))
    batch = int(_cfg(config, "batch_size", 1))
    seed = int(_cfg(config, "seed", 0))

    memory_before = memory_mb()
    with PeakMemoryMonitor() as mon:  # also covers model loading, which can be the biggest spike
        session = create_session(model_path, config)
        if inputs is None:
            inputs = make_dummy_inputs(session, batch_size=batch, seed=seed)
        latencies, total_s, outputs = time_session(session, inputs, warmup, runs)
    memory_after = memory_mb()

    result = {
        "model_path": str(model_path),
        "warmup_runs": warmup,
        "num_runs": runs,
        "batch_size": batch,
        **latency_stats(latencies),
        "throughput_per_sec": throughput_per_sec(runs, total_s, batch),
        "memory_mb": memory_after,                 # memory when the run ended
        "memory_delta_mb": memory_after - memory_before,
        "peak_memory_mb": mon.peak_mb,             # highest memory seen during the run
        "peak_memory_delta_mb": mon.delta_mb,      # peak minus where we started
        "environment": get_environment(session),
    }
    return result, outputs, inputs


def compare_baseline_and_optimized(baseline_path, optimized_path, baseline_config=None,
                                   optimized_config=None, rtol=1e-3, atol=1e-5):
    """Benchmark two models on the SAME input and check their outputs match."""
    rtol = _cfg(optimized_config, "rel_tol", rtol)
    atol = _cfg(optimized_config, "abs_tol", atol)
    baseline_result, baseline_out, inputs = run_benchmark(baseline_path, baseline_config)
    optimized_result, optimized_out, _ = run_benchmark(optimized_path, optimized_config, inputs=inputs)

    correctness = compare_outputs(baseline_out, optimized_out, rtol=rtol, atol=atol)
    speedup = baseline_result["mean_ms"] / optimized_result["mean_ms"]
    return {
        "baseline": baseline_result,
        "optimized": optimized_result,
        "speedup": speedup,
        "correctness": correctness,
    }


def get_environment(session=None):
    """Basic info about the machine, so results can be compared fairly."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "providers": session.get_providers() if session else ort.get_available_providers(),
    }


def to_benchmark_result(data, config=None, correctness=None):
    """Turn the runner's dict into the shared BenchmarkResult.

    data         - the dict returned by run_benchmark()
    config       - the BenchmarkConfig that was used (a default one is made if missing)
    correctness  - optional dict from compare_outputs(); fills correct / max_abs_diff
    """
    from inferbench.common.benchmarktypes import BenchmarkConfig, BenchmarkResult

    if config is None:
        config = BenchmarkConfig(model_path=data.get("model_path", ""))

    return BenchmarkResult(
        config=config,
        success=True,
        latency_mean_ms=data["mean_ms"],
        latency_p50_ms=data["p50_ms"],
        latency_p90_ms=data["p90_ms"],
        latency_p95_ms=data["p95_ms"],
        latency_p99_ms=data["p99_ms"],
        latency_min_ms=data["min_ms"],
        latency_max_ms=data["max_ms"],
        throughput_per_sec=data["throughput_per_sec"],
        peak_memory_mb=data["peak_memory_mb"],
        correct=None if correctness is None else correctness["passed"],
        max_abs_diff=None if correctness is None else correctness["max_abs_diff"],
        environment=data.get("environment", {}),
    )


def failed_result(config, error):
    """A BenchmarkResult for a run that crashed (so the optimizer can skip it)."""
    from inferbench.common.benchmarktypes import BenchmarkResult

    return BenchmarkResult(config=config, success=False, error_message=str(error))


@dataclasses.dataclass
class Baseline:
    """The reference run every candidate config is compared against."""
    config: object
    result: object      # BenchmarkResult for the baseline itself
    outputs: list       # baseline output arrays
    inputs: dict        # the exact inputs used (candidates must reuse these)


def make_baseline(config):
    """Run the plain reference setup once: no graph optimisation, 1 thread.

    Do this ONCE per model, then pass the result to evaluate_config() for
    every candidate, so the baseline is not re-run each time.
    """
    base_cfg = dataclasses.replace(config, graph_optimization_level="disable", num_threads=1)
    data, outputs, inputs = run_benchmark(config.model_path, base_cfg)
    result = to_benchmark_result(data, base_cfg, compare_outputs(outputs, outputs))
    return Baseline(config=base_cfg, result=result, outputs=outputs, inputs=inputs)


def evaluate_config(config, baseline=None):
    """Benchmark one config AND check it against the baseline.

    Always returns a BenchmarkResult and never raises:
      - crashed            -> success=False, error_message filled in
      - wrong outputs      -> success=True, correct=False  (is_valid is False)
      - fast and correct   -> success=True, correct=True   (is_valid is True)
    """
    try:
        if baseline is None:
            baseline = make_baseline(config)
        data, outputs, _ = run_benchmark(config.model_path, config, inputs=baseline.inputs)
        check = compare_outputs(
            baseline.outputs, outputs,
            rtol=_cfg(config, "rel_tol", 1e-3), atol=_cfg(config, "abs_tol", 1e-5),
        )
        return to_benchmark_result(data, config, check)
    except Exception as exc:  # a bad config must not stop the whole search
        return failed_result(config, f"{type(exc).__name__}: {exc}")