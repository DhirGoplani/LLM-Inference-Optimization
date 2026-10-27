"""Runs an ONNX model many times and measures how fast it is."""

import dataclasses
import platform
import time

import numpy as np
import onnxruntime as ort

from inferbench.benchmark.correctness import compare_outputs
from inferbench.benchmark.metrics import latency_stats, memory_mb, throughput_per_sec

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


def create_session(model_path, config=None):
    """Open an ONNX model with the settings from the config."""
    options = ort.SessionOptions()

    threads = _cfg(config, "num_threads", 0)  # 0 = let ONNX Runtime decide
    options.intra_op_num_threads = int(threads)

    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    level_name = str(_cfg(config, "graph_optimization_level", "all")).lower()
    options.graph_optimization_level = levels.get(level_name, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

    providers = _cfg(config, "providers", ["CPUExecutionProvider"])
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
    runs = int(_cfg(config, "num_runs", 100))
    batch = int(_cfg(config, "batch_size", 1))

    memory_before = memory_mb()
    session = create_session(model_path, config)
    if inputs is None:
        inputs = make_dummy_inputs(session, batch_size=batch)

    latencies, total_s, outputs = time_session(session, inputs, warmup, runs)
    memory_after = memory_mb()

    result = {
        "model_path": str(model_path),
        "warmup_runs": warmup,
        "num_runs": runs,
        "batch_size": batch,
        **latency_stats(latencies),
        "throughput_per_sec": throughput_per_sec(runs, total_s, batch),
        "memory_mb": memory_after,
        "memory_delta_mb": memory_after - memory_before,
        "environment": get_environment(session),
    }
    return result, outputs, inputs


def compare_baseline_and_optimized(baseline_path, optimized_path, baseline_config=None,
                                   optimized_config=None, rtol=1e-3, atol=1e-5):
    """Benchmark two models on the SAME input and check their outputs match."""
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


def to_benchmark_result(data):
    """Convert our dict into the shared BenchmarkResult (keeps only fields it has)."""
    from inferbench.common.benchmarktypes import BenchmarkResult

    names = {f.name for f in dataclasses.fields(BenchmarkResult)}
    kept = {k: v for k, v in data.items() if k in names}
    return BenchmarkResult(**kept)