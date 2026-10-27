"""Simple helpers that turn raw timings into numbers we can report."""

import os

import numpy as np
import psutil


def latency_stats(latencies_ms):
    """Take a list of per-run times (in milliseconds) and summarise them."""
    arr = np.asarray(latencies_ms, dtype=float)
    if arr.size == 0:
        raise ValueError("No latencies were recorded.")
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),  # the median run
        "p95_ms": float(np.percentile(arr, 95)),  # 95% of runs were faster than this
        "p99_ms": float(np.percentile(arr, 99)),  # 99% of runs were faster than this
    }


def throughput_per_sec(num_runs, total_seconds, batch_size=1):
    """How many samples are processed per second."""
    if total_seconds <= 0:
        return 0.0
    return float(num_runs * batch_size / total_seconds)


def memory_mb():
    """Memory used by this Python process right now, in MB."""
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))