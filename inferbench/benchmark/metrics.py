"""Simple helpers that turn raw timings into numbers we can report."""

import os
import threading

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
        "p90_ms": float(np.percentile(arr, 90)),
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


class PeakMemoryMonitor:
    """Watches this process's memory in a background thread and remembers the highest value.

    Use it as a context manager:
        with PeakMemoryMonitor() as mon:
            ...do the work...
        print(mon.start_mb, mon.peak_mb, mon.delta_mb)

    It checks every `interval` seconds, so a spike shorter than that can be missed.
    """

    def __init__(self, interval=0.01):
        self.interval = interval
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = None
        self.start_mb = 0.0
        self.peak_mb = 0.0

    def _sample(self):
        self.peak_mb = max(self.peak_mb, self._process.memory_info().rss / (1024 ** 2))

    def _watch(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.start_mb = self.peak_mb = self._process.memory_info().rss / (1024 ** 2)
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()  # one last look
        return False

    @property
    def delta_mb(self):
        """How much higher the peak was than where we started."""
        return self.peak_mb - self.start_mb