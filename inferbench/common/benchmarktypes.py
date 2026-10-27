from dataclasses import dataclass, field, asdict
from typing import Any, Optional

@dataclass
class BenchmarkConfig:
    model_path: str
    execution_provider: str = "CPUExecutionProvider"
    execution_providers: Optional[tuple[str, ...]] = None
    graph_optimization_level: str = "all"
    num_threads: int = 0
    inter_op_num_threads: int = 0
    execution_mode: str = "sequential"
    enable_mem_pattern: bool = True
    batch_size: int = 1
    warmup_runs: int = 10
    timed_runs: int = 100
    rel_tol: float = 1e-3
    abs_tol: float = 1e-5
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """What you return after benchmarking one config."""
    config: BenchmarkConfig
    success: bool                           # did the run finish without crashing?
    error_message: Optional[str] = None     # filled in if it crashed

    # Latency in milliseconds
    latency_mean_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p90_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    latency_min_ms: Optional[float] = None
    latency_max_ms: Optional[float] = None

    # Speed and memory
    throughput_per_sec: Optional[float] = None
    peak_memory_mb: Optional[float] = None

    # Correctness (compared with the baseline output)
    correct: Optional[bool] = None
    max_abs_diff: Optional[float] = None    # biggest difference from baseline

    # Environment info (OS, CPU, versions)
    environment: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True only if it ran AND the output is correct."""
        return self.success and self.correct is True

    def to_dict(self) -> dict:
        return asdict(self)