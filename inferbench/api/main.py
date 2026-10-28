"""FastAPI application for InferBench AutoTune."""

from pathlib import Path
from typing import Any, Dict, Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from inferbench.benchmark.export import result_to_dict
from inferbench.benchmark.runner import evaluate_config, failed_result
from inferbench.common.benchmarktypes import BenchmarkConfig
from inferbench.optimizer import run_autotune

app = FastAPI(
    title="InferBench AutoTune API",
    version="0.1.0",
)


class BenchmarkRequest(BaseModel):
    model_path: str = Field(..., min_length=1, description="Path to the ONNX model file")
    execution_provider: str = Field(
        default="CPUExecutionProvider",
        description="ONNX execution provider (e.g. CPUExecutionProvider)",
    )
    graph_optimization_level: str = Field(
        default="all",
        description="Graph optimization level (disable, basic, extended, all)",
    )
    num_threads: int = Field(
        default=0,
        ge=0,
        description="Intra-op thread count (0 = auto)",
    )
    batch_size: int = Field(
        default=1,
        gt=0,
        description="Batch size for benchmark inputs",
    )
    warmup_runs: int = Field(
        default=10,
        ge=0,
        description="Number of warmup runs",
    )
    timed_runs: int = Field(
        default=100,
        gt=0,
        description="Number of timed benchmark runs",
    )
    rel_tol: float = Field(
        default=1e-3,
        ge=0.0,
        description="Relative tolerance for output correctness",
    )
    abs_tol: float = Field(
        default=1e-5,
        ge=0.0,
        description="Absolute tolerance for output correctness",
    )
    seed: int = Field(
        default=0,
        description="Seed for dummy input generation",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "InferBench AutoTune API"}


@app.post("/benchmark")
def run_benchmark_endpoint(request: BenchmarkRequest) -> Dict[str, Any]:
    cleaned_path = request.model_path.strip()
    if not cleaned_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model path cannot be empty",
        )

    model_file = Path(cleaned_path)
    if not model_file.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model path does not exist: {cleaned_path}",
        )
    if not model_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model path is not a file: {cleaned_path}",
        )

    config = BenchmarkConfig(
        model_path=cleaned_path,
        execution_provider=request.execution_provider,
        graph_optimization_level=request.graph_optimization_level,
        num_threads=request.num_threads,
        batch_size=request.batch_size,
        warmup_runs=request.warmup_runs,
        timed_runs=request.timed_runs,
        rel_tol=request.rel_tol,
        abs_tol=request.abs_tol,
        seed=request.seed,
    )

    try:
        result = evaluate_config(config)
    except Exception as exc:
        result = failed_result(config, f"Benchmark execution failed: {type(exc).__name__}: {exc}")

    return result_to_dict(result)


class TuneRequest(BaseModel):
    model_path: str = Field(..., min_length=1, description="Path to the ONNX model file")
    execution_provider: str = Field(
        default="CPUExecutionProvider",
        description="ONNX execution provider (e.g. CPUExecutionProvider)",
    )
    batch_size: int = Field(
        default=1,
        gt=0,
        description="Batch size for benchmark inputs",
    )
    warmup_runs: int = Field(
        default=5,
        ge=0,
        description="Number of warmup runs per candidate",
    )
    timed_runs: int = Field(
        default=20,
        gt=0,
        description="Number of timed benchmark runs per candidate",
    )
    rel_tol: float = Field(
        default=1e-3,
        ge=0.0,
        description="Relative tolerance for output correctness",
    )
    abs_tol: float = Field(
        default=1e-5,
        ge=0.0,
        description="Absolute tolerance for output correctness",
    )
    seed: int = Field(
        default=0,
        description="Seed for dummy input generation",
    )
    metric: Literal["latency_p50_ms", "throughput_per_sec"] = Field(
        default="latency_p50_ms",
        description="Target metric to optimize: latency_p50_ms or throughput_per_sec",
    )


@app.post(
    "/tune",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Model path is empty, does not exist, or is not a file",
        },
    },
)
def run_tune_endpoint(request: TuneRequest) -> Dict[str, Any]:
    """Execute AutoTune candidate exploration and select the best valid configuration.

    Returns HTTP 400 when the model path is missing, does not exist, or is not a regular file.
    """
    cleaned_path = request.model_path.strip()
    if not cleaned_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model path cannot be empty",
        )

    model_file = Path(cleaned_path)
    if not model_file.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model path does not exist: {cleaned_path}",
        )
    if not model_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model path is not a file: {cleaned_path}",
        )

    base_config = BenchmarkConfig(
        model_path=cleaned_path,
        execution_provider=request.execution_provider,
        graph_optimization_level="all",
        num_threads=0,
        batch_size=request.batch_size,
        warmup_runs=request.warmup_runs,
        timed_runs=request.timed_runs,
        rel_tol=request.rel_tol,
        abs_tol=request.abs_tol,
        seed=request.seed,
    )

    try:
        tune_result = run_autotune(base_config, metric=request.metric)
    except Exception as exc:
        tune_result = {
            "model_path": cleaned_path,
            "metric": request.metric,
            "baseline": None,
            "best": None,
            "speedup": None,
            "candidates_evaluated": 0,
            "candidates_valid": 0,
            "candidates": [],
            "error_message": f"AutoTune optimization failed: {type(exc).__name__}: {exc}",
        }

    return tune_result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inferbench.api.main:app", host="127.0.0.1", port=8000, reload=True)
