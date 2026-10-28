"""API client for communicating with the InferBench AutoTune FastAPI backend."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_BENCHMARK_TIMEOUT = 120.0
DEFAULT_TUNE_TIMEOUT = 300.0
DEFAULT_HEALTH_TIMEOUT = 5.0


def discover_models(models_dir: str = "models") -> List[str]:
    """Discover .onnx files in the given directory and its subdirectories.

    Returns a sorted list of relative path strings. Returns an empty list
    if the directory does not exist or contains no .onnx files.
    """
    path = Path(models_dir)
    if not path.exists() or not path.is_dir():
        return []

    found: List[str] = []
    for file_path in path.glob("**/*.onnx"):
        if file_path.is_file():
            found.append(str(file_path))

    return sorted(found)


@dataclass
class HealthResponse:
    """Represents the result of a GET /health check."""
    ok: bool
    status: Optional[str] = None
    service: Optional[str] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


@dataclass
class BenchmarkApiResponse:
    """Represents the HTTP result of a POST /benchmark request."""
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


@dataclass
class TuneApiResponse:
    """Represents the HTTP result of a POST /tune request."""
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


def compute_throughput_change(
    baseline: Optional[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Calculate (best_throughput / baseline_throughput - 1) * 100.

    Returns None if:
    - baseline or best is not a dict
    - baseline or best is not valid (is_valid is not True)
    - throughput_per_sec in baseline or best is not finite or <= 0
    """
    if not isinstance(baseline, dict) or not isinstance(best, dict):
        return None

    if not baseline.get("is_valid") or not best.get("is_valid"):
        return None

    base_tp = baseline.get("throughput_per_sec")
    best_tp = best.get("throughput_per_sec")

    if not isinstance(base_tp, (int, float)) or not isinstance(best_tp, (int, float)):
        return None

    if not math.isfinite(base_tp) or not math.isfinite(best_tp):
        return None

    if base_tp <= 0 or best_tp <= 0:
        return None

    return (best_tp / base_tp - 1.0) * 100.0


def format_throughput_change(change: Optional[float]) -> str:
    """Format throughput change as percentage string, e.g. '+15.2%', or 'N/A'."""
    if change is None or not isinstance(change, (int, float)) or not math.isfinite(change):
        return "N/A"
    return f"{change:+.1f}%"


def compute_latency_speedup(
    baseline: Optional[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Calculate baseline_p50 / best_p50.

    Returns None if:
    - baseline or best is not a dict
    - baseline or best is not valid (is_valid is not True)
    - latency_p50_ms in baseline or best is not finite or <= 0
    """
    if not isinstance(baseline, dict) or not isinstance(best, dict):
        return None

    if not baseline.get("is_valid") or not best.get("is_valid"):
        return None

    base_lat = baseline.get("latency_p50_ms")
    best_lat = best.get("latency_p50_ms")

    if not isinstance(base_lat, (int, float)) or not isinstance(best_lat, (int, float)):
        return None

    if not math.isfinite(base_lat) or not math.isfinite(best_lat):
        return None

    if base_lat <= 0 or best_lat <= 0:
        return None

    return round(base_lat / best_lat, 2)


def format_latency_speedup(speedup: Optional[float]) -> str:
    """Format latency speedup as string, e.g. '1.50x', or 'N/A'."""
    if speedup is None or not isinstance(speedup, (int, float)) or not math.isfinite(speedup):
        return "N/A"
    return f"{speedup:.2f}x"


class InferBenchApiClient:
    """HTTP client for InferBench AutoTune FastAPI backend."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        benchmark_timeout: float = DEFAULT_BENCHMARK_TIMEOUT,
        tune_timeout: float = DEFAULT_TUNE_TIMEOUT,
        health_timeout: float = DEFAULT_HEALTH_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.benchmark_timeout = benchmark_timeout
        self.tune_timeout = tune_timeout
        self.health_timeout = health_timeout

    def check_health(self, timeout: Optional[float] = None) -> HealthResponse:
        """Call GET /health to verify backend connectivity and health."""
        req_timeout = timeout if timeout is not None else self.health_timeout
        url = f"{self.base_url}/health"
        try:
            resp = requests.get(url, timeout=req_timeout)
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                    return HealthResponse(
                        ok=True,
                        status=payload.get("status"),
                        service=payload.get("service"),
                        status_code=resp.status_code,
                    )
                except Exception as exc:
                    return HealthResponse(
                        ok=False,
                        error=f"Invalid JSON response from {url}: {exc}",
                        status_code=resp.status_code,
                    )
            else:
                return HealthResponse(
                    ok=False,
                    error=f"Backend returned HTTP {resp.status_code}: {resp.text.strip()}",
                    status_code=resp.status_code,
                )
        except requests.exceptions.Timeout:
            return HealthResponse(
                ok=False,
                error=f"Connection timed out after {req_timeout}s connecting to {url}.",
            )
        except requests.exceptions.ConnectionError:
            return HealthResponse(
                ok=False,
                error=f"Could not connect to FastAPI server at {self.base_url}. Ensure the server is running.",
            )
        except requests.exceptions.RequestException as exc:
            return HealthResponse(
                ok=False,
                error=f"Network error while reaching {url}: {exc}",
            )

    def run_benchmark(
        self, payload: Dict[str, Any], timeout: Optional[float] = None
    ) -> BenchmarkApiResponse:
        """Send a benchmark request payload to POST /benchmark."""
        req_timeout = timeout if timeout is not None else self.benchmark_timeout
        url = f"{self.base_url}/benchmark"
        try:
            resp = requests.post(url, json=payload, timeout=req_timeout)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    return BenchmarkApiResponse(ok=True, data=data, status_code=resp.status_code)
                except Exception as exc:
                    return BenchmarkApiResponse(
                        ok=False,
                        error=f"Invalid JSON returned by backend: {exc}",
                        status_code=resp.status_code,
                    )
            else:
                # Attempt to extract detailed error message from FastAPI HTTPException or validation error
                error_detail = None
                try:
                    body = resp.json()
                    if isinstance(body, dict):
                        detail = body.get("detail")
                        if isinstance(detail, list):
                            # FastAPI validation error list
                            msgs = [
                                f"{'.'.join(str(k) for k in item.get('loc', []))}: {item.get('msg')}"
                                for item in detail
                            ]
                            error_detail = "; ".join(msgs)
                        elif isinstance(detail, str):
                            error_detail = detail
                        else:
                            error_detail = str(body)
                except Exception:
                    error_detail = resp.text.strip() or f"HTTP status {resp.status_code}"

                return BenchmarkApiResponse(
                    ok=False,
                    error=error_detail or f"HTTP {resp.status_code}",
                    status_code=resp.status_code,
                )
        except requests.exceptions.Timeout:
            return BenchmarkApiResponse(
                ok=False,
                error=f"Benchmark request timed out after {req_timeout}s.",
            )
        except requests.exceptions.ConnectionError:
            return BenchmarkApiResponse(
                ok=False,
                error=f"Could not connect to FastAPI server at {self.base_url}. Ensure the server is running.",
            )
        except requests.exceptions.RequestException as exc:
            return BenchmarkApiResponse(
                ok=False,
                error=f"Request error while connecting to {url}: {exc}",
            )

    def run_tune(
        self, payload: Dict[str, Any], timeout: Optional[float] = None
    ) -> TuneApiResponse:
        """Send an autotuning request payload to POST /tune."""
        req_timeout = timeout if timeout is not None else self.tune_timeout
        url = f"{self.base_url}/tune"
        try:
            resp = requests.post(url, json=payload, timeout=req_timeout)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    return TuneApiResponse(ok=True, data=data, status_code=resp.status_code)
                except Exception as exc:
                    return TuneApiResponse(
                        ok=False,
                        error=f"Invalid JSON returned by backend: {exc}",
                        status_code=resp.status_code,
                    )
            else:
                error_detail = None
                try:
                    body = resp.json()
                    if isinstance(body, dict):
                        detail = body.get("detail")
                        if isinstance(detail, list):
                            msgs = [
                                f"{'.'.join(str(k) for k in item.get('loc', []))}: {item.get('msg')}"
                                for item in detail
                            ]
                            error_detail = "; ".join(msgs)
                        elif isinstance(detail, str):
                            error_detail = detail
                        else:
                            error_detail = str(body)
                except Exception:
                    error_detail = resp.text.strip() or f"HTTP status {resp.status_code}"

                return TuneApiResponse(
                    ok=False,
                    error=error_detail or f"HTTP {resp.status_code}",
                    status_code=resp.status_code,
                )
        except requests.exceptions.Timeout:
            return TuneApiResponse(
                ok=False,
                error=f"AutoTune request timed out after {req_timeout}s.",
            )
        except requests.exceptions.ConnectionError:
            return TuneApiResponse(
                ok=False,
                error=f"Could not connect to FastAPI server at {self.base_url}. Ensure the server is running.",
            )
        except requests.exceptions.RequestException as exc:
            return TuneApiResponse(
                ok=False,
                error=f"Request error while connecting to {url}: {exc}",
            )
