"""API client for communicating with the InferBench AutoTune FastAPI backend."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_BENCHMARK_TIMEOUT = 120.0
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


class InferBenchApiClient:
    """HTTP client for InferBench AutoTune FastAPI backend."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        benchmark_timeout: float = DEFAULT_BENCHMARK_TIMEOUT,
        health_timeout: float = DEFAULT_HEALTH_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.benchmark_timeout = benchmark_timeout
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
