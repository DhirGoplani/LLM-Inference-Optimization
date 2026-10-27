"""Focused tests for the InferBench AutoTune FastAPI application."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferbench.api.main import app
from inferbench.common.benchmarktypes import BenchmarkConfig, BenchmarkResult


def call_api(method: str, path: str, json_data=None):
    """Invoke the FastAPI app via ASGI in-memory without network sockets or httpx."""
    body_bytes = json.dumps(json_data).encode("utf-8") if json_data is not None else b""
    headers = []
    if json_data is not None:
        headers.append((b"content-type", b"application/json"))
        headers.append((b"content-length", str(len(body_bytes)).encode("ascii")))

    response_started = {}
    response_body = bytearray()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 80),
    }

    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response_started.update(message)
        elif message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))

    async def _run():
        await app(scope, receive, send)

    asyncio.run(_run())

    status_code = response_started.get("status")
    body_text = response_body.decode("utf-8")
    try:
        data = json.loads(body_text)
    except Exception:
        data = body_text

    return status_code, data


# ---------- /health endpoint tests ----------

def test_health_endpoint():
    status_code, data = call_api("GET", "/health")
    assert status_code == 200
    assert data == {"status": "healthy", "service": "InferBench AutoTune API"}


# ---------- Request validation tests ----------

def test_benchmark_missing_required_fields():
    status_code, data = call_api("POST", "/benchmark", {})
    assert status_code == 422
    assert "detail" in data


def test_benchmark_empty_model_path():
    status_code, data = call_api("POST", "/benchmark", {"model_path": ""})
    assert status_code == 422


def test_benchmark_whitespace_model_path():
    status_code, data = call_api("POST", "/benchmark", {"model_path": "   "})
    assert status_code == 400
    assert "empty" in data["detail"].lower()


def test_benchmark_invalid_field_constraints():
    # batch_size must be > 0
    status_code, data = call_api("POST", "/benchmark", {"model_path": "test.onnx", "batch_size": 0})
    assert status_code == 422

    # num_threads must be >= 0
    status_code, data = call_api("POST", "/benchmark", {"model_path": "test.onnx", "num_threads": -1})
    assert status_code == 422

    # timed_runs must be > 0
    status_code, data = call_api("POST", "/benchmark", {"model_path": "test.onnx", "timed_runs": 0})
    assert status_code == 422

    # warmup_runs must be >= 0
    status_code, data = call_api("POST", "/benchmark", {"model_path": "test.onnx", "warmup_runs": -1})
    assert status_code == 422


# ---------- Missing / invalid model path tests ----------

def test_benchmark_missing_model_file_on_disk():
    status_code, data = call_api("POST", "/benchmark", {"model_path": "nonexistent_model_12345.onnx"})
    assert status_code == 400
    assert "does not exist" in data["detail"]


def test_benchmark_model_path_is_directory(tmp_path):
    dir_path = tmp_path / "models_dir"
    dir_path.mkdir()
    status_code, data = call_api("POST", "/benchmark", {"model_path": str(dir_path)})
    assert status_code == 400
    assert "not a file" in data["detail"]


# ---------- Mocked integration tests ----------

def test_benchmark_success_mocked_integration(tmp_path):
    dummy_model = tmp_path / "model.onnx"
    dummy_model.write_bytes(b"dummy onnx content")

    mock_result = BenchmarkResult(
        config=BenchmarkConfig(model_path=str(dummy_model), num_threads=4, batch_size=2),
        success=True,
        latency_mean_ms=12.5,
        latency_p50_ms=12.1,
        latency_p90_ms=13.0,
        latency_p95_ms=13.5,
        latency_p99_ms=14.0,
        latency_min_ms=11.8,
        latency_max_ms=14.5,
        throughput_per_sec=80.0,
        peak_memory_mb=210.5,
        correct=True,
        max_abs_diff=1.0e-5,
        environment={"os": "Darwin", "cpu": "ARM"},
    )

    with patch("inferbench.api.main.evaluate_config", return_value=mock_result) as mock_eval:
        status_code, data = call_api(
            "POST",
            "/benchmark",
            {
                "model_path": str(dummy_model),
                "num_threads": 4,
                "batch_size": 2,
            },
        )

        assert status_code == 200
        assert mock_eval.called
        called_config = mock_eval.call_args[0][0]
        assert isinstance(called_config, BenchmarkConfig)
        assert called_config.model_path == str(dummy_model)
        assert called_config.num_threads == 4
        assert called_config.batch_size == 2

        # Verify response structure and values
        assert data["success"] is True
        assert data["correct"] is True
        assert data["is_valid"] is True
        assert data["latency_mean_ms"] == 12.5
        assert data["throughput_per_sec"] == 80.0
        assert data["peak_memory_mb"] == 210.5
        assert data["max_abs_diff"] == 1.0e-5
        assert data["config"]["batch_size"] == 2
        assert data["config"]["num_threads"] == 4


# ---------- Correctness & validity reporting tests ----------

def test_benchmark_correctness_failure_reported_as_invalid(tmp_path):
    dummy_model = tmp_path / "model.onnx"
    dummy_model.write_bytes(b"dummy onnx content")

    divergent_result = BenchmarkResult(
        config=BenchmarkConfig(model_path=str(dummy_model)),
        success=True,
        latency_mean_ms=10.0,
        throughput_per_sec=100.0,
        peak_memory_mb=150.0,
        correct=False,
        max_abs_diff=0.08,
        environment={},
    )

    with patch("inferbench.api.main.evaluate_config", return_value=divergent_result):
        status_code, data = call_api("POST", "/benchmark", {"model_path": str(dummy_model)})

        assert status_code == 200
        assert data["success"] is True
        assert data["correct"] is False
        assert data["is_valid"] is False  # Must not be valid if correct is False
        assert data["max_abs_diff"] == 0.08


def test_benchmark_execution_failure_reported_gracefully(tmp_path):
    dummy_model = tmp_path / "model.onnx"
    dummy_model.write_bytes(b"dummy onnx content")

    crashed_result = BenchmarkResult(
        config=BenchmarkConfig(model_path=str(dummy_model)),
        success=False,
        error_message="RuntimeError: Failed to initialize session",
        correct=None,
    )

    with patch("inferbench.api.main.evaluate_config", return_value=crashed_result):
        status_code, data = call_api("POST", "/benchmark", {"model_path": str(dummy_model)})

        assert status_code == 200
        assert data["success"] is False
        assert data["is_valid"] is False
        assert data["error_message"] == "RuntimeError: Failed to initialize session"
        assert data["latency_mean_ms"] is None


def test_benchmark_unexpected_exception_handled_gracefully(tmp_path):
    dummy_model = tmp_path / "model.onnx"
    dummy_model.write_bytes(b"dummy onnx content")

    with patch("inferbench.api.main.evaluate_config", side_effect=RuntimeError("Engine crash")):
        status_code, data = call_api("POST", "/benchmark", {"model_path": str(dummy_model)})

        assert status_code == 200
        assert data["success"] is False
        assert data["is_valid"] is False
        assert "Benchmark execution failed" in data["error_message"]
        # Ensure no traceback leaked
        assert "Traceback" not in json.dumps(data)
