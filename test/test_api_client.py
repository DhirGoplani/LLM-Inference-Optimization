"""Tests for the InferBench AutoTune Dashboard API client and discovery utilities."""

from unittest.mock import MagicMock, patch
import pytest
import requests

from inferbench.dashboard.api_client import (
    BenchmarkApiResponse,
    HealthResponse,
    InferBenchApiClient,
    discover_models,
)


# ---------- Model Discovery Tests ----------

def test_discover_models_finds_onnx_files(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_a.onnx").write_bytes(b"model-a")
    (models_dir / "notes.txt").write_text("not an onnx file")

    sub_dir = models_dir / "subfolder"
    sub_dir.mkdir()
    (sub_dir / "model_b.onnx").write_bytes(b"model-b")

    found = discover_models(str(models_dir))
    assert len(found) == 2
    assert str(models_dir / "model_a.onnx") in found
    assert str(sub_dir / "model_b.onnx") in found
    assert str(models_dir / "notes.txt") not in found


def test_discover_models_nonexistent_or_empty_dir(tmp_path):
    # Nonexistent folder
    assert discover_models(str(tmp_path / "nonexistent")) == []

    # Empty folder
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert discover_models(str(empty_dir)) == []


# ---------- Health Check Tests ----------

def test_check_health_success():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "healthy", "service": "InferBench AutoTune API"}

    with patch("requests.get", return_value=mock_resp) as mock_get:
        health = client.check_health()
        assert health.ok is True
        assert health.status == "healthy"
        assert health.service == "InferBench AutoTune API"
        assert health.status_code == 200
        mock_get.assert_called_once_with("http://127.0.0.1:8000/health", timeout=5.0)


def test_check_health_http_error():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable"

    with patch("requests.get", return_value=mock_resp):
        health = client.check_health()
        assert health.ok is False
        assert health.status_code == 503
        assert "503" in health.error


def test_check_health_connection_error():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        health = client.check_health()
        assert health.ok is False
        assert health.status_code is None
        assert "Could not connect" in health.error


def test_check_health_timeout():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    with patch("requests.get", side_effect=requests.exceptions.Timeout("Timed out")):
        health = client.check_health(timeout=3.0)
        assert health.ok is False
        assert health.status_code is None
        assert "timed out" in health.error.lower()


def test_check_health_invalid_json():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Malformed JSON")

    with patch("requests.get", return_value=mock_resp):
        health = client.check_health()
        assert health.ok is False
        assert "Invalid JSON" in health.error


# ---------- Benchmark Request Tests ----------

def test_run_benchmark_success():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    mock_result_data = {
        "config": {"model_path": "models/model.onnx"},
        "success": True,
        "is_valid": True,
        "correct": True,
        "latency_mean_ms": 12.34,
        "throughput_per_sec": 81.0,
        "peak_memory_mb": 150.0,
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_result_data

    payload = {"model_path": "models/model.onnx", "num_threads": 4}
    with patch("requests.post", return_value=mock_resp) as mock_post:
        resp = client.run_benchmark(payload)
        assert resp.ok is True
        assert resp.status_code == 200
        assert resp.data["latency_mean_ms"] == 12.34
        assert resp.data["is_valid"] is True
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8000/benchmark", json=payload, timeout=120.0
        )


def test_run_benchmark_client_error_400():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"detail": "Model path does not exist: missing.onnx"}

    with patch("requests.post", return_value=mock_resp):
        resp = client.run_benchmark({"model_path": "missing.onnx"})
        assert resp.ok is False
        assert resp.status_code == 400
        assert "Model path does not exist" in resp.error


def test_run_benchmark_validation_error_422():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.json.return_value = {
        "detail": [
            {"loc": ["body", "batch_size"], "msg": "Input should be greater than 0"}
        ]
    }

    with patch("requests.post", return_value=mock_resp):
        resp = client.run_benchmark({"model_path": "test.onnx", "batch_size": 0})
        assert resp.ok is False
        assert resp.status_code == 422
        assert "body.batch_size: Input should be greater than 0" in resp.error


def test_run_benchmark_connection_error():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Refused")):
        resp = client.run_benchmark({"model_path": "test.onnx"})
        assert resp.ok is False
        assert resp.status_code is None
        assert "Could not connect" in resp.error


def test_run_benchmark_timeout():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Post timeout")):
        resp = client.run_benchmark({"model_path": "test.onnx"}, timeout=10.0)
        assert resp.ok is False
        assert resp.status_code is None
        assert "timed out after 10.0s" in resp.error


def test_run_benchmark_backend_failure_response():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    mock_result_data = {
        "config": {"model_path": "models/model.onnx"},
        "success": False,
        "is_valid": False,
        "error_message": "RuntimeError: Incompatible ONNX opset",
        "latency_mean_ms": None,
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_result_data

    with patch("requests.post", return_value=mock_resp):
        resp = client.run_benchmark({"model_path": "models/model.onnx"})
        # HTTP call succeeded (200), but backend returned a failed benchmark result
        assert resp.ok is True
        assert resp.data["success"] is False
        assert resp.data["is_valid"] is False
        assert "Incompatible ONNX opset" in resp.data["error_message"]
