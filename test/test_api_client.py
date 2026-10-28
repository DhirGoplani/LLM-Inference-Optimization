"""Tests for the InferBench AutoTune Dashboard API client and discovery utilities."""

from unittest.mock import MagicMock, patch
import pytest
import requests

from inferbench.dashboard.api_client import (
    BenchmarkApiResponse,
    HealthResponse,
    InferBenchApiClient,
    TuneApiResponse,
    compute_latency_speedup,
    compute_throughput_change,
    discover_models,
    format_latency_speedup,
    format_throughput_change,
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


# ---------- AutoTune (run_tune) Client Tests ----------

def test_run_tune_success():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    payload = {"model_path": "models/model.onnx", "metric": "latency_p50_ms"}
    mock_data = {
        "model_path": "models/model.onnx",
        "metric": "latency_p50_ms",
        "baseline": {"latency_p50_ms": 10.0, "is_valid": True},
        "best": {"latency_p50_ms": 5.0, "is_valid": True},
        "speedup": 2.0,
        "candidates_evaluated": 8,
        "candidates_valid": 8,
        "candidates": [],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_data

    with patch("requests.post", return_value=mock_resp) as mock_post:
        resp = client.run_tune(payload)
        assert resp.ok is True
        assert resp.status_code == 200
        assert resp.data == mock_data
        assert resp.data["speedup"] == 2.0
        mock_post.assert_called_once_with("http://127.0.0.1:8000/tune", json=payload, timeout=300.0)


def test_run_tune_client_error_400():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"detail": "Model path does not exist: nonexistent.onnx"}

    with patch("requests.post", return_value=mock_resp):
        resp = client.run_tune({"model_path": "nonexistent.onnx"})
        assert resp.ok is False
        assert resp.status_code == 400
        assert "Model path does not exist" in resp.error


def test_run_tune_validation_error_422():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.json.return_value = {
        "detail": [{"loc": ["body", "model_path"], "msg": "Field required"}]
    }

    with patch("requests.post", return_value=mock_resp):
        resp = client.run_tune({})
        assert resp.ok is False
        assert resp.status_code == 422
        assert "body.model_path: Field required" in resp.error


def test_run_tune_connection_error():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        resp = client.run_tune({"model_path": "model.onnx"})
        assert resp.ok is False
        assert resp.status_code is None
        assert "Could not connect" in resp.error


def test_run_tune_timeout():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Post timeout")):
        resp = client.run_tune({"model_path": "test.onnx"}, timeout=15.0)
        assert resp.ok is False
        assert resp.status_code is None
        assert "timed out after 15.0s" in resp.error


def test_run_tune_backend_failure_response():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    mock_result_data = {
        "model_path": "corrupted.onnx",
        "metric": "latency_p50_ms",
        "baseline": None,
        "best": None,
        "speedup": None,
        "candidates_evaluated": 0,
        "candidates_valid": 0,
        "candidates": [],
        "error_message": "Baseline run failed: ONNXRuntimeError",
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_result_data

    with patch("requests.post", return_value=mock_resp):
        resp = client.run_tune({"model_path": "corrupted.onnx"})
        assert resp.ok is True
        assert resp.data["best"] is None
        assert resp.data["candidates_valid"] == 0
        assert "Baseline run failed" in resp.data["error_message"]


def test_run_tune_throughput_mode_success():
    client = InferBenchApiClient("http://127.0.0.1:8000")
    payload = {"model_path": "models/model.onnx", "metric": "throughput_per_sec"}
    mock_data = {
        "model_path": "models/model.onnx",
        "metric": "throughput_per_sec",
        "baseline": {"latency_p50_ms": 10.0, "throughput_per_sec": 100.0, "is_valid": True},
        "best": {"latency_p50_ms": 8.0, "throughput_per_sec": 125.0, "is_valid": True},
        "speedup": 1.25,
        "candidates_evaluated": 8,
        "candidates_valid": 8,
        "candidates": [],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_data

    with patch("requests.post", return_value=mock_resp) as mock_post:
        resp = client.run_tune(payload)
        assert resp.ok is True
        assert resp.data["metric"] == "throughput_per_sec"
        assert resp.data["best"]["throughput_per_sec"] == 125.0
        assert resp.data["baseline"]["throughput_per_sec"] == 100.0


# ---------- Metric Calculation & Formatting Helper Tests ----------

def test_compute_throughput_change_valid_positive():
    baseline = {"is_valid": True, "throughput_per_sec": 100.0}
    best = {"is_valid": True, "throughput_per_sec": 125.0}
    change = compute_throughput_change(baseline, best)
    assert change == pytest.approx(25.0)

    # Throughput decrease
    best_slower = {"is_valid": True, "throughput_per_sec": 80.0}
    assert compute_throughput_change(baseline, best_slower) == pytest.approx(-20.0)

    # Identical throughput
    best_same = {"is_valid": True, "throughput_per_sec": 100.0}
    assert compute_throughput_change(baseline, best_same) == pytest.approx(0.0)


def test_compute_throughput_change_missing_or_invalid_results():
    valid = {"is_valid": True, "throughput_per_sec": 100.0}
    invalid = {"is_valid": False, "throughput_per_sec": 120.0}
    missing_key = {"is_valid": True}

    assert compute_throughput_change(None, valid) is None
    assert compute_throughput_change(valid, None) is None
    assert compute_throughput_change(invalid, valid) is None
    assert compute_throughput_change(valid, invalid) is None
    assert compute_throughput_change(missing_key, valid) is None
    assert compute_throughput_change(valid, missing_key) is None
    assert compute_throughput_change("not_a_dict", valid) is None


def test_compute_throughput_change_non_finite_or_non_positive():
    valid = {"is_valid": True, "throughput_per_sec": 100.0}

    # Zero throughput
    assert compute_throughput_change({"is_valid": True, "throughput_per_sec": 0.0}, valid) is None
    assert compute_throughput_change(valid, {"is_valid": True, "throughput_per_sec": 0.0}) is None

    # Negative throughput
    assert compute_throughput_change({"is_valid": True, "throughput_per_sec": -10.0}, valid) is None
    assert compute_throughput_change(valid, {"is_valid": True, "throughput_per_sec": -5.0}) is None

    # Non-finite throughput (inf, -inf, nan)
    assert compute_throughput_change({"is_valid": True, "throughput_per_sec": float("inf")}, valid) is None
    assert compute_throughput_change(valid, {"is_valid": True, "throughput_per_sec": float("inf")}) is None
    assert compute_throughput_change({"is_valid": True, "throughput_per_sec": float("-inf")}, valid) is None
    assert compute_throughput_change({"is_valid": True, "throughput_per_sec": float("nan")}, valid) is None
    assert compute_throughput_change(valid, {"is_valid": True, "throughput_per_sec": float("nan")}) is None

    # Non-numeric throughput
    assert compute_throughput_change({"is_valid": True, "throughput_per_sec": "fast"}, valid) is None


def test_compute_latency_speedup_valid_and_invalid():
    baseline = {"is_valid": True, "latency_p50_ms": 10.0}
    best = {"is_valid": True, "latency_p50_ms": 5.0}

    assert compute_latency_speedup(baseline, best) == 2.0

    # Missing / invalid
    assert compute_latency_speedup(None, best) is None
    assert compute_latency_speedup(baseline, None) is None
    assert compute_latency_speedup({"is_valid": False, "latency_p50_ms": 10.0}, best) is None
    assert compute_latency_speedup(baseline, {"is_valid": False, "latency_p50_ms": 5.0}) is None

    # Non-positive or non-finite
    assert compute_latency_speedup({"is_valid": True, "latency_p50_ms": 0.0}, best) is None
    assert compute_latency_speedup(baseline, {"is_valid": True, "latency_p50_ms": 0.0}) is None
    assert compute_latency_speedup({"is_valid": True, "latency_p50_ms": -2.0}, best) is None
    assert compute_latency_speedup({"is_valid": True, "latency_p50_ms": float("inf")}, best) is None
    assert compute_latency_speedup(baseline, {"is_valid": True, "latency_p50_ms": float("nan")}) is None
    assert compute_latency_speedup({"is_valid": True, "latency_p50_ms": "fast"}, best) is None


def test_format_metric_helpers():
    # Throughput format
    assert format_throughput_change(25.43) == "+25.4%"
    assert format_throughput_change(-12.87) == "-12.9%"
    assert format_throughput_change(0.0) == "+0.0%"
    assert format_throughput_change(None) == "N/A"
    assert format_throughput_change(float("inf")) == "N/A"
    assert format_throughput_change(float("nan")) == "N/A"
    assert format_throughput_change("invalid") == "N/A"

    # Latency speedup format
    assert format_latency_speedup(1.5) == "1.50x"
    assert format_latency_speedup(2.345) == "2.35x"
    assert format_latency_speedup(None) == "N/A"
    assert format_latency_speedup(float("inf")) == "N/A"
    assert format_latency_speedup(float("nan")) == "N/A"
    assert format_latency_speedup("invalid") == "N/A"
