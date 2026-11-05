"""Record where and how a benchmark ran, so the numbers can be reproduced."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def model_files(model_path: str | Path) -> list[Path]:
    """The ONNX graph and any external-data files for a model file or directory."""
    path = Path(model_path)
    if path.is_file():
        folder, graphs = path.parent, [path]
    else:
        folder, graphs = path, sorted(path.glob("*.onnx"))
    files = list(graphs)
    for extra in sorted(folder.glob("*.onnx.data")) + sorted(folder.glob("*.onnx_data")):
        if extra not in files:
            files.append(extra)
    return [f for f in files if f.is_file()]


def model_fingerprint(model_path: str | Path, compute_hash: bool = True) -> dict[str, Any]:
    """Name, size and SHA-256 of every model file, plus one combined hash."""
    files = model_files(model_path)
    entries = []
    for file in files:
        entries.append({
            "file": file.name,
            "size_bytes": file.stat().st_size,
            "sha256": file_sha256(file) if compute_hash else None,
        })
    combined = None
    if compute_hash and entries:
        combined = hashlib.sha256(
            "".join(f"{e['file']}:{e['sha256']}" for e in entries).encode()
        ).hexdigest()
    folder = Path(model_path)
    return {
        "name": folder.stem if folder.is_file() else folder.name,
        "path": str(model_path),
        "files": entries,
        "sha256": combined,
    }


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(repo_dir: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=5
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def collect_environment(
    model_path: str | Path | None = None,
    seed: int = 0,
    benchmark_config: dict[str, Any] | None = None,
    hash_model: bool = True,
    provider: str = "CPUExecutionProvider",
) -> dict[str, Any]:
    """Everything needed to reproduce a benchmark run."""
    try:
        import onnxruntime as ort

        available_providers = list(ort.get_available_providers())
    except ImportError:
        available_providers = []
    memory = psutil.virtual_memory()
    environment: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "provider": provider,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "hardware": {
            "logical_cpus": os.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "total_ram_gb": round(memory.total / (1024 ** 3), 2),
            "gpu_used": False if provider == "CPUExecutionProvider" else None,
        },
        "runtime": {
            "onnxruntime": _version("onnxruntime"),
            "available_providers": available_providers,
            "numpy": _version("numpy"),
            "transformers": _version("transformers"),
            "onnx": _version("onnx"),
        },
        "git": _git_state(Path(__file__).resolve().parents[2]),
        "benchmark_config": benchmark_config or {},
    }
    if model_path is not None:
        environment["model"] = model_fingerprint(model_path, compute_hash=hash_model)
    return environment