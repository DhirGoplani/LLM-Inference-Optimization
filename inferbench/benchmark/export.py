"""Save benchmark results as JSON files, so other modules (and people) can read them."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def _clean(value):
    """Make a value safe for strict JSON (no numpy types, no inf/NaN)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None  # inf / NaN -> null
    if isinstance(value, Path):
        return str(value)
    return value


def result_to_dict(result):
    """One BenchmarkResult as a plain, JSON-safe dict."""
    data = _clean(result.to_dict())
    data["is_valid"] = bool(result.is_valid)
    return data


def results_to_json(results):
    """One or many BenchmarkResults as a JSON string."""
    if not isinstance(results, (list, tuple)):
        results = [results]
    document = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": [result_to_dict(r) for r in results],
    }
    return json.dumps(document, indent=2, allow_nan=False)


def save_results(results, path=None, results_dir="results"):
    """Write results to a .json file and return its path.

    If no path is given, a name is made from the model name and the time,
    inside results_dir (created if it doesn't exist).
    """
    items = results if isinstance(results, (list, tuple)) else [results]
    if path is None:
        stem = Path(items[0].config.model_path).stem or "model"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(results_dir) / f"{stem}_{stamp}.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results_to_json(items), encoding="utf-8")
    return path


def load_results(path):
    """Read a results file back. Returns the full document (a dict)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))