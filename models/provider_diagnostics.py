import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from models.sample_inputs import (
    ONNX_TO_NUMPY_DTYPES,
    generate_diagnostic_inputs,
)


def _summarize_outputs(
    session: ort.InferenceSession,
    values: list[np.ndarray],
) -> list[dict[str, Any]]:
    expected_outputs = session.get_outputs()
    if len(values) != len(expected_outputs):
        raise RuntimeError(
            f"Inference returned {len(values)} outputs; "
            f"the model declares {len(expected_outputs)}."
        )
    if not values:
        raise RuntimeError("Inference returned no outputs.")

    summaries = []
    for metadata, value in zip(expected_outputs, values):
        if not isinstance(value, np.ndarray):
            raise RuntimeError(
                f"Inference output {metadata.name!r} is not a NumPy tensor."
            )

        expected_shape = metadata.shape
        if len(value.shape) != len(expected_shape):
            raise RuntimeError(
                f"Inference output {metadata.name!r} has rank {value.ndim}; "
                f"the model declares rank {len(expected_shape)}."
            )
        for axis, (actual, expected) in enumerate(
            zip(value.shape, expected_shape)
        ):
            if isinstance(expected, int) and expected > 0 and actual != expected:
                raise RuntimeError(
                    f"Inference output {metadata.name!r} dimension {axis} "
                    f"is {actual}; the model declares {expected}."
                )

        expected_dtype = ONNX_TO_NUMPY_DTYPES.get(metadata.type)
        if expected_dtype is not None and value.dtype != np.dtype(expected_dtype):
            raise RuntimeError(
                f"Inference output {metadata.name!r} has dtype {value.dtype}; "
                f"the model declares {metadata.type}."
            )

        summaries.append({
            "name": metadata.name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        })

    return summaries


def _read_node_provider_counts(profile_path: str) -> dict[str, int]:
    events = json.loads(Path(profile_path).read_text())
    counts: dict[str, int] = {}
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = event.get("args", {}).get("provider")
        if not provider:
            continue
        counts[provider] = counts.get(provider, 0) + 1
    return counts


def _execution_status(
    requested_provider: str,
    observed_node_providers: dict[str, int],
    inference_succeeded: bool,
) -> tuple[str, bool | None]:
    if not inference_succeeded:
        return "inference_failed", None
    if not observed_node_providers:
        return "no_nodes_observed", None

    fallback = any(
        provider != requested_provider
        for provider in observed_node_providers
    )
    if requested_provider not in observed_node_providers:
        return "fallback", True
    if fallback:
        return "partial_fallback", True
    return "full", False


def diagnose_execution_providers(
    model_path: str,
) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Model path is not a file: {path}")

    if path.suffix.lower() != ".onnx":
        raise ValueError(f"Expected an ONNX model: {path}")

    results = []

    for provider in ort.get_available_providers():
        result = {
            "provider": provider,
            "session_created": False,
            "session_providers": [],
            "observed_node_providers": {},
            "actual_node_providers": [],
            "provider_execution_status": "session_creation_failed",
            "provider_fallback": None,
            "inference_succeeded": False,
            "input_shapes": {},
            "input_dtypes": {},
            "outputs": [],
            "error": None,
            "profiling_error": None,
        }

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                options = ort.SessionOptions()
                options.enable_profiling = True
                options.profile_file_prefix = str(
                    Path(temp_dir) / "profile"
                )

                session = ort.InferenceSession(
                    str(path),
                    sess_options=options,
                    providers=[provider],
                )
                result["session_created"] = True
                result["session_providers"] = session.get_providers()

                try:
                    feed = generate_diagnostic_inputs(session.get_inputs())
                    result["input_shapes"] = {
                        name: list(value.shape)
                        for name, value in feed.items()
                    }
                    result["input_dtypes"] = {
                        name: str(value.dtype)
                        for name, value in feed.items()
                    }

                    output_values = session.run(None, feed)
                    result["outputs"] = _summarize_outputs(
                        session,
                        output_values,
                    )
                    result["inference_succeeded"] = True
                except Exception as exc:
                    result["error"] = f"{type(exc).__name__}: {exc}"

                try:
                    profile_path = session.end_profiling()
                    result["observed_node_providers"] = (
                        _read_node_provider_counts(profile_path)
                    )
                    result["actual_node_providers"] = list(
                        result["observed_node_providers"]
                    )
                except Exception as exc:
                    result["profiling_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                (
                    result["provider_execution_status"],
                    result["provider_fallback"],
                ) = _execution_status(
                    provider,
                    result["observed_node_providers"],
                    result["inference_succeeded"],
                )

        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"

        results.append(result)

    return {
        "model_path": str(path),
        "available_providers": ort.get_available_providers(),
        "results": results,
        "note": (
            "Diagnostics use synthetic first-step inputs. Dynamic KV caches "
            "start empty and the attention mask covers past plus current "
            "tokens. Node provider assignments describe this inference only; "
            "a session may use provider fallback."
        ),
    }
