
import json
import tempfile
from pathlib import Path
from typing import Any

import onnxruntime as ort


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
            "inference_succeeded": False,
            "error": None,
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

                import numpy as np

                feed = {}
                for item in session.get_inputs():
                    shape = [
                        dim if isinstance(dim, int) and dim > 0 else 1
                        for dim in item.shape
                    ]

                    dtype_map = {
                        "tensor(float)": np.float32,
                        "tensor(double)": np.float64,
                        "tensor(float16)": np.float16,
                        "tensor(int8)": np.int8,
                        "tensor(int16)": np.int16,
                        "tensor(int32)": np.int32,
                        "tensor(int64)": np.int64,
                        "tensor(uint8)": np.uint8,
                        "tensor(uint16)": np.uint16,
                        "tensor(uint32)": np.uint32,
                        "tensor(uint64)": np.uint64,
                        "tensor(bool)": np.bool_,
                    }

                    dtype = dtype_map.get(item.type)
                    if dtype is None:
                        raise ValueError(
                            f"Unsupported diagnostic input type "
                            f"{item.type!r} for {item.name!r}"
                        )

                    feed[item.name] = np.zeros(shape, dtype=dtype)

                session.run(None, feed)
                result["inference_succeeded"] = True

                profile_path = session.end_profiling()
                events = json.loads(
                    Path(profile_path).read_text()
                )

                counts = {}
                for event in events:
                    if event.get("cat") != "Node":
                        continue

                    executed_by = event.get("args", {}).get(
                        "provider", "Unknown"
                    )
                    counts[executed_by] = (
                        counts.get(executed_by, 0) + 1
                    )

                result["observed_node_providers"] = counts

        except Exception as exc:
            result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        results.append(result)

    return {
        "model_path": str(path),
        "available_providers": ort.get_available_providers(),
        "results": results,
        "note": (
            "Session creation does not prove full model compatibility. "
            "Observed node providers describe this diagnostic inference "
            "only; they are not a guarantee for every input or workload."
        ),
    }
