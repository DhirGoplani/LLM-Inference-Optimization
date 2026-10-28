
from pathlib import Path

import onnx
import onnxruntime as ort

from models.schema import ModelSpec


def load_model(model_path: str) -> ModelSpec:
    path = Path(model_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Model path is not a file: {path}")

    if path.suffix.lower() != ".onnx":
        raise ValueError(f"Expected an ONNX model: {path}")

    try:
        onnx.checker.check_model(str(path))
    except Exception as exc:
        raise ValueError(f"Invalid ONNX model: {exc}") from exc

    available_providers = ort.get_available_providers()
    compatible_providers = []

    session = None

    for provider in available_providers:
        try:
            test_session = ort.InferenceSession(
                str(path),
                providers=[provider],
            )
            compatible_providers.append(provider)

            if provider == "CPUExecutionProvider":
                session = test_session

        except Exception:
            continue

    if session is None:
        raise RuntimeError(
            "Could not create a model session using CPUExecutionProvider."
        )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    input_names = [item.name for item in inputs]
    output_names = [item.name for item in outputs]

    input_shapes = {
        item.name: list(item.shape) for item in inputs
    }
    output_shapes = {
        item.name: list(item.shape) for item in outputs
    }

    input_dtypes = {
        item.name: item.type for item in inputs
    }
    output_dtypes = {
        item.name: item.type for item in outputs
    }

    return ModelSpec(
        model_path=str(path),
        input_names=input_names,
        output_names=output_names,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        available_providers=available_providers,
        compatible_providers=compatible_providers,
    )
