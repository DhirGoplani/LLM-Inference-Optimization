
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


def load_llm_model(
    model_dir_or_path: str | Path,
    *,
    model_file_name: str | None = None,
) -> ModelSpec:
    """Load an LLM ONNX model and enrich it with configuration and architecture metadata.

    When pointing to a directory:
    - If exactly one .onnx file exists, it is loaded.
    - If multiple .onnx files exist, model_file_name must be explicitly provided to
      prevent silently selecting an arbitrary model.
    """
    from models.metadata import load_llm_metadata

    target = Path(model_dir_or_path).expanduser().resolve()

    if target.is_file():
        model_path = target
        model_dir = target.parent
    elif target.is_dir():
        model_dir = target
        if model_file_name is not None:
            if not model_file_name.lower().endswith(".onnx"):
                raise ValueError(
                    f"model_file_name must end with .onnx, got '{model_file_name}'"
                )
            resolved_file = (model_dir / model_file_name).resolve()
            if not resolved_file.is_relative_to(model_dir):
                raise ValueError(
                    f"model_file_name '{model_file_name}' attempts path traversal outside {model_dir}"
                )
            if not resolved_file.is_file():
                raise FileNotFoundError(
                    f"Specified model file '{model_file_name}' not found in {model_dir}"
                )
            model_path = resolved_file
        else:
            onnx_files = sorted(model_dir.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"No .onnx model files found in directory: {model_dir}")
            if len(onnx_files) > 1:
                names = [f.name for f in onnx_files]
                raise ValueError(
                    f"Multiple ONNX models found in {model_dir}: {names}. "
                    f"Specify model_file_name explicitly (e.g. model_file_name='{names[0]}')."
                )
            model_path = onnx_files[0]
    else:
        raise FileNotFoundError(f"Model path does not exist: {target}")

    spec = load_model(str(model_path))

    if (model_dir / "config.json").is_file():
        try:
            meta = load_llm_metadata(model_dir)
            spec.metadata.update(meta)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LLM metadata from {model_dir}: {exc}"
            ) from exc

    return spec
