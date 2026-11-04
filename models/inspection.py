
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort


def inspect_model_configuration(model_path: str) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Model path is not a file: {path}")

    if path.suffix.lower() != ".onnx":
        raise ValueError(f"Expected an ONNX model: {path}")

    try:
        model = onnx.load(str(path), load_external_data=True)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ValueError(f"Invalid ONNX model: {exc}") from exc

    opset_imports = [
        {
            "domain": item.domain or "ai.onnx",
            "version": item.version,
        }
        for item in model.opset_import
    ]

    operators = sorted({
        f"{node.domain or 'ai.onnx'}::{node.op_type}"
        for node in model.graph.node
    })

    warnings = []

    if not model.graph.input:
        warnings.append("The model declares no graph inputs.")

    if not model.graph.output:
        warnings.append("The model declares no graph outputs.")

    try:
        session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        session_inputs = session.get_inputs()
        session_outputs = session.get_outputs()

        runtime_inputs = [
            {
                "name": item.name,
                "shape": list(item.shape),
                "dtype": item.type,
            }
            for item in session_inputs
        ]

        runtime_outputs = [
            {
                "name": item.name,
                "shape": list(item.shape),
                "dtype": item.type,
            }
            for item in session_outputs
        ]

        status = "session_created"

    except Exception as exc:
        runtime_inputs = []
        runtime_outputs = []
        status = "session_creation_failed"
        warnings.append(
            f"CPU session creation failed: {type(exc).__name__}: {exc}"
        )

    return {
        "model_path": str(path),
        "opsets": opset_imports,
        "operators": operators,
        "runtime_status": status,
        "inputs": runtime_inputs,
        "outputs": runtime_outputs,
        "warnings": warnings,
    }
