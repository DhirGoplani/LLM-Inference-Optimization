"""INT8 quantization helpers for InferBench AutoTune."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inferbench.benchmark.correctness import compare_quality
from inferbench.benchmark.runner import create_session

# Half of the samples calibrate the quantizer, the other half judge accuracy.
MIN_SAMPLES = 4

Feed = dict[str, np.ndarray]


def model_input_names(model_path) -> list[str]:
    session = create_session(model_path)
    return [i.name for i in session.get_inputs()]


def load_samples(npz_path, input_names) -> list[Feed]:
    """Load an .npz of N samples into N single-sample feeds (batch dimension kept at 1).

    The .npz needs one array per model input, named exactly like the input and
    shaped (N, ...), e.g. key "input" with shape (40, 3, 224, 224). Float inputs only.
    """
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration samples not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}

    missing = [n for n in input_names if n not in arrays]
    if missing:
        raise ValueError(
            f"Samples file is missing model inputs {missing}; it contains {sorted(arrays)}"
        )

    counts = {len(arrays[n]) for n in input_names}
    if len(counts) != 1:
        raise ValueError("All model inputs must contain the same number of samples")
    count = counts.pop()
    if count < MIN_SAMPLES:
        raise ValueError(f"Need at least {MIN_SAMPLES} samples, found {count}")

    return [
        {
            name: np.ascontiguousarray(arrays[name][i : i + 1]).astype(np.float32)
            for name in input_names
        }
        for i in range(count)
    ]


def split_samples(feeds: list[Feed]) -> tuple[list[Feed], list[Feed]]:
    """Even-indexed samples calibrate the quantizer, odd-indexed ones evaluate accuracy."""
    return feeds[0::2], feeds[1::2]


class _FeedReader:
    """Calibration reader built lazily so onnxruntime.quantization is only imported on demand."""


def _make_reader(feeds):
    from onnxruntime.quantization import CalibrationDataReader

    class FeedReader(CalibrationDataReader):
        def __init__(self, items):
            self._items = list(items)
            self._iter = iter(self._items)

        def get_next(self):
            return next(self._iter, None)

        def rewind(self):
            self._iter = iter(self._items)

    return FeedReader(feeds)


def quantize_model(model_path, out_dir, method, calibration_feeds=None) -> Path:
    """Create an INT8 copy of the model and return its path.

    method="static"  -> QDQ static quantization, needs calibration feeds (best for CNNs).
    method="dynamic" -> weights-only dynamic quantization, no calibration needed.
    """
    from onnxruntime.quantization import (
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(model_path).stem}_int8_{method}.onnx"

    if method == "static":
        if not calibration_feeds:
            raise ValueError("Static quantization needs calibration samples.")
        quantize_static(
            str(model_path),
            str(out_path),
            _make_reader(calibration_feeds),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
    elif method == "dynamic":
        quantize_dynamic(str(model_path), str(out_path), weight_type=QuantType.QUInt8)
    else:
        raise ValueError(f"Unknown quantization method: {method!r}")

    return out_path


def measure_quality(reference_path, candidate_path, eval_feeds, *, min_cosine, min_top1) -> dict:
    """Run FP32 and quantized models on the same real samples and compare them."""
    reference = create_session(reference_path)
    candidate = create_session(candidate_path)

    reference_runs = []
    candidate_runs = []
    for feed in eval_feeds:
        reference_runs.append(reference.run(None, feed))
        candidate_runs.append(candidate.run(None, feed))

    return compare_quality(
        reference_runs, candidate_runs, min_cosine=min_cosine, min_top1=min_top1
    )