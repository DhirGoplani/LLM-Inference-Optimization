import numpy as np
import pytest

from inferbench.benchmark.correctness import compare_quality
from inferbench.optimizer import OptimizationContext, get_tuning_registry
from inferbench.optimizer.quantization import MIN_SAMPLES, load_samples, split_samples


def _runs(rows):
    return [[np.asarray(row, dtype=np.float32)] for row in rows]


def test_compare_quality_identical_passes():
    runs = _runs([[0.1, 0.9], [0.8, 0.2]])
    result = compare_quality(runs, runs)
    assert result["passed"]
    assert result["top1_agreement"] == 1.0


def test_compare_quality_flipped_predictions_fail():
    reference = _runs([[0.1, 0.9], [0.8, 0.2]])
    candidate = _runs([[0.9, 0.1], [0.2, 0.8]])
    assert not compare_quality(reference, candidate)["passed"]


def test_quantization_is_opt_in_and_proposes_both_methods():
    assert "quantization" not in get_tuning_registry().names()

    registry = get_tuning_registry(include_quantization=True)
    candidates, skipped = registry.generate_candidates(
        OptimizationContext(), strategy_names=("quantization",)
    )
    assert {c.config["method"] for c in candidates} == {"static", "dynamic"}
    assert all(c.requires_artifact_transform for c in candidates)
    assert not skipped


def test_load_samples_validates_and_splits(tmp_path):
    path = tmp_path / "samples.npz"
    np.savez(path, input=np.zeros((MIN_SAMPLES, 3, 4, 4), dtype=np.float32))

    feeds = load_samples(path, ["input"])
    assert len(feeds) == MIN_SAMPLES
    assert feeds[0]["input"].shape == (1, 3, 4, 4)

    calibration, evaluation = split_samples(feeds)
    assert len(calibration) + len(evaluation) == len(feeds)

    with pytest.raises(ValueError):
        load_samples(path, ["wrong_name"])