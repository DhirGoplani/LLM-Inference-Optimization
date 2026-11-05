"""Quality / correctness checks for LLM generation. No model needed."""

from types import SimpleNamespace

import numpy as np
import pytest

from inferbench.benchmark.correctness import (
    check_deterministic,
    check_eos,
    check_finite,
    check_length_constraints,
    compare_logits,
    evaluate_generation_quality,
    token_agreement,
)

EOS = 9


def res(rid, ids, prompt_tokens=3):
    return SimpleNamespace(
        request_id=rid, generated_token_ids=tuple(ids), prompt_token_count=prompt_tokens
    )


def test_token_agreement_exact_and_divergent():
    assert token_agreement([1, 2, 3], [1, 2, 3])["exact"] is True
    out = token_agreement([1, 2, 3, 4], [1, 2, 7, 4])
    assert out["exact"] is False
    assert out["first_divergence"] == 2 and out["common_prefix_len"] == 2
    assert out["agreement"] == pytest.approx(0.75)


def test_token_agreement_different_lengths_and_empty():
    out = token_agreement([1, 2], [1, 2, 3, 4])
    assert out["agreement"] == pytest.approx(0.5) and out["first_divergence"] == 2
    assert token_agreement([], [])["agreement"] == 1.0


def test_eos_must_be_last_token():
    assert check_eos([1, 2, EOS], EOS, 8)["passed"]
    assert check_eos([1, 2, EOS], EOS, 8)["stopped_by_eos"]
    assert not check_eos([1, EOS, 2], EOS, 8)["passed"]


def test_stopping_early_without_eos_fails_but_hitting_the_limit_passes():
    assert not check_eos([1, 2], EOS, 8)["passed"]
    assert check_eos([1, 2, 3, 4], EOS, 4)["passed"]
    assert not check_eos([], EOS, 4)["passed"]
    assert check_eos([], EOS, 0)["passed"]
    assert check_eos([1], None, 4)["skipped"]


def test_length_constraints():
    assert check_length_constraints([1, 2, 3], 3)["passed"]
    assert not check_length_constraints([1, 2, 3, 4], 3)["passed"]
    assert not check_length_constraints([1, 2], 5, prompt_token_count=9, max_sequence_length=10)["passed"]
    assert check_length_constraints([1, 2], 5, prompt_token_count=8, max_sequence_length=10)["passed"]


def test_check_finite_detects_nan_and_inf_and_skips_ints():
    ok = check_finite([np.zeros((2, 2), dtype=np.float32), np.array([1, 2])])
    assert ok["passed"] and ok["arrays_checked"] == 1
    bad = check_finite([np.array([1.0, np.nan, np.inf, -np.inf])])
    assert not bad["passed"] and bad["nan_count"] == 1 and bad["inf_count"] == 2
    assert not check_finite(np.array([np.nan]))["passed"]


def test_compare_logits():
    ref = np.array([[1.0, 5.0, 2.0], [4.0, 1.0, 0.5]])
    assert compare_logits(ref, ref + 0.01)["passed"]
    flipped = np.array([[5.0, 1.0, 2.0], [4.0, 1.0, 0.5]])
    assert not compare_logits(ref, flipped)["passed"]
    assert compare_logits(None, ref) == {"available": False, "passed": None}
    assert not compare_logits(ref, ref[:1])["passed"]
    assert not compare_logits(ref, np.full_like(ref, np.nan))["passed"]


def test_check_deterministic():
    stable = lambda request: [1, 2, 3]
    assert check_deterministic(stable, "x", runs=3)["passed"]
    counter = iter(range(100))
    flaky = lambda request: [next(counter)]
    out = check_deterministic(flaky, "x", runs=2)
    assert not out["passed"] and out["distinct_outputs"] == 2
    with pytest.raises(ValueError):
        check_deterministic(stable, "x", runs=1)


def test_evaluate_quality_passes_when_candidate_matches_reference():
    ref = [res("a", [1, 2, EOS]), res("b", [3, 4, 5, 6])]
    cand = [res("b", [3, 4, 5, 6]), res("a", [1, 2, EOS])]  # order must not matter
    out = evaluate_generation_quality(ref, cand, eos_token_id=EOS, max_new_tokens=4)
    assert out["passed"] and out["exact_match_rate"] == 1.0 and out["n_compared"] == 2


def test_evaluate_quality_threshold_on_exact_match_rate():
    ref = [res(f"r{i}", [1, 2, 3, 4]) for i in range(10)]
    cand = [res(f"r{i}", [1, 2, 3, 4]) for i in range(9)] + [res("r9", [1, 2, 3, 7])]
    assert evaluate_generation_quality(ref, cand, max_new_tokens=4)["passed"]  # 0.9 >= 0.9
    strict = evaluate_generation_quality(ref, cand, max_new_tokens=4, min_exact_match_rate=1.0)
    assert not strict["passed"] and strict["failures"][0]["request_id"] == "r9"


def test_evaluate_quality_hard_failures():
    ref = [res("a", [1, 2, EOS])]
    assert not evaluate_generation_quality(ref, [], max_new_tokens=4)["passed"]  # missing
    missing = evaluate_generation_quality(ref, [res("zzz", [1, 2, EOS])], max_new_tokens=4)
    assert missing["missing_requests"] == ["a"] and not missing["passed"]
    early = evaluate_generation_quality(
        [res("a", [1, 2])], [res("a", [1, 2])], eos_token_id=EOS, max_new_tokens=8
    )
    assert early["eos_failures"] == 1 and not early["passed"]
    long = evaluate_generation_quality(
        [res("a", [1, 2, 3])], [res("a", [1, 2, 3])], max_new_tokens=2
    )
    assert long["length_failures"] == 1 and not long["passed"]


def test_evaluate_quality_without_reference_runs_intrinsic_checks_only():
    out = evaluate_generation_quality(None, [res("a", [1, 2, EOS])], eos_token_id=EOS, max_new_tokens=4)
    assert out["passed"] and out["exact_match_rate"] is None


def test_existing_generic_checks_still_work():
    from inferbench.benchmark.correctness import compare_generated_tokens, compare_outputs

    assert compare_generated_tokens([1, 2], [1, 2])["passed"]
    assert compare_outputs([np.ones(3)], [np.ones(3)])["passed"]