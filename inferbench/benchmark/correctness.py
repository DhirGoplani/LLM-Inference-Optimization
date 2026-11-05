"""Check that the optimized model gives (almost) the same answers as the baseline."""

import numpy as np


def compare_generated_tokens(cached, uncached):
    """Compare greedy token sequences produced with and without KV-cache."""
    cached_ids = tuple(cached.generated_token_ids if hasattr(cached, "generated_token_ids") else cached)
    uncached_ids = tuple(uncached.generated_token_ids if hasattr(uncached, "generated_token_ids") else uncached)
    return {
        "passed": cached_ids == uncached_ids,
        "cached_tokens": cached_ids,
        "uncached_tokens": uncached_ids,
        "error": None if cached_ids == uncached_ids else "Generated token sequences differ",
    }


def validate_generation_request(prompt, max_new_tokens):
    """Validate prompt and generation length before creating an ORT session run."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")


def compare_outputs(baseline, optimized, rtol=1e-3, atol=1e-5):
    """Compare two lists of output arrays.

    Two numbers a (baseline) and b (optimized) are "close enough" when:
        |a - b| <= atol + rtol * |b|

    Returns a dict with:
        passed        - True only if every output is close enough
        max_abs_diff  - biggest absolute difference found
        max_rel_diff  - biggest relative difference found
        details       - one entry per output with its own result
    """
    if len(baseline) != len(optimized):
        return {
            "passed": False,
            "max_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
            "details": [{"error": f"Output count differs: {len(baseline)} vs {len(optimized)}"}],
        }

    details = []
    all_passed = True
    max_abs = 0.0
    max_rel = 0.0

    for i, (a, b) in enumerate(zip(baseline, optimized)):
        a = np.asarray(a)
        b = np.asarray(b)

        if a.shape != b.shape:
            details.append({"index": i, "passed": False,
                            "error": f"Shape differs: {a.shape} vs {b.shape}"})
            all_passed = False
            max_abs = max_rel = float("inf")
            continue

        a64 = a.astype(np.float64)
        b64 = b.astype(np.float64)

        # NaN in one model but not the other is always a failure
        if not np.array_equal(np.isnan(a64), np.isnan(b64)):
            details.append({"index": i, "passed": False, "error": "NaN values do not match"})
            all_passed = False
            max_abs = max_rel = float("inf")
            continue

        diff = np.abs(a64 - b64)
        diff = np.nan_to_num(diff)  # matching NaNs count as zero difference
        abs_diff = float(diff.max()) if diff.size else 0.0
        rel_diff = float((diff / (np.abs(b64) + 1e-12)).max()) if diff.size else 0.0
        ok = bool(np.allclose(a64, b64, rtol=rtol, atol=atol, equal_nan=True))

        details.append({"index": i, "passed": ok,
                        "max_abs_diff": abs_diff, "max_rel_diff": rel_diff})
        all_passed = all_passed and ok
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)

    return {
        "passed": all_passed,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "details": details,
    }


def compare_quality(reference_runs, candidate_runs, min_cosine=0.99, min_top1=0.90):
    """Judge a quantized model by how close it stays to the FP32 model.

    reference_runs / candidate_runs: one entry per evaluation sample, each entry
    being the list of output arrays from session.run(). Only the FIRST output is
    used, and it is treated as class scores for the top-1 check.

    A model passes when its worst cosine similarity is >= min_cosine AND the
    fraction of samples with the same top-1 prediction is >= min_top1.
    """
    n = len(reference_runs)
    if n == 0 or n != len(candidate_runs):
        return {
            "passed": False,
            "max_abs_diff": float("inf"),
            "error": "Reference and candidate runs are empty or differ in length",
        }

    cosines = []
    agree = 0
    max_abs = 0.0
    for ref, cand in zip(reference_runs, candidate_runs):
        a = np.asarray(ref[0], dtype=np.float64).ravel()
        b = np.asarray(cand[0], dtype=np.float64).ravel()

        if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
            cosines.append(0.0)
            max_abs = float("inf")
            continue

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            cosines.append(1.0 if not a.any() and not b.any() else 0.0)
        else:
            cosines.append(float(np.dot(a, b) / denom))
        agree += int(np.argmax(a) == np.argmax(b))
        max_abs = max(max_abs, float(np.max(np.abs(a - b))))

    cosine_min = min(cosines)
    top1 = agree / n
    return {
        "passed": bool(cosine_min >= min_cosine and top1 >= min_top1),
        "max_abs_diff": max_abs,
        "cosine_min": cosine_min,
        "cosine_mean": float(np.mean(cosines)),
        "top1_agreement": top1,
        "n_eval_samples": n,
        "min_cosine": min_cosine,
        "min_top1": min_top1,
    }