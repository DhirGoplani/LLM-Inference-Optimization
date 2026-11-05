"""Check that the optimized model gives (almost) the same answers as the baseline.

The first half of this file covers generic tensor / classification checks. The
second half (``token_agreement`` onwards) covers LLM generation quality: token
agreement, logits agreement, EOS behaviour, length limits, NaN/Inf and
determinism.
"""

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


# ---------------------------------------------------------------------------
# LLM generation quality checks
# ---------------------------------------------------------------------------


def _ids(value):
    """Token ids from a GenerationResult-like object or a plain sequence."""
    if hasattr(value, "generated_token_ids"):
        return tuple(int(t) for t in value.generated_token_ids)
    return tuple(int(t) for t in value)


def token_agreement(reference, candidate):
    """How closely two greedy token sequences agree.

    agreement is the fraction of positions (out of the longer sequence) holding
    the same token. Greedy decoding diverges for good after the first different
    token, so ``common_prefix_len`` and ``first_divergence`` are also reported.
    """
    ref, cand = _ids(reference), _ids(candidate)
    longest = max(len(ref), len(cand))
    prefix = 0
    for a, b in zip(ref, cand):
        if a != b:
            break
        prefix += 1
    matches = sum(1 for a, b in zip(ref, cand) if a == b)
    exact = ref == cand
    return {
        "exact": exact,
        "agreement": 1.0 if longest == 0 else matches / longest,
        "common_prefix_len": prefix,
        "first_divergence": None if exact else prefix,
        "reference_length": len(ref),
        "candidate_length": len(cand),
    }


def check_eos(generated_ids, eos_token_id, max_new_tokens):
    """Generation must stop at EOS (the EOS token is the last one) or at max_new_tokens.

    Fails when tokens follow an EOS, or when the output is shorter than the
    limit without ending in EOS (it stopped for no reason).
    """
    ids = _ids(generated_ids)
    if eos_token_id is None:
        return {"passed": True, "skipped": True, "error": None, "stopped_by_eos": False}
    error = None
    if max_new_tokens > 0 and not ids:
        error = "no tokens were generated"
    elif eos_token_id in ids and ids.index(eos_token_id) != len(ids) - 1:
        error = "tokens were generated after the EOS token"
    elif len(ids) < max_new_tokens and ids and ids[-1] != eos_token_id:
        error = "stopped before max_new_tokens without producing EOS"
    return {
        "passed": error is None,
        "skipped": False,
        "error": error,
        "stopped_by_eos": bool(ids) and ids[-1] == eos_token_id,
    }


def check_length_constraints(
    generated_ids, max_new_tokens, prompt_token_count=None, max_sequence_length=None
):
    """Output must not exceed max_new_tokens, nor prompt + output the context window."""
    ids = _ids(generated_ids)
    error = None
    if len(ids) > max_new_tokens:
        error = f"generated {len(ids)} tokens, limit is {max_new_tokens}"
    elif (
        max_sequence_length is not None
        and prompt_token_count is not None
        and prompt_token_count + len(ids) > max_sequence_length
    ):
        error = (
            f"prompt ({prompt_token_count}) + output ({len(ids)}) exceeds "
            f"max_sequence_length ({max_sequence_length})"
        )
    return {"passed": error is None, "error": error, "generated_length": len(ids)}


def check_finite(arrays):
    """Fail if any floating-point array contains NaN or Inf."""
    if isinstance(arrays, np.ndarray) or not hasattr(arrays, "__iter__"):
        arrays = [arrays]
    nan_count = inf_count = checked = 0
    for array in arrays:
        arr = np.asarray(array)
        if arr.dtype.kind not in "fc":
            continue
        checked += 1
        nan_count += int(np.isnan(arr).sum())
        inf_count += int(np.isinf(arr).sum())
    return {
        "passed": nan_count == 0 and inf_count == 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "arrays_checked": checked,
    }


def compare_logits(reference, candidate, rel_tol=0.05, min_top1=1.0):
    """Compare logits where both runs have them ([steps, vocab] or [vocab]).

    Passes when the argmax agrees often enough (min_top1) and the largest
    absolute difference stays below rel_tol of the reference's largest logit.
    Returns passed=None, available=False when either side has no logits.
    """
    if reference is None or candidate is None:
        return {"available": False, "passed": None}
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        return {
            "available": True,
            "passed": False,
            "error": f"Shape differs: {a.shape} vs {b.shape}",
        }
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {"available": True, "passed": False, "error": "logits contain NaN or Inf"}
    if a.size == 0:
        return {"available": True, "passed": True, "top1_agreement": 1.0, "max_abs_diff": 0.0}
    top1 = float(np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1)))
    max_abs = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(a))) + 1e-12
    return {
        "available": True,
        "passed": bool(top1 >= min_top1 and max_abs / scale <= rel_tol),
        "top1_agreement": top1,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_abs / scale,
        "rel_tol": rel_tol,
    }


def check_deterministic(generate_fn, request, runs=2):
    """Greedy decoding must give identical tokens every time for the same request."""
    if runs < 2:
        raise ValueError("runs must be at least 2")
    sequences = [_ids(generate_fn(request)) for _ in range(runs)]
    distinct = len(set(sequences))
    return {
        "passed": distinct == 1,
        "runs": runs,
        "distinct_outputs": distinct,
        "error": None if distinct == 1 else "greedy output changed between runs",
    }


def evaluate_generation_quality(
    reference_results,
    candidate_results,
    eos_token_id=None,
    max_new_tokens=None,
    max_sequence_length=None,
    min_exact_match_rate=0.9,
):
    """Judge a configuration's generations against a reference configuration.

    reference_results / candidate_results: GenerationResult-like objects, matched
    by ``request_id`` (or by position when ids are missing). Pass
    reference_results=None to run only the intrinsic checks (EOS and length).

    Hard checks (every request must pass): EOS behaviour, length limits, no
    missing requests. Quality check: the share of requests whose tokens match
    the reference exactly must reach ``min_exact_match_rate``. Small numeric
    differences from padding or FP16 can flip a greedy token, so the threshold
    is configurable; both the rate and the mean agreement are reported.
    """
    candidates = list(candidate_results)
    failures = []
    missing = []

    if reference_results is None:
        pairs = [(None, cand) for cand in candidates]
    else:
        references = list(reference_results)
        by_id = {
            getattr(c, "request_id", None): c
            for c in candidates
            if getattr(c, "request_id", None) is not None
        }
        pairs = []
        for index, ref in enumerate(references):
            rid = getattr(ref, "request_id", None)
            if rid is not None:
                cand = by_id.get(rid)
            else:
                cand = candidates[index] if index < len(candidates) else None
            if cand is None:
                missing.append(rid if rid is not None else index)
            else:
                pairs.append((ref, cand))

    agreements, exact_count, eos_failures, length_failures = [], 0, 0, 0
    for ref, cand in pairs:
        rid = getattr(cand, "request_id", None)
        if ref is not None:
            match = token_agreement(ref, cand)
            agreements.append(match["agreement"])
            exact_count += int(match["exact"])
            if not match["exact"] and len(failures) < 20:
                failures.append({
                    "request_id": rid,
                    "reason": "tokens differ from reference",
                    "first_divergence": match["first_divergence"],
                })
        if max_new_tokens is not None:
            eos = check_eos(cand, eos_token_id, max_new_tokens)
            if not eos["passed"]:
                eos_failures += 1
                if len(failures) < 20:
                    failures.append({"request_id": rid, "reason": eos["error"]})
            length = check_length_constraints(
                cand,
                max_new_tokens,
                getattr(cand, "prompt_token_count", None),
                max_sequence_length,
            )
            if not length["passed"]:
                length_failures += 1
                if len(failures) < 20:
                    failures.append({"request_id": rid, "reason": length["error"]})

    n = len(pairs)
    exact_rate = exact_count / n if (n and reference_results is not None) else None
    quality_ok = exact_rate is None or exact_rate >= min_exact_match_rate
    return {
        "passed": bool(
            n > 0 and not missing and eos_failures == 0 and length_failures == 0 and quality_ok
        ),
        "n_compared": n,
        "exact_match_rate": exact_rate,
        "mean_token_agreement": float(np.mean(agreements)) if agreements else None,
        "min_token_agreement": float(np.min(agreements)) if agreements else None,
        "eos_failures": eos_failures,
        "length_failures": length_failures,
        "missing_requests": missing,
        "failures": failures,
        "min_exact_match_rate": min_exact_match_rate,
    }