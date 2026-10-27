"""Check that the optimized model gives (almost) the same answers as the baseline."""

import numpy as np


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