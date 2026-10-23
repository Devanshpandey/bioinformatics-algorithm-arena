"""
Tier-0 calibration: score the incumbent AGAINST ITSELF.

Per SPEC.md II.3, this is the single highest-value check in the whole build.
Run the identical implementation as both "candidate" and "incumbent", timed by
the real harness with the real interleaved A/B protocol, and demand that the
pipeline report a speedup of 1.00 with a confidence interval straddling 1.

If it does not, the measurement setup is biased and EVERY subsequent number the
arena produces is wrong -- including, especially, the exciting ones. Running
this before any search loop costs minutes and is the cheapest possible way to
discover that your benchmark harness lies.

Implements the mandatory rules from SPEC.md II.3:
  * interleaved I C I C ... (never IIIII CCCCC), randomised starting arm
  * r repeats, decide on the MEDIAN, report min and raw times too
  * bootstrap CI on the ratio median(candidate)/median(incumbent)
  * machine fingerprint recorded with the result
"""
from __future__ import annotations

import random
import time

import numpy as np

from measure import machine_fingerprint


def interleaved_ab(fn_candidate, fn_incumbent, repeats=5, seed=42, warmup=True):
    """Run the two arms interleaved. Returns (cand_times, inc_times)."""
    rng = random.Random(seed)
    if warmup:
        fn_incumbent(); fn_candidate()
    cand, inc = [], []
    for _ in range(repeats):
        order = [("C", fn_candidate, cand), ("I", fn_incumbent, inc)]
        if rng.random() < 0.5:
            order.reverse()
        for _, fn, sink in order:
            t0 = time.perf_counter()
            fn()
            sink.append(time.perf_counter() - t0)
    return cand, inc


def ratio_ci(cand, inc, n_boot=10_000, seed=42):
    """Bootstrap CI on median(candidate)/median(incumbent)."""
    rng = np.random.default_rng(seed)
    c, i = np.asarray(cand), np.asarray(inc)
    point = float(np.median(c) / np.median(i))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = (np.median(rng.choice(c, size=c.size, replace=True)) /
                    np.median(rng.choice(i, size=i.size, replace=True)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def calibrate(fn, label="incumbent-vs-itself", repeats=5, seed=42, verbose=True):
    """The self-calibration check. Returns a result dict; `passed` is the verdict."""
    cand, inc = interleaved_ab(fn, fn, repeats=repeats, seed=seed)
    point, lo, hi = ratio_ci(cand, inc)
    passed = lo <= 1.0 <= hi
    res = dict(label=label, ratio=point, ci_lo=lo, ci_hi=hi, passed=bool(passed),
               cand_times=cand, inc_times=inc, repeats=repeats,
               machine=machine_fingerprint())
    if verbose:
        print(f"\n=== Tier-0 calibration: {label} ===")
        print(f"  candidate median {np.median(cand):.4f}s   "
              f"incumbent median {np.median(inc):.4f}s")
        print(f"  speedup ratio    {point:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"  spread (max/min) {max(cand+inc)/min(cand+inc):.3f}x")
        if passed:
            print("  PASSED — the harness reports no difference where none exists.")
        else:
            print("  FAILED — the CI excludes 1.0. The measurement setup is BIASED;")
            print("           every downstream number is suspect. Do not run a search loop.")
    return res
