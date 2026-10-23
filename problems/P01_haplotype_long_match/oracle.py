"""
P01 exact oracle: brute-force enumeration of maximal long matches.

TASK (this is the formal problem the arena scores):
  Given panel X of N haplotypes over M biallelic sites, query haplotypes Z,
  and a length threshold L, report every triple (i, j, s, e) such that
      Z[i][s:e] == X[j][s:e],           (a match)
      e - s >= L,                        (long enough)
      s == 0 or Z[i][s-1] != X[j][s-1],  (left-maximal)
      e == M or Z[i][e]   != X[j][e].    (right-maximal)

The oracle is O(Q*N*M) and is EXACT by construction -- it is the definition,
not an approximation of it. This is what makes P01 a math-grade evaluator:
correctness is a set equality against a computable ground truth, not a
statistical comparison to a curated answer key.
"""
from __future__ import annotations

import numpy as np


def check_biallelic(*arrays) -> None:
    """Assert the problem's stated precondition. Loudly.

    A silently violated precondition is worse than a crash: the PBWT baseline
    kept returning plausible answers on multi-allelic input while being wrong.
    Preconditions that the scored instances can violate must be CHECKED by the
    evaluator, not assumed by it.
    """
    for a in arrays:
        bad = int((a > 1).sum())
        if bad:
            raise ValueError(
                f"P01 precondition violated: panel/query must be biallelic, "
                f"found {bad} entries with allele > 1. A PBWT column update "
                f"splits on allele==0 vs non-zero and will silently corrupt "
                f"divergence arrays on multi-allelic input.")


def long_matches(panel: np.ndarray, query: np.ndarray, L: int):
    """Return the exact set of maximal matches of length >= L."""
    check_biallelic(panel, query)
    out = set()
    M = panel.shape[1]
    for i in range(query.shape[0]):
        z = query[i]
        eq_all = panel == z  # (N, M) boolean
        for j in range(panel.shape[0]):
            eq = eq_all[j]
            s = 0
            while s < M:
                if not eq[s]:
                    s += 1
                    continue
                e = s
                while e < M and eq[e]:
                    e += 1
                if e - s >= L:
                    out.add((i, j, s, e))
                s = e + 1
    return out
