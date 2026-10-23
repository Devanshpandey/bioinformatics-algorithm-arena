"""
P01 submission: VECTORISED PBWT -- the SAME algorithm as the incumbent, with
the column update written in numpy instead of a Python loop.

This submission exists as a control. It contains zero algorithmic novelty:
identical index, identical query path, identical asymptotics, identical work
counters. Any wall-clock improvement it shows is pure implementation effort.

If the arena's scoring rule cannot distinguish this from a real algorithmic
advance, the scoring rule is broken. It is the harness's negative control.
"""
from __future__ import annotations

import numpy as np

from baseline_pbwt import PBWTIndex, query  # noqa: F401  (query is unchanged)
from pbwt_core import pbwt_step


def build(panel: np.ndarray) -> PBWTIndex:
    N, M = panel.shape
    a = np.zeros((M + 1, N), dtype=np.int32)
    d = np.zeros((M + 1, N), dtype=np.int32)
    a[0] = np.arange(N, dtype=np.int32)
    zeros = np.zeros(M, dtype=np.int32)
    cur_a, cur_d = a[0].copy(), d[0].copy()
    for k in range(M):
        alleles = panel[cur_a, k]
        zeros[k] = int(np.count_nonzero(alleles == 0))
        cur_a, cur_d = pbwt_step(cur_a, cur_d, alleles, k)
        a[k + 1] = cur_a
        d[k + 1] = cur_d
    return PBWTIndex(a, d, zeros, panel)
