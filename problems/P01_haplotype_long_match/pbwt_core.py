"""Shared vectorised PBWT column update, used by several P01 submissions."""
from __future__ import annotations

import numpy as np


def pbwt_step(cur_a, cur_d, alleles, k):
    """One PBWT column update, vectorised.

    alleles = panel[cur_a, k]. Returns (next_a, next_d).

    The divergence update is a segmented running maximum with reset, which
    np.maximum.reduceat computes in one pass -- this is the whole trick that
    turns Durbin's O(N) per-column Python loop into O(N) numpy work.
    """
    zero_mask = alleles == 0
    z_idx = np.flatnonzero(zero_mask)
    o_idx = np.flatnonzero(~zero_mask)
    next_a = np.concatenate((cur_a[z_idx], cur_a[o_idx]))

    def seg_max(idx):
        if idx.size == 0:
            return np.empty(0, dtype=np.int32)
        starts = np.empty(idx.size, dtype=np.int64)
        starts[0] = 0
        starts[1:] = idx[:-1] + 1
        # reduceat's LAST segment runs to the end of the array, which would
        # over-extend past idx[-1]; truncating the input ends it correctly.
        m = np.maximum.reduceat(cur_d[: idx[-1] + 1], starts).astype(np.int32)
        m[0] = max(int(m[0]), k + 1)  # the p/q running max starts at k+1
        return m

    next_d = np.concatenate((seg_max(z_idx), seg_max(o_idx)))
    return next_a, next_d
