"""
P01 submission: STREAMING PBWT -- an algorithmically distinct point on the
Pareto frontier, included to demonstrate that the harness measures tradeoffs
rather than declaring a single winner.

Idea: the incumbent stores a_k and d_k for every site (8 bytes per panel cell,
64x the bit-packed panel). But the query sweep visits sites left to right in
order, so the arrays never need to be stored at all -- they can be regenerated
on the fly. The O(N) per-site regeneration is then AMORTISED over all Q queries
answered in the same sweep.

Cost profile versus the incumbent:
    index size   : ~0 extra bytes         (vs 8 bytes per panel cell)
    build time   : ~0                     (vs O(N*M))
    query time   : O(N*M/Q + per-query scans) amortised, i.e. it gets cheaper
                   per query as the query batch grows -- a genuinely different
                   scaling law, not a constant-factor improvement.

Note the API consequence: this algorithm only exists if the contract permits
batched queries. A per-query-only API silently forbids an entire algorithm
class. That is a lesson about arena API design, not about PBWT.
"""
from __future__ import annotations

import numpy as np

from pbwt_core import pbwt_step


class StreamingIndex:
    __slots__ = ("panel", "N", "M")

    def __init__(self, panel):
        self.panel = panel
        self.N, self.M = panel.shape

    def nbytes(self) -> int:
        """Bytes retained by this index -- INCLUDING the panel.

        This used to return 0 with the comment "stores nothing beyond the panel
        it was handed". That was the exact accounting dishonesty the arena's
        verification agents are instructed to hunt for, sitting in our own
        baseline: the object holds `self.panel` alive and answers queries from
        it, so those bytes are part of the index whatever we call them.
        Reporting 0 made the streaming submission look free and understated
        every comparison against it by 8 bits/cell.
        """
        return int(self.panel.nbytes)


def build(panel: np.ndarray) -> StreamingIndex:
    return StreamingIndex(panel)


def query_batch(index: StreamingIndex, Z: np.ndarray, L: int):
    """Answer every query in a single left-to-right sweep."""
    panel, N, M = index.panel, index.N, index.M
    Q = Z.shape[0]
    out = [[] for _ in range(Q)]

    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)
    t = np.zeros(Q, dtype=np.int64)
    d_above = np.zeros(Q, dtype=np.int64)
    d_below = np.zeros(Q, dtype=np.int64)

    for k in range(M + 1):
        if k < M:
            col_k = panel[:, k]
            alleles = col_k[cur_a]
            pref0 = np.empty(N + 1, dtype=np.int64)
            pref0[0] = 0
            np.cumsum(alleles == 0, out=pref0[1:])
            n_zero = int(pref0[N])
        ad = cur_a
        dd = cur_d

        for qi in range(Q):
            c = int(Z[qi, k]) if k < M else -1
            ti = int(t[qi])

            # report maximal matches ending at k
            j = ti - 1
            cur = int(d_above[qi])
            while j >= 0 and k - cur >= L:
                hj = int(ad[j])
                if k == M or int(panel[hj, k]) != c:
                    out[qi].append((hj, cur, k))
                nd = int(dd[j])
                if nd > cur:
                    cur = nd
                j -= 1
            j = ti
            cur = int(d_below[qi])
            while j < N and k - cur >= L:
                hj = int(ad[j])
                if k == M or int(panel[hj, k]) != c:
                    out[qi].append((hj, cur, k))
                j += 1
                if j < N:
                    nd = int(dd[j])
                    if nd > cur:
                        cur = nd
            if k == M:
                continue

            # advance virtual insertion
            n0 = int(pref0[ti])
            t[qi] = n0 if c == 0 else n_zero + (ti - n0)

            j = ti - 1
            cur = int(d_above[qi])
            da = k + 1
            while j >= 0:
                if int(alleles[j]) == c:
                    da = cur
                    break
                nd = int(dd[j])
                if nd > cur:
                    cur = nd
                j -= 1
            j = ti
            cur = int(d_below[qi])
            db = k + 1
            while j < N:
                if int(alleles[j]) == c:
                    db = cur
                    break
                j += 1
                if j < N:
                    nd = int(dd[j])
                    if nd > cur:
                        cur = nd
            d_above[qi], d_below[qi] = da, db

        if k < M:
            cur_a, cur_d = pbwt_step(cur_a, cur_d, alleles, k)

    return out


def query(index: StreamingIndex, z: np.ndarray, L: int):
    return query_batch(index, z.reshape(1, -1), L)[0]
