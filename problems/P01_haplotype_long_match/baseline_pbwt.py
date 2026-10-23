"""
P01 incumbent: PBWT (Durbin 2014) positional prefix + divergence arrays,
with virtual-insertion query reporting all maximal matches of length >= L.

This is the algorithm the arena asks agents to beat. It is implemented against
the SubmissionAPI contract below, which is the only interface the harness uses:

    build(panel)            -> index          [timed, RAM-measured, size-measured]
    query(index, z, L)      -> list of (j, s, e) maximal matches  [timed]

Any submission implementing this contract can be scored against the same
oracle and the same measurement harness. The representation of `index` is
entirely up to the submission -- that is the point of the problem.

Complexity of this baseline: build O(N*M) time and O(N*M) index words; query
O(M + block scanned + output) per query, versus the O(N*M) per query of the
brute-force oracle. Beating it means changing one of those terms.
"""
from __future__ import annotations

import numpy as np


class PBWTIndex:
    __slots__ = ("a", "d", "N", "M", "zeros", "panel")

    def __init__(self, a, d, zeros, panel):
        self.a = a          # (M+1, N) int32 prefix arrays
        self.d = d          # (M+1, N) int32 divergence arrays
        self.zeros = zeros  # (M,) int32 count of 0-alleles per site
        self.panel = panel
        self.N, self.M = panel.shape

    def nbytes(self) -> int:
        """Index size in bytes, excluding the panel itself."""
        return int(self.a.nbytes + self.d.nbytes + self.zeros.nbytes)


def build(panel: np.ndarray) -> PBWTIndex:
    N, M = panel.shape
    a = np.zeros((M + 1, N), dtype=np.int32)
    d = np.zeros((M + 1, N), dtype=np.int32)
    a[0] = np.arange(N, dtype=np.int32)
    zeros = np.zeros(M, dtype=np.int32)

    cur_a = a[0].copy()
    cur_d = d[0].copy()
    for k in range(M):
        col = panel[:, k]
        alleles = col[cur_a]
        zeros[k] = int(np.count_nonzero(alleles == 0))

        u_a, v_a, u_d, v_d = [], [], [], []
        p = q = k + 1
        for i in range(N):
            di = cur_d[i]
            if di > p:
                p = di
            if di > q:
                q = di
            if alleles[i] == 0:
                u_a.append(cur_a[i]); u_d.append(p); p = 0
            else:
                v_a.append(cur_a[i]); v_d.append(q); q = 0
        cur_a = np.array(u_a + v_a, dtype=np.int32)
        cur_d = np.array(u_d + v_d, dtype=np.int32)
        a[k + 1] = cur_a
        d[k + 1] = cur_d
    return PBWTIndex(a, d, zeros, panel)


def query(index: PBWTIndex, z: np.ndarray, L: int):
    """All maximal matches (j, s, e) between z and the panel with e - s >= L."""
    N, M = index.N, index.M
    panel, a, d, zeros = index.panel, index.a, index.d, index.zeros
    out = []

    t = 0             # virtual insertion position of z in a[k]
    d_above = 0       # match start between z and a[k][t-1]
    d_below = 0       # match start between z and a[k][t]

    for k in range(M + 1):
        ak, dk = a[k], d[k]
        c = int(z[k]) if k < M else -1

        # --- report every maximal match ending at k ---
        j = t - 1
        cur = d_above
        while j >= 0 and k - cur >= L:
            hj = int(ak[j])
            if k == M or int(panel[hj, k]) != c:
                out.append((hj, int(cur), k))
            nd = int(dk[j])
            if nd > cur:
                cur = nd
            j -= 1
        j = t
        cur = d_below
        while j < N and k - cur >= L:
            hj = int(ak[j])
            if k == M or int(panel[hj, k]) != c:
                out.append((hj, int(cur), k))
            j += 1
            if j < N:
                nd = int(dk[j])
                if nd > cur:
                    cur = nd
        if k == M:
            break

        # --- advance the virtual insertion to site k+1 ---
        col_k = panel[:, k]
        # new position: rank of z's allele among the first t entries
        below = col_k[ak[:t]]
        n0 = int(np.count_nonzero(below == 0)) if t else 0
        t_new = n0 if c == 0 else int(zeros[k]) + (t - n0)

        # new d_above: walk up to the nearest haplotype carrying allele c
        j = t - 1
        cur = d_above
        da = k + 1
        while j >= 0:
            if int(col_k[ak[j]]) == c:
                da = cur
                break
            nd = int(dk[j])
            if nd > cur:
                cur = nd
            j -= 1

        # new d_below: walk down to the nearest haplotype carrying allele c
        j = t
        cur = d_below
        db = k + 1
        while j < N:
            if int(col_k[ak[j]]) == c:
                db = cur
                break
            j += 1
            if j < N:
                nd = int(dk[j])
                if nd > cur:
                    cur = nd

        t, d_above, d_below = t_new, da, db

    return out


def run_all(panel: np.ndarray, query_haps: np.ndarray, L: int):
    """Convenience: full (i, j, s, e) match set, matching the oracle's format."""
    idx = build(panel)
    res = set()
    for i in range(query_haps.shape[0]):
        for (j, s, e) in query(idx, query_haps[i], L):
            res.add((i, j, s, e))
    return res
