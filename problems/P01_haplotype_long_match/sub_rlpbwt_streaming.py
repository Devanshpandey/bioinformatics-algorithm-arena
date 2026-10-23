"""
P01 submission: RUN-LENGTH ENCODED PBWT columns, no panel, no a/d arrays.

This is the first submission that attacks the measured target directly.

The observation that makes it work: during a left-to-right sweep, EVERY access
the algorithm makes to the panel reduces to `alleles[j]` -- the allele of the
haplotype at position j of the current PBWT column. The streaming submission
computes that as `panel[cur_a, k]`, which is why it must keep the panel
(8 bits/cell as uint8). But the PBWT column in a_k order IS the run-length
encoded object the RLPBWT literature stores, so we can hold the runs instead and
decode column k on the fly. The panel is then never needed at query time at all.

Measured run structure on the small tier (run_stats.py): r = 50,700 runs over
9,143,400 cells, i.e. 5.43 runs per column. At one int32 per run boundary that
is 0.177 bits/cell -- versus 8 bits/cell for the uint8 panel the streaming
submission carries, and 64 bits/cell for the dense a+d index of the baseline.

What this does NOT solve: the divergence array. `d` is still recomputed during
the sweep in O(N) per site rather than stored, which is fine for a sweep but is
exactly why a single online query still costs O(N*M). Representing `d` in
o(N*M) space with O(1) access remains the open problem (SPEC.md Finding 6).
"""
from __future__ import annotations

import numpy as np

from pbwt_core import pbwt_step


class RLPBWTIndex:
    """Run-length encoded PBWT columns. Holds no panel and no a/d arrays."""

    __slots__ = ("starts", "offsets", "first", "N", "M")

    def __init__(self, starts, offsets, first, N, M):
        self.starts = starts      # int32, concatenated run-start positions
        self.offsets = offsets    # int64, offsets[k]:offsets[k+1] slices column k
        self.first = first        # uint8, allele of the first run in each column
        self.N, self.M = N, M

    def nbytes(self) -> int:
        return int(self.starts.nbytes + self.offsets.nbytes + self.first.nbytes)

    def n_runs(self) -> int:
        return int(self.starts.size)

    def column(self, k: int) -> np.ndarray:
        """Decode PBWT column k into a dense allele array, O(N)."""
        s = self.starts[self.offsets[k]:self.offsets[k + 1]]
        lengths = np.diff(np.append(s, self.N))
        vals = (int(self.first[k]) ^ (np.arange(s.size) & 1)).astype(np.uint8)
        return np.repeat(vals, lengths)


def build(panel: np.ndarray) -> RLPBWTIndex:
    N, M = panel.shape
    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)
    all_starts, offsets, first = [], [0], []
    for k in range(M):
        alleles = panel[cur_a, k]
        bounds = np.flatnonzero(alleles[1:] != alleles[:-1]) + 1
        starts = np.empty(bounds.size + 1, dtype=np.int32)
        starts[0] = 0
        starts[1:] = bounds
        all_starts.append(starts)
        offsets.append(offsets[-1] + starts.size)
        first.append(alleles[0] if N else 0)
        cur_a, cur_d = pbwt_step(cur_a, cur_d, alleles, k)
    return RLPBWTIndex(
        np.concatenate(all_starts) if all_starts else np.zeros(0, np.int32),
        np.array(offsets, dtype=np.int64),
        np.array(first, dtype=np.uint8), N, M)


def query_batch(index: RLPBWTIndex, Z: np.ndarray, L: int):
    """Identical sweep to sub_streaming_pbwt, but columns come from the runs."""
    N, M = index.N, index.M
    Q = Z.shape[0]
    out = [[] for _ in range(Q)]

    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)
    t = np.zeros(Q, dtype=np.int64)
    d_above = np.zeros(Q, dtype=np.int64)
    d_below = np.zeros(Q, dtype=np.int64)

    for k in range(M + 1):
        if k < M:
            alleles = index.column(k)          # <- the panel is never touched
            pref0 = np.empty(N + 1, dtype=np.int64)
            pref0[0] = 0
            np.cumsum(alleles == 0, out=pref0[1:])
            n_zero = int(pref0[N])
        ad, dd = cur_a, cur_d

        for qi in range(Q):
            c = int(Z[qi, k]) if k < M else -1
            ti = int(t[qi])

            j = ti - 1
            cur = int(d_above[qi])
            while j >= 0 and k - cur >= L:
                if k == M or int(alleles[j]) != c:
                    out[qi].append((int(ad[j]), cur, k))
                nd = int(dd[j])
                if nd > cur:
                    cur = nd
                j -= 1
            j = ti
            cur = int(d_below[qi])
            while j < N and k - cur >= L:
                if k == M or int(alleles[j]) != c:
                    out[qi].append((int(ad[j]), cur, k))
                j += 1
                if j < N:
                    nd = int(dd[j])
                    if nd > cur:
                        cur = nd
            if k == M:
                continue

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


def query(index: RLPBWTIndex, z: np.ndarray, L: int):
    return query_batch(index, z.reshape(1, -1), L)[0]
