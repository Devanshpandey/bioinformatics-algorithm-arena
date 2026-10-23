"""
P01 submission -- COLUMN-DIFF: divergence-conditioned entropy coding of the
PBWT columns.  The index is one arithmetic-coded bitstream (28,704 B on the
scored panel) plus a 900-byte probability table.  No panel, no prefix array, no divergence array, no run
array is retained.

Measured, small tier, seed 42, panel 980 x 9330 = 9,143,400 cells:

    baseline_pbwt, dense a + d                              64      bits/cell
    bit-packed panel                                         1
    sub_rlpbwt_streaming (int32 run starts + int64 offsets)  0.251
    memoryless run-boundary entropy                          0.0417
    xz -9 on the bit-packed panel                            0.0355
    THIS                                                     0.0259   (9.7x)

Every stage is accounted for: the model's own entropy on this panel is 229,547
bits, the 8-bit probability ladder adds 21 bits (0.01%) and the range coder
adds 64 bits (0.03%), for a 229,632-bit payload.  There is nothing left to win
in the coder; only a better model can move this number.

=============================================================================
What the "column diff" island actually contains
=============================================================================

1.  The literal diff is EMPTY, and that is a theorem, not a measurement.  A run
    boundary at position i of column k says the two adjacent haplotypes DIFFER
    at site k, so the stable partition that builds a_{k+1} sends them to
    opposite blocks: every boundary pair of column k is destroyed by
    construction.  Of the N-1 adjacencies of column k+1, exactly N - r_k are
    inherited -- and all of those were NON-boundaries at k -- while r_k - 1 are
    freshly created at run junctions.  Transporting the previous boundary map
    forward therefore carries no direct information whatsoever.
    Empirical confirmation: adding "was this adjacency just created at the last
    step" to the context model moves the coded size by 0.1%, and conditioning
    on the previous column's run count r_{k-1} makes things WORSE once its
    table is paid for (0.03324 vs 0.03209 in the same harness).

2.  The genuine inter-column redundancy is already accumulated in the
    DIVERGENCE ARRAY.  d_k[i] is the site at which the adjacency (i-1, i) was
    born, so k - d_k[i] is exactly how long this pair has agreed and a boundary
    is the event "they disagree now".  P(boundary) falls from 0.168 at match
    length 0 to 0.0001 at match length 8192.  Crucially d_k is FREE: the query
    is a left-to-right sweep that already maintains it, so the decoder can
    rebuild the exact context of every bit before decoding that bit, storing no
    side information at all.  Match length alone: 0.0334 bits/cell.

3.  The rest of the structure is INTRA-column, and it is bigger than I
    expected.  By the time position i is being decoded, positions 0..i-1 of the
    same column are already known, so the current run length and the length of
    the run before it are free context.  They capture what divergence cannot:
    a singleton allele produces a length-1 run, i.e. two boundaries in a row,
    and short runs cluster.  Adding them takes 0.0334 -> 0.0263.
    (A within-column "boundaries seen vs boundaries expected" rate feature also
    works -- 0.0334 -> 0.0321 -- but run lengths subsume it and cost less
    table, so it is not used.)

Probability tables are stored as 8-bit codes on a log ladder.  Quantisation
error enters the code length only through a KL divergence, which is second
order: a 2% relative error on p costs about 3e-4 bits per event, i.e. tens of
bits over the whole panel, while halving the table.

=============================================================================
Layout
=============================================================================
    payload   arithmetic-coded stream of M*N binary decisions: per column the
              first allele, then N-1 "differs from the haplotype above" bits
    ptab      uint8[15 * 10 * 3 * 2], P(bit=1 | match-length octave,
              run-length-so-far octave, previous-run-length octave,
              was-the-run-before-that-a-singleton)
    N, M
nbytes() counts all of it.  Nothing else survives build().
"""
from __future__ import annotations

import math

import numpy as np

from pbwt_core import pbwt_step

# ------------------------------------------------------------- coder constants
PBITS = 16
TOT = 1 << PBITS
# 64-bit registers.  With a 32-bit range the split point (rng >> PBITS) * q0
# is truncated to a relative precision of only 2^-8 at the renormalisation
# floor, which measured 2,900 bits (1.23%) of pure coder waste on the small
# tier.  At 64 bits the truncation is 2^-40 and the loss disappears.
TOP = 1 << 56
MASKW = (1 << 64) - 1
HIBYTE = 0xFF00000000000000

# ------------------------------------------------------------- model geometry
NML = 14                # octave buckets of the match length k - d_k[i]
CAP = 8192
FIRST_CTX = NML         # the first allele of a column gets its own class
NCLASS = NML + 1
NRL = 10                # octave buckets of the current run length
NPRL = 3                # octave buckets of the previous run length
NPRL2 = 2               # was the run before that a singleton?
NCTX = NCLASS * NRL * NPRL * NPRL2

_MLB = np.empty(CAP + 1, dtype=np.int64)
for _v in range(CAP + 1):
    _MLB[_v] = min(NML - 1, int(math.log2(_v + 1)))

# 8-bit log ladder: fine in p below 1/2 and fine in 1-p above it.
_LADDER = np.empty(256, dtype=np.int64)
for _j in range(128):
    _LADDER[_j] = min(TOT - 1, max(1, int(round(TOT * 2.0 ** (-16.0 + _j * 15.0 / 127.0)))))
for _j in range(128, 256):
    _LADDER[_j] = min(TOT - 1, max(1, int(round(
        TOT * (1.0 - 2.0 ** (-1.0 - (_j - 128) * 15.0 / 127.0))))))
_LADP = _LADDER / float(TOT)
_LCOST1 = -np.log2(_LADP)          # cost of coding a 1 with each ladder entry
_LCOST0 = -np.log2(1.0 - _LADP)


def _rl_tables(n: int):
    """Pre-scaled context contributions of the three run-length features,
    indexed by the raw length 0..n so the decoder loop is three list reads."""
    v = np.arange(n + 1, dtype=np.int64)
    lg = np.zeros(n + 1, dtype=np.int64)
    if n >= 1:
        lg[1:] = np.log2(v[1:]).astype(np.int64)
    rlt = np.minimum(lg, NRL - 1) * (NPRL * NPRL2)
    prlt = np.minimum(lg, NPRL - 1) * NPRL2
    prl2t = np.minimum(lg, NPRL2 - 1)
    return rlt, prlt, prl2t


# --------------------------------------------------------------- range encoder
class _Enc:
    """LZMA-style carry-propagating binary range encoder."""

    __slots__ = ("low", "rng", "cache", "csize", "out")

    def __init__(self):
        self.low = 0
        self.rng = MASKW
        self.cache = 0
        self.csize = 1
        self.out = bytearray()

    def put(self, q0list, bits) -> None:
        low, rng, cache, csize, out = self.low, self.rng, self.cache, self.csize, self.out
        for i, b in enumerate(bits):
            bound = (rng >> PBITS) * q0list[i]
            if b:
                low += bound
                rng -= bound
            else:
                rng = bound
            while rng < TOP:
                rng <<= 8
                if low < HIBYTE or low > MASKW:
                    carry = low >> 64
                    temp = cache
                    while True:
                        out.append((temp + carry) & 0xFF)
                        temp = 0xFF
                        csize -= 1
                        if csize == 0:
                            break
                    cache = (low >> 56) & 0xFF
                csize += 1
                low = (low << 8) & MASKW
        self.low, self.rng, self.cache, self.csize = low, rng, cache, csize

    def finish(self) -> bytes:
        for _ in range(9):
            low, cache, csize = self.low, self.cache, self.csize
            if low < HIBYTE or low > MASKW:
                carry = low >> 64
                temp = cache
                while True:
                    self.out.append((temp + carry) & 0xFF)
                    temp = 0xFF
                    csize -= 1
                    if csize == 0:
                        break
                cache = (low >> 56) & 0xFF
            csize += 1
            self.low = (low << 8) & MASKW
            self.cache, self.csize = cache, csize
        return bytes(self.out)


# --------------------------------------------------------------------- index
class ColumnDiffIndex:
    __slots__ = ("payload", "ptab", "N", "M")

    def __init__(self, payload: bytes, ptab: np.ndarray, N: int, M: int):
        self.payload = payload
        self.ptab = ptab                 # uint8 ladder codes, NCTX entries
        self.N = N
        self.M = M

    def nbytes(self) -> int:
        return int(len(self.payload) + self.ptab.nbytes + 8)


def _mlbase(N: int, k: int, cur_d: np.ndarray) -> np.ndarray:
    """class(i) * NRL * NPRL for every coded position of column k."""
    cls = np.empty(N, dtype=np.int64)
    cls[0] = FIRST_CTX
    if N > 1:
        ml = k - cur_d[1:].astype(np.int64)
        np.clip(ml, 0, CAP, out=ml)
        cls[1:] = _MLB[ml]
    return cls * (NRL * NPRL * NPRL2)


def _run_features(bits: np.ndarray, rlt: np.ndarray, prlt: np.ndarray,
                  prl2t: np.ndarray) -> np.ndarray:
    """Vectorised twin of the decoder's sequential run-length bookkeeping."""
    n = bits.size
    ar = np.arange(n, dtype=np.int64)
    nz = np.flatnonzero(bits)
    tmp = np.full(n, -1, dtype=np.int64)
    if nz.size:
        tmp[nz] = nz
    acc = np.maximum.accumulate(tmp)
    last = np.empty(n, dtype=np.int64)
    last[0] = -1
    if n > 1:
        last[1:] = acc[:-1]
    rl = ar - last
    lmap = np.zeros(n, dtype=np.int64)
    l2map = np.zeros(n, dtype=np.int64)
    if nz.size:
        pnz = np.empty(nz.size, dtype=np.int64)
        pnz[0] = -1
        pnz[1:] = nz[:-1]
        runlen = nz - pnz
        lmap[nz] = runlen
        back = np.zeros(nz.size, dtype=np.int64)
        back[1:] = runlen[:-1]
        l2map[nz] = back
    at = np.maximum(last, 0)
    prev = np.where(last >= 0, lmap[at], 0)
    prev2 = np.where(last >= 0, l2map[at], 0)
    return rlt[rl] + prlt[prev] + prl2t[prev2]


def _fit(idx: np.ndarray, bits: np.ndarray, n: int) -> np.ndarray:
    """Pick, per context, the ladder code that minimises the true code length."""
    n1 = np.bincount(idx[bits == 1], minlength=n).astype(np.float64)
    nt = np.bincount(idx, minlength=n).astype(np.float64)
    n0 = nt - n1
    p = np.where(nt > 0, n1 / np.maximum(nt, 1.0), 0.5)
    j = np.searchsorted(_LADDER, np.clip(p * TOT, 1, TOT - 1))
    np.clip(j, 1, 255, out=j)
    lo, hi = j - 1, j
    cost_lo = n1 * _LCOST1[lo] + n0 * _LCOST0[lo]
    cost_hi = n1 * _LCOST1[hi] + n0 * _LCOST0[hi]
    return np.where(cost_lo <= cost_hi, lo, hi).astype(np.uint8)


# --------------------------------------------------------------------- build
def build(panel: np.ndarray) -> ColumnDiffIndex:
    panel = np.ascontiguousarray(panel, dtype=np.uint8)
    N, M = panel.shape
    if N == 0 or M == 0:
        return ColumnDiffIndex(b"", np.zeros(0, dtype=np.uint8), int(N), int(M))

    rlt, prlt, prl2t = _rl_tables(N)

    # pass A -- replay the sweep, keeping the coded bits and their contexts.
    # Both arrays are scratch; neither survives into the index.
    bit_all = np.empty((M, N), dtype=np.int8)
    idx_all = np.empty((M, N), dtype=np.int16)
    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)
    for k in range(M):
        al = panel[cur_a, k]
        bits = np.empty(N, dtype=np.int64)
        bits[0] = al[0]
        if N > 1:
            bits[1:] = (al[1:] != al[:-1])
        bit_all[k] = bits
        idx_all[k] = _mlbase(N, k, cur_d) + _run_features(bits, rlt, prlt, prl2t)
        cur_a, cur_d = pbwt_step(cur_a, cur_d, al, k)

    flat_idx = idx_all.reshape(-1).astype(np.int64)
    flat_bit = bit_all.reshape(-1).astype(np.int64)
    ptab = _fit(flat_idx, flat_bit, NCTX)

    # pass B -- encode
    q0 = (TOT - _LADDER[ptab.astype(np.int64)])[flat_idx]
    enc = _Enc()
    for k in range(M):
        enc.put(q0[k * N:(k + 1) * N].tolist(), bit_all[k].tolist())
    return ColumnDiffIndex(enc.finish(), ptab, int(N), int(M))


# ------------------------------------------------------------- column decoder
class _ColumnStream:
    """Hand it cur_d for column k, get column k back.  Sequential by design:
    the query is one left-to-right sweep, so no checkpoints are needed."""

    __slots__ = ("buf", "blen", "pos", "rng", "code", "N", "qt",
                 "rlt", "prlt", "prl2t")

    def __init__(self, index: ColumnDiffIndex):
        buf = index.payload
        self.buf = buf
        self.blen = len(buf)
        c = 0
        for i in range(1, 9):
            c = (c << 8) | (buf[i] if i < self.blen else 0)
        self.code = c
        self.pos = 9
        self.rng = MASKW
        self.N = index.N
        self.qt = (TOT - _LADDER[index.ptab.astype(np.int64)]).tolist()
        rlt, prlt, prl2t = _rl_tables(index.N)
        self.rlt = rlt.tolist()
        self.prlt = prlt.tolist()
        self.prl2t = prl2t.tolist()

    def column(self, k: int, cur_d: np.ndarray) -> np.ndarray:
        N = self.N
        base = _mlbase(N, k, cur_d).tolist()
        qt, rlt, prlt, prl2t = self.qt, self.rlt, self.prlt, self.prl2t
        rng, code, pos, buf, blen = self.rng, self.code, self.pos, self.buf, self.blen
        top, pb = TOP, PBITS

        bits = []
        ap = bits.append
        last = -1
        prev = 0
        prev2 = 0
        for i in range(N):
            cur = i - last
            bound = (rng >> pb) * qt[base[i] + rlt[cur] + prlt[prev] + prl2t[prev2]]
            if code < bound:
                rng = bound
                ap(0)
            else:
                code -= bound
                rng -= bound
                ap(1)
                prev2 = prev
                prev = cur
                last = i
            while rng < top:
                rng <<= 8
                code = (code << 8) | (buf[pos] if pos < blen else 0)
                pos += 1
        self.rng, self.code, self.pos = rng, code, pos

        b = np.array(bits, dtype=np.int64)
        al = np.empty(N, dtype=np.uint8)
        al[0] = b[0]
        if N > 1:
            al[1:] = ((b[0] + np.cumsum(b[1:])) & 1).astype(np.uint8)
        return al


# --------------------------------------------------------------------- query
def query_batch(index: ColumnDiffIndex, Z: np.ndarray, L: int):
    """Durbin virtual-insertion sweep; columns come from the entropy stream."""
    N, M = index.N, index.M
    Z = np.asarray(Z, dtype=np.uint8)
    Q = Z.shape[0]
    out = [[] for _ in range(Q)]
    if N == 0:
        return out

    stream = _ColumnStream(index)
    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)
    t = np.zeros(Q, dtype=np.int64)
    d_above = np.zeros(Q, dtype=np.int64)
    d_below = np.zeros(Q, dtype=np.int64)

    alleles = None
    pref0 = None
    n_zero = 0
    for k in range(M + 1):
        if k < M:
            alleles = stream.column(k, cur_d)
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


def query(index: ColumnDiffIndex, z: np.ndarray, L: int):
    return query_batch(index, np.asarray(z).reshape(1, -1), L)[0]
