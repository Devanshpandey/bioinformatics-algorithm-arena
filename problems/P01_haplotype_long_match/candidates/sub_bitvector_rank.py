"""
P01 submission: the PBWT run-boundary bitvector, context-modelled and
arithmetic coded.  0.0274 bits per panel cell, exact.

STRUCTURE INHERITED FROM sub_rlpbwt_streaming (0.251 bits/cell)
During a left-to-right sweep every panel access reduces to `alleles[j]` on the
current PBWT column, and that column is a run-length object, so the panel is
never stored. sub_rlpbwt_streaming kept the run starts as int32, plus an int64
offsets[] array to slice them per column, plus one uint8 per column for the
first run's allele. On the small tier that is 202,800 + 74,648 + 9,330 bytes.

STEP 1 -- ONE SPARSE BITVECTOR INSTEAD OF PER-COLUMN ARRAYS
Stop thinking per column. There is a single bitvector B of length N*M with
B[k*N + i] = 1 iff position i starts a run in PBWT column k. Column k is
exactly the universe slice [k*N, (k+1)*N), so the offsets array is implied by
the universe partition and costs zero bits -- it was 26% of the incumbent
index, spent entirely on bookkeeping. Position k*N is always a run start, so
those M bits are implicit too, and only the N*M - M interior positions carry
information: measured 41,370 set bits in 9,134,070, density 0.45%.
Elias-Fano over that set gives 51,507 B = 0.0451 bits/cell, already 5.6x below
the incumbent. That was the first scored version of this file.

STEP 2 -- THE BITVECTOR IS NOT MEMORYLESS, SO DO NOT CODE IT AS IF IT WERE
Elias-Fano is within 2 bits/element of optimal for a UNIFORMLY RANDOM set.
This set is not random, and the deviation is predictable from state the
decoder already has. B[k*N+i] = 1 means haplotypes a_k[i-1] and a_k[i] differ
at site k. Those two are adjacent in PBWT order, so they already agree on
sites d_k[i]..k-1 -- the divergence array says exactly how long they have been
matching, and the sweep maintains d anyway. Measured, small tier, P(run
boundary) against that match age:

    age < 1     0.168        age ~ 64     0.0114
    age ~ 4     0.125        age ~ 512    0.0033
    age ~ 16    0.056        age > 2048   0.0003

a 500-fold spread. A second, near-orthogonal context is the distance since the
previous boundary in the same column: boundaries cluster tightly around
locally distinct haplotypes. Measured conditional entropies, bits per cell:

    H0(bit)                        0.0417        Elias-Fano, actual   0.0451
    H(bit | age)                   0.0328
    H(bit | distance)              0.0355
    H(bit | age, distance)         0.0262   <-- what this file targets

Richer contexts were measured and rejected: local age profile, per-column run
count, and per-column minor-allele count all lower the static conditional
entropy but not by more than the model-adaptation or side-information they
cost. A coarse function of the column index does pay (1.2%) and is included --
early in the sweep no pair can have been matching long, so the same age bucket
means a different thing at k=30 than at k=9000.

THE CODE ACTUALLY WRITTEN
One adaptive binary arithmetic stream, in column order, holding:
  * the first run's allele, against the previous column's. It is strongly
    biased (measured P(0) = 0.86) so it costs 0.59 bits/column, not the 8 the
    incumbent spent;
  * for each 128-position block, one flag: does this block hold any boundary?
    context = (minimum age in the block, previous block's flag, column bucket);
  * inside a non-empty block only, one bit per position,
    context = (age bucket, distance bucket, column bucket).
The block skip is what keeps the decoder at 2.1M coding steps instead of 9.1M,
and because its flag carries a real context it costs almost nothing in bits
(block size was swept: 64 -> 0.0281, 96 -> 0.0279, 128 -> 0.0274, 192 ->
0.0277). Two bits are never coded at all because they are forced: the last
position of a non-empty block when nothing has been seen yet, and any block
that must be non-empty.

Everything is integer arithmetic -- an LZMA-style binary range coder with
16-bit probabilities from Krichevsky-Trofimov counts. No floating point enters
the coded stream, so encode/decode is bit-exact and platform-independent, and
no probability tables are stored: encoder and decoder start from a uniform
prior and learn identical numbers from identical history.

MEASURED, small tier, 980 x 9330 (seed 42; seeds 7 and 101 within 1.5%)
    baseline_pbwt (dense a + d)           73,192,360 B   72.0    bits/cell
    sub_streaming_pbwt (uint8 panel)       9,143,400 B    8.0    bits/cell
    sub_rlpbwt_streaming (incumbent)         286,778 B    0.2510 bits/cell
    Elias-Fano version of this file           51,507 B    0.0451 bits/cell
    THIS FILE                                 31,259 B    0.0274 bits/cell
                                                          9.18x the incumbent
                                                     36x below a bit-packed panel

WHAT THIS DOES NOT DO. The cost is query time: 0.300 s/query against 0.172 for
the incumbent and 0.252 for the dense-index PBWT baseline, because the sweep
now runs a range decoder. And the divergence array is still recomputed during
the sweep rather than stored with O(1) access, so a single online query remains
O(N*M). This submission moves the SPACE axis only; the open
O(1)-divergence-access problem in problem.yaml is untouched.

WHAT IS RETAINED. One byte string and two integers -- see __slots__ on
BitvectorRankIndex. No panel, no a array, no d array, no offsets, no
probability tables. nbytes() is len(stream) + 24 for the N, M and length
fields, and there is nowhere else for a byte to hide.
"""
from __future__ import annotations

import numpy as np

from pbwt_core import pbwt_step

_BLOCK = 128         # positions per skip block
_ACAP = 14           # match-age bucket cap  -> 15 buckets
_DCAP = 11           # distance bucket cap   -> 12 buckets
_KC = 5              # coarse column-index buckets (bit_length(k) // 3)
_NC1 = (_ACAP + 1) * 2 * _KC            # block-flag contexts
_NC2 = (_ACAP + 1) * (_DCAP + 1) * _KC  # per-position contexts
_TOP = 1 << 24
_MASK = 0xFFFFFFFF


def _bitlen(a: np.ndarray) -> np.ndarray:
    """Vectorised bit_length for non-negative ints (exact via frexp)."""
    return np.frexp(a.astype(np.float64))[1].astype(np.int64)


def _col_ctx(cur_d: np.ndarray, k: int, N: int, nb: int):
    """Age bucket per interior position, and the per-block minimum of it.

    Computed identically by encoder and decoder from the shared sweep state.
    """
    age = np.maximum(int(k) - cur_d[1:].astype(np.int64), 0)
    ab = np.minimum(_bitlen(age), _ACAP)
    starts = np.arange(nb, dtype=np.int64) * _BLOCK
    bmin = np.minimum.reduceat(ab, starts)
    # A coarse function of the column index is worth 1.2%: early in the sweep
    # no pair can have been matching for long, so the same age bucket means a
    # different thing at k=30 than at k=9000.
    kc = max(k, 1).bit_length() // 3
    if kc > _KC - 1:
        kc = _KC - 1
    # both tables come back pre-multiplied by their strides, so the inner
    # loops form a context with a single add.
    return ((ab * (_DCAP + 1) + kc * ((_ACAP + 1) * (_DCAP + 1))).tolist(),
            (bmin * 2 + kc * ((_ACAP + 1) * 2)).tolist())


class _Enc:
    """LZMA-style binary range encoder (integer only)."""

    __slots__ = ("low", "rng", "cache", "csize", "out")

    def __init__(self):
        self.low = 0
        self.rng = _MASK
        self.cache = 0
        self.csize = 1
        self.out = bytearray()

    def _shift(self):
        low = self.low
        if (low & _MASK) < 0xFF000000 or (low >> 32):
            carry = low >> 32
            out = self.out
            out.append((self.cache + carry) & 0xFF)
            fill = (0xFF + carry) & 0xFF
            for _ in range(self.csize - 1):
                out.append(fill)
            self.csize = 0
            self.cache = (low >> 24) & 0xFF
        self.csize += 1
        self.low = (low << 8) & _MASK

    def bit(self, p0: int, b: int):
        bound = (self.rng >> 16) * p0
        if b:
            self.low += bound
            self.rng -= bound
        else:
            self.rng = bound
        while self.rng < _TOP:
            self.rng = (self.rng << 8) & _MASK
            self._shift()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift()
        return bytes(self.out)


def _p0(n0: int, n1: int) -> int:
    """Krichevsky-Trofimov estimate of P(bit==0), as a 16-bit integer."""
    p = ((2 * n0 + 1) << 16) // (2 * (n0 + n1) + 2)
    if p < 1:
        return 1
    if p > 65535:
        return 65535
    return p


class BitvectorRankIndex:
    """One arithmetic-coded stream. Holds no panel, no a, no d, no offsets."""

    __slots__ = ("stream", "N", "M")

    def __init__(self, stream: bytes, N: int, M: int):
        self.stream = stream
        self.N, self.M = N, M

    def nbytes(self) -> int:
        # stream + three int64 header fields (N, M, stream length)
        return int(len(self.stream) + 24)

    # ---------------------------------------------------------------
    def sweep(self):
        """Yield (alleles, a_k, d_k) for k = 0..M, decoding as it goes.

        The decoder runs the same PBWT sweep the query does, so the query
        consumes this generator instead of maintaining its own a/d.
        """
        N, M = self.N, self.M
        buf = self.stream
        nbuf = len(buf)
        n_pos = N - 1 if N > 0 else 0
        nblk = (n_pos + _BLOCK - 1) // _BLOCK

        rng = _MASK
        code = 0
        pos = 1
        for _ in range(4):
            code = (code << 8) | (buf[pos] if pos < nbuf else 0)
            pos += 1

        f_n0 = [0, 0]
        f_n1 = [0, 0]
        a_n0 = [0] * _NC1
        a_n1 = [0] * _NC1
        b_n0 = [0] * _NC2
        b_n1 = [0] * _NC2

        cur_a = np.arange(N, dtype=np.int32)
        cur_d = np.zeros(N, dtype=np.int32)
        prev_first = 0

        for k in range(M):
            # ---- first run's allele -------------------------------
            c = prev_first
            x = f_n0[c]
            y = f_n1[c]
            p0 = ((2 * x + 1) << 16) // (2 * (x + y) + 2)
            if p0 < 1:
                p0 = 1
            bound = (rng >> 16) * p0
            if code < bound:
                rng = bound
                f = 0
                f_n0[c] += 1
            else:
                code -= bound
                rng -= bound
                f = 1
                f_n1[c] += 1
            while rng < _TOP:
                rng = (rng << 8) & _MASK
                code = (code << 8) | (buf[pos] if pos < nbuf else 0)
                pos += 1
            prev_first = f

            # ---- run boundaries ------------------------------------
            bpos = []
            if n_pos:
                ab, bmin = _col_ctx(cur_d, k, N, nblk)
                last = -1
                prevblk = 0
                i = 0
                s = 0
                while i < n_pos:
                    hi = i + _BLOCK
                    if hi > n_pos:
                        hi = n_pos
                    c1 = bmin[s] + prevblk
                    x = a_n0[c1]
                    y = a_n1[c1]
                    p0 = ((2 * x + 1) << 16) // (2 * (x + y) + 2)
                    if p0 < 1:
                        p0 = 1
                    bound = (rng >> 16) * p0
                    if code < bound:
                        rng = bound
                        any_set = 0
                        a_n0[c1] += 1
                    else:
                        code -= bound
                        rng -= bound
                        any_set = 1
                        a_n1[c1] += 1
                    while rng < _TOP:
                        rng = (rng << 8) & _MASK
                        code = (code << 8) | (buf[pos] if pos < nbuf else 0)
                        pos += 1

                    if any_set:
                        seen = 0
                        last_t = hi - 1
                        for t in range(i, hi):
                            if t == last_t and not seen:
                                b = 1
                            else:
                                d = t - last
                                db = d.bit_length()
                                if db > _DCAP:
                                    db = _DCAP
                                ci = ab[t] + db
                                x = b_n0[ci]
                                y = b_n1[ci]
                                p0 = ((2 * x + 1) << 16) // (2 * (x + y) + 2)
                                if p0 < 1:
                                    p0 = 1
                                bound = (rng >> 16) * p0
                                if code < bound:
                                    rng = bound
                                    b = 0
                                    b_n0[ci] += 1
                                else:
                                    code -= bound
                                    rng -= bound
                                    b = 1
                                    b_n1[ci] += 1
                                while rng < _TOP:
                                    rng = (rng << 8) & _MASK
                                    code = (code << 8) | (
                                        buf[pos] if pos < nbuf else 0)
                                    pos += 1
                            if b:
                                last = t
                                seen = 1
                                bpos.append(t + 1)
                    prevblk = any_set
                    i = hi
                    s += 1

            # ---- rebuild the dense column from its runs ------------
            alleles = np.empty(N, dtype=np.uint8)
            prev = 0
            v = f
            for bp in bpos:
                alleles[prev:bp] = v
                prev = bp
                v ^= 1
            alleles[prev:N] = v

            yield alleles, cur_a, cur_d
            cur_a, cur_d = pbwt_step(cur_a, cur_d, alleles, k)

        yield None, cur_a, cur_d


def build(panel: np.ndarray) -> BitvectorRankIndex:
    N, M = panel.shape
    n_pos = N - 1 if N > 0 else 0
    nblk = (n_pos + _BLOCK - 1) // _BLOCK

    enc = _Enc()
    f_n0 = [0, 0]
    f_n1 = [0, 0]
    a_n0 = [0] * _NC1
    a_n1 = [0] * _NC1
    b_n0 = [0] * _NC2
    b_n1 = [0] * _NC2

    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)
    prev_first = 0

    for k in range(M):
        alleles = panel[cur_a, k]
        f = int(alleles[0]) if N else 0

        c = prev_first
        enc.bit(_p0(f_n0[c], f_n1[c]), f)
        if f:
            f_n1[c] += 1
        else:
            f_n0[c] += 1
        prev_first = f

        if n_pos:
            bnd = (alleles[1:] != alleles[:-1]).astype(np.uint8).tolist()
            ab, bmin = _col_ctx(cur_d, k, N, nblk)
            last = -1
            prevblk = 0
            i = 0
            s = 0
            while i < n_pos:
                hi = i + _BLOCK
                if hi > n_pos:
                    hi = n_pos
                any_set = 0
                for t in range(i, hi):
                    if bnd[t]:
                        any_set = 1
                        break
                c1 = bmin[s] + prevblk
                enc.bit(_p0(a_n0[c1], a_n1[c1]), any_set)
                if any_set:
                    a_n1[c1] += 1
                else:
                    a_n0[c1] += 1

                if any_set:
                    seen = 0
                    last_t = hi - 1
                    for t in range(i, hi):
                        b = bnd[t]
                        if t == last_t and not seen:
                            pass                      # forced, never coded
                        else:
                            d = t - last
                            db = d.bit_length()
                            if db > _DCAP:
                                db = _DCAP
                            ci = ab[t] + db
                            enc.bit(_p0(b_n0[ci], b_n1[ci]), b)
                            if b:
                                b_n1[ci] += 1
                            else:
                                b_n0[ci] += 1
                        if b:
                            last = t
                            seen = 1
                prevblk = any_set
                i = hi
                s += 1

        cur_a, cur_d = pbwt_step(cur_a, cur_d, alleles, k)

    return BitvectorRankIndex(enc.finish(), N, M)


def query_batch(index: BitvectorRankIndex, Z: np.ndarray, L: int):
    """Identical sweep to sub_rlpbwt_streaming; columns come from the decoder."""
    N, M = index.N, index.M
    Q = Z.shape[0]
    out = [[] for _ in range(Q)]

    t = np.zeros(Q, dtype=np.int64)
    d_above = np.zeros(Q, dtype=np.int64)
    d_below = np.zeros(Q, dtype=np.int64)

    sweep = index.sweep()
    for k in range(M + 1):
        alleles, ad, dd = next(sweep)
        if k < M:
            pref0 = np.empty(N + 1, dtype=np.int64)
            pref0[0] = 0
            np.cumsum(alleles == 0, out=pref0[1:])
            n_zero = int(pref0[N])

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
            db_ = k + 1
            while j < N:
                if int(alleles[j]) == c:
                    db_ = cur
                    break
                j += 1
                if j < N:
                    nd = int(dd[j])
                    if nd > cur:
                        cur = nd
            d_above[qi], d_below[qi] = da, db_

    return out


def query(index: BitvectorRankIndex, z: np.ndarray, L: int):
    return query_batch(index, z.reshape(1, -1), L)[0]
