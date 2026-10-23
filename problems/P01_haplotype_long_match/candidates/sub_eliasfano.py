"""
P01 submission: ELIAS-FANO-STYLE CODING OF THE PBWT RUN BOUNDARIES.

    incumbent  sub_rlpbwt_streaming   286,778 B   0.2509 bits/cell
    this file                          ~41,000 B   ~0.036 bits/cell

Where the incumbent's bytes go, on the small tier (980 x 9330, r = 50,700 runs
in the PBWT allele columns):

    starts  50,700 x int32 = 202,800 B
    offsets  9,331 x int64 =  74,648 B    <- 26% of the index, pure bookkeeping
    first    9,330 x uint8 =   9,330 B    <- one byte to carry one bit

Every one of those three is waste, and this submission removes all three.


STEP 1 -- ONE MONOTONE SEQUENCE, NO PER-COLUMN OFFSETS
------------------------------------------------------
The run starts of column k are positions in [0, N).  The first one is always 0
and is never stored, so column k contributes r_k - 1 interior boundaries.
Concatenating the columns with a stride of N,

    g = k * N + b          (b the boundary, 0 < b < N)

gives ONE strictly increasing sequence over the universe U = N*M -- strictly
increasing because within a column the b increase and across columns the k*N
term dominates (b < N).  The column index comes back for free as g // N, so the
9,331-entry offset array disappears completely.  That is the observation that
makes a single flat Elias-Fano-class sequence sufficient for the whole panel.


STEP 2 -- ELIAS-FANO ON THAT SEQUENCE  (measured: 0.04506 bits/cell)
--------------------------------------------------------------------
Classic Elias-Fano splits each value into a high part g >> l and a low part
g & (2^l - 1); the lows are concatenated as fixed l-bit fields and the highs
go into a bitvector setting bit (g_i >> l) + i, which is a unary code of the
successive high-part differences.  Cost n*l + n + U/2^l bits, minimised at
l = log2(U ln2 / n).  Here n = 41,370 and U = 9,143,400, so l = 7 and the
index is 51,499 B = 0.04506 bits/cell.  That version passed the gate and is
already 5.6x below the incumbent.

It is also, measurably, not the end.  Elias-Fano charges EVERY element a floor
of l + 1 = 8 bits, because the low part is l bits of the ABSOLUTE value and is
uniform no matter how close two neighbours are.  The measured gap distribution
is nowhere near uniform: 16% of gaps are exactly 1, 28% are <= 3, the median is
18, and the 90th percentile is 506.  Order-0 entropy of the gaps is 7.60
bits/gap against Elias-Fano's 9.72.


STEP 3 -- KEEP THE SPLIT, ENTROPY-CODE THE HIGH PART  (this submission)
------------------------------------------------------------------------
So keep Elias-Fano's idea -- split each number into a coarse part and a raw
low part -- but apply it to the GAPS rather than the absolute values, and let
the coarse part be entropy-coded rather than unary:

    gap = 2^e + m,   0 <= m < 2^e

    e  ("which bucket")  -> canonical Huffman over the exponent alphabet
    m  ("where inside")  -> e raw bits, packed variable-width

This is the Elias gamma/delta family with a Huffman-shaped length field.  A
gap of 1 now costs the Huffman code for e=0 alone (~2 bits) instead of Elias-
Fano's 8-bit floor, while a gap of 500 costs ~4 + 8.  Measured on the small
tier: exponents 142,197 bits + mantissas 178,366 bits = 7.75 bits/gap, within
2% of the 7.60-bit order-0 entropy of the gap sequence, and 1.25x below plain
Elias-Fano.

Reference points measured on the same instance, all in bits/cell:
    dense a + d index (baseline_pbwt)              64.0
    uint8 panel (sub_streaming_pbwt)                8.0
    bit-packed panel                                1.0
    incumbent RLE, int32 starts                     0.2509
    plain global Elias-Fano (step 2)                0.0450
    xz -9e on the bit-packed PBWT columns           0.0442
    THIS                                           ~0.0360
    order-0 entropy of the gap sequence            ~0.0356
The xz line is the useful one: a general-purpose compressor given the same
PBWT-transformed columns does WORSE than this coder, so the remaining bytes
are gap entropy and not unexploited redundancy.

Two more small wins folded in, both free once the machinery exists:
  * the first-run allele of each column is a bit, not a byte, and it is heavily
    biased (P(1) = 0.14), so instead of M raw bits the POSITIONS of the set
    bits are appended to the very same monotone sequence, shifted past N*M.
    One coder, one stream, ~2,900 bits saved.
  * the exponent alphabet is tiny (12 symbols observed, max Huffman code
    length 5), so the code table costs ~15 bytes and decoding is a single
    table lookup per gap.

Decoding is done once per query_batch into LOCAL arrays that are dropped when
the sweep ends.  Nothing is cached on the index object, so .nbytes() below is
the complete retained footprint: four packed bit arrays plus six python ints.

What this does NOT solve, unchanged from the incumbent: the divergence array is
recomputed during the sweep rather than stored with O(1) access, so a single
online query is still O(N*M).  This is a space result only.
"""
from __future__ import annotations

import heapq

import numpy as np

from pbwt_core import pbwt_step

# 2^0 .. 2^61; every gap in this problem is below N*M <= 2^61.
_POW2 = (np.int64(1) << np.arange(62, dtype=np.int64))
_MAXLEN = 16          # Huffman codes are length-limited so one table decodes all


# --------------------------------------------------------------------------
# variable-width bit field packing
# --------------------------------------------------------------------------

def _offsets(widths: np.ndarray) -> np.ndarray:
    off = np.zeros(widths.size, dtype=np.int64)
    if widths.size > 1:
        np.cumsum(widths[:-1], out=off[1:])
    return off


def _pack_var(vals: np.ndarray, widths: np.ndarray, total: int) -> np.ndarray:
    """Concatenate vals[i] as a big-endian field of widths[i] bits."""
    if total <= 0:
        return np.zeros(0, dtype=np.uint8)
    off = _offsets(widths)
    bits = np.zeros(total, dtype=np.uint8)
    for j in range(int(widths.max())):
        m = widths > j
        bits[off[m] + j] = ((vals[m] >> (widths[m] - 1 - j)) & 1).astype(np.uint8)
    return np.packbits(bits)


def _unpack_var(packed: np.ndarray, widths: np.ndarray, total: int) -> np.ndarray:
    """Inverse of _pack_var (widths must be known to the caller)."""
    out = np.zeros(widths.size, dtype=np.int64)
    if total <= 0 or widths.size == 0:
        return out
    bits = np.unpackbits(packed)[:total].astype(np.int64)
    off = _offsets(widths)
    for j in range(int(widths.max())):
        m = widths > j
        out[m] = (out[m] << 1) | bits[off[m] + j]
    return out


# --------------------------------------------------------------------------
# canonical, length-limited Huffman over a tiny integer alphabet
# --------------------------------------------------------------------------

def _huffman_lengths(counts: np.ndarray) -> np.ndarray:
    """Code lengths per symbol; 0 for unused.  Falls back to a fixed-width
    code if the Huffman tree is deeper than _MAXLEN, so the decoder never
    needs a second code path."""
    lengths = np.zeros(counts.size, dtype=np.int64)
    used = np.flatnonzero(counts > 0)
    if used.size == 0:
        return lengths
    if used.size == 1:
        lengths[used[0]] = 1
        return lengths
    heap = [(int(counts[s]), i, [int(s)]) for i, s in enumerate(used)]
    heapq.heapify(heap)
    tie = len(heap)
    while len(heap) > 1:
        w1, _, g1 = heapq.heappop(heap)
        w2, _, g2 = heapq.heappop(heap)
        for s in g1:
            lengths[s] += 1
        for s in g2:
            lengths[s] += 1
        heapq.heappush(heap, (w1 + w2, tie, g1 + g2))
        tie += 1
    if int(lengths.max()) > _MAXLEN:
        lengths[:] = 0
        lengths[used] = max(1, int(used.size - 1).bit_length())
    return lengths


def _canonical(lengths: np.ndarray):
    """Canonical code words for the given lengths (symbols ordered by
    (length, symbol), which makes the assignment reproducible from `lengths`
    alone -- so only `lengths` has to be stored)."""
    codes = np.zeros(lengths.size, dtype=np.int64)
    used = np.flatnonzero(lengths > 0)
    if used.size == 0:
        return codes, 0
    maxlen = int(lengths.max())
    order = used[np.argsort(lengths[used], kind="stable")]
    code = 0
    prev = int(lengths[order[0]])
    for s in order:
        l = int(lengths[s])
        code <<= (l - prev)
        prev = l
        codes[s] = code
        code += 1
    return codes, maxlen


def _decode_symbols(packed: np.ndarray, nbits: int, n: int,
                    lengths: np.ndarray) -> np.ndarray:
    """Decode n canonical-Huffman symbols with one table lookup each."""
    out = np.zeros(n, dtype=np.int64)
    if n == 0:
        return out
    codes, maxlen = _canonical(lengths)
    tbl_sym = np.zeros(1 << maxlen, dtype=np.int64)
    tbl_len = np.ones(1 << maxlen, dtype=np.int64)
    for s in np.flatnonzero(lengths > 0):
        l = int(lengths[s])
        base = int(codes[s]) << (maxlen - l)
        span = 1 << (maxlen - l)
        tbl_sym[base:base + span] = s
        tbl_len[base:base + span] = l
    bits = np.unpackbits(packed)[:nbits]
    padded = np.concatenate((bits, np.zeros(maxlen, dtype=np.uint8)))
    win = np.zeros(nbits, dtype=np.int64)
    for j in range(maxlen):
        win = (win << 1) | padded[j:j + nbits]
    wl, sl, ll = win.tolist(), tbl_sym.tolist(), tbl_len.tolist()
    res = [0] * n
    pos = 0
    for i in range(n):
        v = wl[pos]
        res[i] = sl[v]
        pos += ll[v]
    out[:] = res
    return out


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

class EFRunIndex:
    """Gap-coded PBWT run boundaries.  Holds no panel and no a/d arrays."""

    __slots__ = ("exp", "man", "clen", "n", "n_bnd", "nb_exp", "nb_man", "N", "M")

    def __init__(self, exp, man, clen, n, n_bnd, nb_exp, nb_man, N, M):
        self.exp = exp        # uint8: packed Huffman-coded gap exponents
        self.man = man        # uint8: packed variable-width gap mantissas
        self.clen = clen      # uint8: Huffman code length per exponent symbol
        self.n = n            # elements in the coded monotone sequence
        self.n_bnd = n_bnd    # how many of them are run boundaries
        self.nb_exp = nb_exp  # valid bits in `exp`
        self.nb_man = nb_man  # valid bits in `man`
        self.N = N
        self.M = M

    def nbytes(self) -> int:
        """Every retained byte.  exp/man/clen are the only arrays kept alive;
        the six remaining fields are python ints, charged 8 bytes each."""
        return int(self.exp.nbytes + self.man.nbytes + self.clen.nbytes + 48)

    # ---- decoding: purely local, never cached on the object ---------------
    def _decode(self):
        """Return (bpos, off, first_bits)."""
        N, M, n = self.N, self.M, self.n
        first_bits = np.zeros(M, dtype=np.uint8)
        if n == 0:
            return np.zeros(0, np.int64), np.zeros(M + 1, np.int64), first_bits
        e = _decode_symbols(self.exp, self.nb_exp, n, self.clen.astype(np.int64))
        m = _unpack_var(self.man, e, self.nb_man)
        g = np.cumsum(_POW2[e] + m) - 1

        nb = self.n_bnd
        if n > nb:                      # tail: positions of the set first-bits
            first_bits[g[nb:] - N * M] = 1
        g = g[:nb]
        col, bpos = np.divmod(g, N)
        off = np.zeros(M + 1, dtype=np.int64)
        np.cumsum(np.bincount(col, minlength=M), out=off[1:])
        return bpos, off, first_bits


def _encode(seq: np.ndarray, n_bnd: int, N: int, M: int) -> EFRunIndex:
    n = int(seq.size)
    if n == 0:
        z = np.zeros(0, dtype=np.uint8)
        return EFRunIndex(z, z, z, 0, 0, 0, 0, N, M)
    gaps = np.empty(n, dtype=np.int64)
    gaps[0] = seq[0] + 1
    gaps[1:] = seq[1:] - seq[:-1]
    e = np.searchsorted(_POW2, gaps, side="right") - 1      # exact floor(log2)
    man = gaps - _POW2[e]

    counts = np.bincount(e, minlength=int(e.max()) + 1)
    clen = _huffman_lengths(counts)
    codes, _ = _canonical(clen)

    w_exp = clen[e]
    nb_exp = int(w_exp.sum())
    exp = _pack_var(codes[e], w_exp, nb_exp)
    nb_man = int(e.sum())
    man_packed = _pack_var(man, e, nb_man)
    return EFRunIndex(exp, man_packed, clen.astype(np.uint8),
                      n, n_bnd, nb_exp, nb_man, N, M)


def build(panel: np.ndarray) -> EFRunIndex:
    N, M = panel.shape
    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)
    cols, first = [], np.zeros(M, dtype=np.uint8)
    for k in range(M):
        alleles = panel[cur_a, k]
        b = np.flatnonzero(alleles[1:] != alleles[:-1]) + 1
        cols.append(b.astype(np.int64) + k * N)
        if N:
            first[k] = alleles[0]
        cur_a, cur_d = pbwt_step(cur_a, cur_d, alleles, k)

    bnd = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    # the set first-bits ride in the same monotone sequence, past the boundary
    # universe N*M, so one code serves both streams.
    tail = np.flatnonzero(first).astype(np.int64) + N * M
    return _encode(np.concatenate((bnd, tail)), int(bnd.size), N, M)


# --------------------------------------------------------------------------
# query: the PBWT virtual-insertion sweep, with the allele column rebuilt
# from the decoded run boundaries instead of read from a stored panel.
# --------------------------------------------------------------------------

def query_batch(index: EFRunIndex, Z: np.ndarray, L: int):
    N, M = index.N, index.M
    Q = Z.shape[0]
    out = [[] for _ in range(Q)]

    bpos, off, first_bits = index._decode()

    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)
    t = np.zeros(Q, dtype=np.int64)
    d_above = np.zeros(Q, dtype=np.int64)
    d_below = np.zeros(Q, dtype=np.int64)

    for k in range(M + 1):
        if k < M:
            b = bpos[off[k]:off[k + 1]]
            starts = np.empty(b.size + 1, dtype=np.int64)
            starts[0] = 0
            starts[1:] = b
            lengths = np.diff(np.append(starts, N))
            vals = (int(first_bits[k]) ^ (np.arange(starts.size) & 1)).astype(np.uint8)
            alleles = np.repeat(vals, lengths)
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


def query(index: EFRunIndex, z: np.ndarray, L: int):
    return query_batch(index, z.reshape(1, -1), L)[0]
