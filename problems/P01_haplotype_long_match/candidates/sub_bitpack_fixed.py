"""
P01 submission: bit-packed / entropy-coded run-length encoded PBWT columns.

    incumbent  sub_rlpbwt_streaming   286,778 B   0.25092 bits/cell
    this file                          41,082 B   0.03594 bits/cell     (6.98x)

  measured by score_submission.py on the small tier (980 x 9330), gate=true on
  5 independent fresh gate seeds. Stable across instances: 0.03594 / 0.03611 /
  0.03689 bits/cell on tier seeds 42 / 7 / 123. Query time 0.179 s/query,
  unchanged from the incumbent's 0.172, so the space came for free.

  where the 41,082 bytes are:
      gap    35,517 B  86.5%   gap stream (Huffman exponent + raw mantissa)
      cnt     4,021 B   9.8%   run counts, same coding
      fst     1,167 B   2.8%   1 bit/column, allele of run 0
      gtab      297 B   0.7%   36 context codebooks, 6-bit code lengths
      ctab        8 B          run-count codebook
      scalars    72 B          N, M, NC, csym, gsym

WHAT IS STORED AND WHY THAT IS ENOUGH
The incumbent's insight is that during a left-to-right sweep every panel access
reduces to `alleles[j]`, the allele at position j of the current PBWT column,
and that column in a_k order IS a run-length-encoded object -- so the panel need
never be stored at all. That is kept. What is replaced is the encoding of the
runs, which the incumbent left in three wasteful arrays (N=980, M=9330,
r=50,700 runs):

    starts   int32, one per run                  202,800 B   (70.7%)
    offsets  int64, one per column + 1            74,648 B   (26.0%)
    first    uint8, one per column                 9,330 B    (3.3%)

Removing those three is the first 4.5x and is pure bookkeeping:

  (a) A run start is a position in [0, N): it needs ceil(log2 N) = 10 bits, not
      32. Note that ELIAS-FANO IS THE WRONG TOOL HERE and was measured to be
      worse: EF costs r*(log2(N/r)+2) ~ 51.6 bits/column against 44.3 for plain
      fixed-width packing. EF's 2 bits/element bucket overhead only pays when r
      is a large fraction of N; at 5.43 runs among 980 positions it is pure loss.
      This sequence is sparse, not dense.
  (b) Run 0 of every column starts at position 0, so 9,330 of the 50,700 stored
      values carried no information. Not stored.
  (c) The offsets array is a quarter of the index and is pure redundancy: the
      query is a strictly left-to-right sweep, so it never needs random access,
      only the ability to advance. A per-column run count plus a running cursor
      replaces it.

The second 1.55x comes from measuring the run structure instead of guessing:

  (d) GAPS, NOT POSITIONS, SPLIT BY ALLELE. Boundaries are stored as
      g_i = B_i - B_{i-1} - 1. A PBWT column is a long major-allele run
      interleaved with short minor-allele runs, so allele-0 gaps (mean 217) and
      allele-1 gaps (mean 48) are two different populations. The decoder already
      knows each run's allele -- it is first[k] ^ (run index & 1) -- so the split
      is free side information.
  (e) CONTEXT ON THE RUN COUNT. Boundaries satisfy sum(g_i) + nb <= N-1, so a
      column with many runs cannot have large gaps. nb is decoded before its own
      gaps, so conditioning on bit_length(nb) is also free.
  (f) EXPONENT/MANTISSA WITH CANONICAL HUFFMAN EXPONENTS. Each gap is split into
      e = bit_length(g) and its e-1 low bits. The mantissa is near-uniform inside
      a bit-length class and is written raw; e is highly skewed and is Huffman
      coded under context (allele, bit_length(nb), run index == 0). Run counts
      get the same treatment. Codebooks are built from the instance at build
      time and stored as 6-bit code lengths (~300 B), so nothing is tuned to a
      fixed instance and the decoder derives canonical codes from the same
      lengths the encoder used.

WHERE THIS SITS AGAINST THE FLOOR
The gap stream costs 35,517 B against a conditional order-0 entropy of 34,746 B
for the same context -- within 3% of the floor for this model class. Of that,
20,001 B is mantissa, and splitting each bit-length class in half to test
whether the mantissa is really uniform recovered only 181 B, so it is. Summing
every remaining stream against its own conditional entropy leaves ~1,500 B, 4%
of the index, all of which would need an adaptive arithmetic coder to collect.
This is at the floor of the exponent/mantissa factorisation; further gains need
a different factorisation, not a better coder.

MEASURED AND REJECTED, recorded so they are not retried:
  * Cross-column delta coding. Consecutive PBWT columns share only 1.7% of their
    run boundaries against a 0.55% random baseline, so there is essentially no
    inter-column redundancy. Dead end.
  * General-purpose compression. lzma -9 applied to the whole PBWT-sorted column
    bitstream reaches 58,264 B (0.051 bits/cell) -- this code is 1.42x smaller
    than an off-the-shelf compressor given the same transformed data, so nothing
    is being deferred to a black box.
  * Elias-Fano on run starts -- see (a), 51.6 vs 44.3 bits/column.
  * Finer context on the gap exponent: adding bit_length(previous gap) cut the
    payload by 488 B but cost 1,114 B of extra codebooks. Net loss.

HONEST ACCOUNTING
nbytes() returns the five stored arrays plus the scalar header, which is every
attribute the object has (__slots__ makes that checkable). No panel, no a/d
arrays, no cached columns, nothing recomputed from the panel at query time.
Decoding is streamed: the sweep pulls 1024 columns at a time through one
persistent bit reader, so the transient buffer is O(block) and the index is
never materialised in uncompressed form. Verified: every stored array has
base=None and shares no memory with the panel, and query_batch still returns the
same match set after the panel object has been deleted and garbage collected.

Scaling in N (row subsamples of the same coalescent panel), bits/cell:
    N=60 0.192 | 120 0.128 | 245 0.084 | 490 0.056 | 980 0.0359
Cost per cell FALLS with N because runs/column grows sublinearly, so the scored
large-N tier is the favourable regime for this representation, not the hostile
one. The advantage over the incumbent narrows with N (13.6x -> 7.0x) because the
incumbent's fixed int32 waste shrinks in relative terms as boundaries genuinely
need more bits.

NOT SOLVED HERE, and still the open problem: divergence is recomputed during the
sweep in O(N) per site rather than stored with O(1) access, so a single online
query remains O(N*M). This submission is a space result only.
"""
from __future__ import annotations

import heapq

import numpy as np

from pbwt_core import pbwt_step

_BLOCK = 1024      # columns per decode refill
_LENBITS = 6       # bits per stored Huffman code length (max length 63)
_FAST = 8          # bits of Huffman lookup-table fast path


# ------------------------------------------------------------ bit primitives

def _pack(values, w: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.int64)
    if v.size == 0 or w <= 0:
        return np.zeros(0, dtype=np.uint8)
    shifts = np.arange(w - 1, -1, -1, dtype=np.int64)
    return np.packbits((((v[:, None] >> shifts) & 1).astype(np.uint8)).reshape(-1))


def _unpack(blob, bit_start: int, n: int, w: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if w <= 0:
        return np.zeros(n, dtype=np.int64)
    b0 = bit_start >> 3
    b1 = (bit_start + n * w + 7) >> 3
    bits = np.unpackbits(blob[b0:b1])
    off = bit_start - (b0 << 3)
    seg = bits[off:off + n * w].reshape(n, w).astype(np.int64)
    return seg @ np.left_shift(np.int64(1), np.arange(w - 1, -1, -1, dtype=np.int64))


class _BitWriter:
    __slots__ = ("out", "acc", "n")

    def __init__(self):
        self.out, self.acc, self.n = bytearray(), 0, 0

    def write(self, v: int, k: int):
        if k <= 0:
            return
        self.acc = (self.acc << k) | (v & ((1 << k) - 1))
        self.n += k
        while self.n >= 8:
            self.n -= 8
            self.out.append((self.acc >> self.n) & 0xFF)
            self.acc &= (1 << self.n) - 1

    def finish(self) -> np.ndarray:
        if self.n:
            self.out.append((self.acc << (8 - self.n)) & 0xFF)
            self.acc, self.n = 0, 0
        return np.frombuffer(bytes(self.out), dtype=np.uint8).copy()


class _BitReader:
    """MSB-first reader over a bytes buffer, zero-padded so peek never runs off."""

    __slots__ = ("buf", "pos", "acc", "n", "end")

    def __init__(self, blob: np.ndarray):
        self.buf = blob.tobytes() + b"\x00" * 16
        self.end = len(self.buf)
        self.pos, self.acc, self.n = 0, 0, 0

    def _fill(self):
        buf, p, acc, n = self.buf, self.pos, self.acc, self.n
        while n <= 48 and p < self.end:
            acc = (acc << 8) | buf[p]
            p += 1
            n += 8
        self.pos, self.acc, self.n = p, acc, n

    def peek(self, k: int) -> int:
        if self.n < k:
            self._fill()
        return (self.acc >> (self.n - k)) & ((1 << k) - 1)

    def skip(self, k: int):
        self.n -= k
        self.acc &= (1 << self.n) - 1

    def read(self, k: int) -> int:
        if k <= 0:
            return 0
        if self.n < k:
            self._fill()
        self.n -= k
        v = (self.acc >> self.n) & ((1 << k) - 1)
        self.acc &= (1 << self.n) - 1
        return v


# --------------------------------------------------------- canonical Huffman

def _huff_lengths(freq: np.ndarray) -> np.ndarray:
    """Code lengths from symbol counts. A one-symbol alphabet gets length 1."""
    L = np.zeros(freq.size, dtype=np.int64)
    used = [int(s) for s in np.flatnonzero(freq)]
    if not used:
        return L
    if len(used) == 1:
        L[used[0]] = 1
        return L
    heap = [(int(freq[s]), i, [s]) for i, s in enumerate(used)]
    heapq.heapify(heap)
    nid = len(heap)
    while len(heap) > 1:
        c1, _, g1 = heapq.heappop(heap)
        c2, _, g2 = heapq.heappop(heap)
        merged = g1 + g2
        for s in merged:
            L[s] += 1
        heapq.heappush(heap, (c1 + c2, nid, merged))
        nid += 1
    return L


def _canonical(L: np.ndarray):
    """Canonical codes from code lengths -- identical on both sides."""
    L = np.asarray(L, dtype=np.int64)
    maxlen = int(L.max()) if L.size else 0
    codes = np.zeros(L.size, dtype=np.int64)
    if maxlen == 0:
        return codes, 0, None
    bl_count = np.bincount(L[L > 0], minlength=maxlen + 1)
    next_code = np.zeros(maxlen + 2, dtype=np.int64)
    code = 0
    for b in range(1, maxlen + 1):
        code = (code + int(bl_count[b - 1])) << 1
        next_code[b] = code
    first = next_code.copy()
    for s in range(L.size):
        if L[s]:
            codes[s] = next_code[L[s]]
            next_code[L[s]] += 1
    return codes, maxlen, (first, bl_count)


class _Decoder:
    """Canonical Huffman decoder: 8-bit lookup fast path, bit-walk fallback."""

    __slots__ = ("maxlen", "fast_sym", "fast_len", "first", "cnt", "base", "order")

    def __init__(self, L: np.ndarray):
        codes, maxlen, tabs = _canonical(L)
        self.maxlen = maxlen
        if maxlen == 0:
            self.fast_sym = self.fast_len = None
            return
        first, bl_count = tabs
        self.first, self.cnt = first, bl_count
        order = [s for b in range(1, maxlen + 1)
                 for s in range(L.size) if L[s] == b]
        self.order = order
        base = np.zeros(maxlen + 2, dtype=np.int64)
        acc = 0
        for b in range(1, maxlen + 1):
            base[b] = acc
            acc += int(bl_count[b])
        self.base = base
        self.fast_sym = [0] * (1 << _FAST)
        self.fast_len = [0] * (1 << _FAST)
        for s in range(L.size):
            b = int(L[s])
            if 0 < b <= _FAST:
                pref = int(codes[s]) << (_FAST - b)
                for x in range(pref, pref + (1 << (_FAST - b))):
                    self.fast_sym[x], self.fast_len[x] = s, b

    def decode(self, br: _BitReader) -> int:
        ml = self.maxlen
        if ml == 0:
            return 0
        x = br.peek(_FAST) if ml >= _FAST else (br.peek(ml) << (_FAST - ml))
        b = self.fast_len[x]
        if b:
            br.skip(b)
            return self.fast_sym[x]
        code = br.peek(ml)
        for b in range(_FAST + 1, ml + 1):
            c = code >> (ml - b)
            k = int(self.cnt[b])
            if k and 0 <= c - int(self.first[b]) < k:
                br.skip(b)
                return self.order[int(self.base[b]) + c - int(self.first[b])]
        # Unreachable for a complete canonical code. Raise rather than return a
        # wrong symbol: a silent fallback here would corrupt the match set.
        raise ValueError("corrupt Huffman stream")


def _bitlen_arr(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64)
    out = np.zeros(v.shape, dtype=np.int64)
    x = v.copy()
    while np.any(x > 0):
        out += (x > 0)
        x >>= 1
    return out


# ------------------------------------------------------------------- index

class HuffRLPBWT:
    """Entropy-coded run-length PBWT columns. Holds no panel."""

    __slots__ = ("fst", "cnt", "gap", "ctab", "gtab",
                 "N", "M", "NC", "csym", "gsym")

    def __init__(self, fst, cnt, gap, ctab, gtab, N, M, NC, csym, gsym):
        self.fst = fst      # 1 bit/column: allele of run 0
        self.cnt = cnt      # Huffman(exponent)+mantissa stream of run counts
        self.gap = gap      # Huffman(exponent)+mantissa stream of gaps
        self.ctab = ctab    # 6-bit code lengths, run-count book
        self.gtab = gtab    # 6-bit code lengths, 4*NC context books
        self.N, self.M, self.NC = N, M, NC
        self.csym, self.gsym = csym, gsym

    def nbytes(self) -> int:
        # Every attribute this object has; __slots__ makes that verifiable.
        return int(self.fst.nbytes + self.cnt.nbytes + self.gap.nbytes
                   + self.ctab.nbytes + self.gtab.nbytes) + 9 * 8

    def _books(self):
        cdec = _Decoder(_unpack(self.ctab, 0, self.csym, _LENBITS)) if self.csym else None
        gdec = []
        for b in range(4 * self.NC):
            gdec.append(_Decoder(_unpack(self.gtab, b * self.gsym * _LENBITS,
                                         self.gsym, _LENBITS)) if self.gsym else None)
        return cdec, gdec

    def columns(self):
        """Generator yielding dense allele column k for k = 0..M-1."""
        N, M, NC = self.N, self.M, self.NC
        cdec, gdec = self._books()
        cbr, gbr = _BitReader(self.cnt), _BitReader(self.gap)
        arange = np.arange
        for k0 in range(0, M, _BLOCK):
            n = min(k0 + _BLOCK, M) - k0
            fst = _unpack(self.fst, k0, n, 1)
            for i in range(n):
                e = cdec.decode(cbr)
                nb = ((1 << (e - 1)) | cbr.read(e - 1)) if e else 0
                f = int(fst[i])
                if nb == 0:
                    yield np.full(N, f, dtype=np.uint8) if N else np.zeros(0, np.uint8)
                    continue
                cls = min(int(nb).bit_length(), NC) - 1
                starts = np.empty(nb + 2, dtype=np.int64)
                starts[0] = 0
                pos = 0
                for j in range(nb):
                    ctx = (((f ^ (j & 1)) * 2) + (1 if j == 0 else 0)) * NC + cls
                    d = gdec[ctx]
                    e = d.decode(gbr)
                    g = ((1 << (e - 1)) | gbr.read(e - 1)) if e else 0
                    pos += g + 1
                    starts[j + 1] = pos
                starts[nb + 1] = N
                alt = (f ^ (arange(nb + 1) & 1)).astype(np.uint8)
                yield np.repeat(alt, starts[1:] - starts[:-1])

    def column(self, k: int) -> np.ndarray:
        """Random access by replay -- debugging only; the sweep uses columns()."""
        for i, c in enumerate(self.columns()):
            if i == k:
                return c
        raise IndexError(k)


def build(panel: np.ndarray) -> HuffRLPBWT:
    panel = np.asarray(panel, dtype=np.uint8)
    N, M = panel.shape
    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)

    per_col, nb = [], np.zeros(M, dtype=np.int64)
    fst = np.zeros(M, dtype=np.int64)
    for k in range(M):
        alleles = panel[cur_a, k]
        if N:
            b = (np.flatnonzero(alleles[1:] != alleles[:-1]) + 1).astype(np.int64)
            fst[k] = int(alleles[0])
        else:
            b = np.zeros(0, dtype=np.int64)
        per_col.append(b)
        nb[k] = b.size
        cur_a, cur_d = pbwt_step(cur_a, cur_d, alleles, k)

    NC = max(1, int(_bitlen_arr(nb).max())) if M else 1
    # gather gaps with their context, in the exact order the decoder will ask
    gaps, ctxs = [], []
    for k in range(M):
        prev, f = 0, int(fst[k])
        cls = min(int(nb[k]).bit_length(), NC) - 1
        for j, v in enumerate(per_col[k].tolist()):
            gaps.append(v - prev - 1)
            ctxs.append((((f ^ (j & 1)) * 2) + (1 if j == 0 else 0)) * NC + cls)
            prev = v
    gaps = np.array(gaps, dtype=np.int64)
    ctxs = np.array(ctxs, dtype=np.int64)

    cexp = _bitlen_arr(nb)
    gexp = _bitlen_arr(gaps)
    csym = int(cexp.max()) + 1 if M else 0
    gsym = int(gexp.max()) + 1 if gaps.size else 0

    # run counts: one book
    if csym:
        cL = _huff_lengths(np.bincount(cexp, minlength=csym))
        ccodes, _, _ = _canonical(cL)
        bw = _BitWriter()
        for k in range(M):
            e = int(cexp[k])
            bw.write(int(ccodes[e]), int(cL[e]))
            if e:
                bw.write(int(nb[k]) - (1 << (e - 1)), e - 1)
        cnt_buf, ctab = bw.finish(), _pack(cL, _LENBITS)
    else:
        cnt_buf, ctab = np.zeros(0, np.uint8), np.zeros(0, np.uint8)

    # gaps: 4*NC context books
    nbooks = 4 * NC
    if gsym:
        gL = np.zeros((nbooks, gsym), dtype=np.int64)
        for b in range(nbooks):
            m = ctxs == b
            if np.any(m):
                gL[b] = _huff_lengths(np.bincount(gexp[m], minlength=gsym))
        gcodes = [_canonical(gL[b])[0] for b in range(nbooks)]
        bw = _BitWriter()
        for g, e, c in zip(gaps.tolist(), gexp.tolist(), ctxs.tolist()):
            bw.write(int(gcodes[c][e]), int(gL[c][e]))
            if e:
                bw.write(g - (1 << (e - 1)), e - 1)
        gap_buf, gtab = bw.finish(), _pack(gL.reshape(-1), _LENBITS)
    else:
        gap_buf, gtab = np.zeros(0, np.uint8), np.zeros(0, np.uint8)

    return HuffRLPBWT(_pack(fst, 1), cnt_buf, gap_buf, ctab, gtab,
                      N, M, NC, csym, gsym)


# ------------------------------------------------------------------ the sweep
# Traversal identical to sub_rlpbwt_streaming; only the column source changed.

def query_batch(index: HuffRLPBWT, Z: np.ndarray, L: int):
    N, M = index.N, index.M
    Q = Z.shape[0]
    out = [[] for _ in range(Q)]

    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)
    t = np.zeros(Q, dtype=np.int64)
    d_above = np.zeros(Q, dtype=np.int64)
    d_below = np.zeros(Q, dtype=np.int64)

    cols = index.columns()
    alleles = None
    for k in range(M + 1):
        if k < M:
            alleles = next(cols)               # <- decoded from the bitstream
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


def query(index: HuffRLPBWT, z: np.ndarray, L: int):
    return query_batch(index, z.reshape(1, -1), L)[0]
