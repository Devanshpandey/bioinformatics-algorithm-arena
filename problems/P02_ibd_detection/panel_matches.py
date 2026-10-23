"""
All-pairs maximal long matches within a panel -- the seeding primitive that
IBD detection is built on, and the point where P01 and P02 share machinery.

Two implementations:
  brute_force_matches  : O(N^2 * M) exact oracle -- the definition
  pbwt_all_matches     : Durbin (2014) Algorithm 3, block enumeration over the
                         positional prefix + divergence arrays

The dependency is the point. If an agent invents a better haplotype index for
P01, this module is where the improvement enters P02, and the arena measures
that transfer instead of asserting it.
"""
from __future__ import annotations

import numpy as np

# Import P01's PBWT column update by explicit path. Do NOT put P01's directory
# on sys.path: both problems define generate.py, and a shared sys.path lets one
# problem's module shadow another's. Cross-problem shadowing already produced a
# silent wrong-instance bug once; explicit-path loading makes it impossible.
import importlib.util as _ilu
import os as _os

_P01 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "P01_haplotype_long_match", "pbwt_core.py")
_spec = _ilu.spec_from_file_location("_p01_pbwt_core", _P01)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pbwt_step = _mod.pbwt_step


def brute_force_matches(panel: np.ndarray, L: int):
    """Every maximal match of >= L sites between every pair. Exact by definition."""
    N, M = panel.shape
    out = set()
    for i in range(N):
        for j in range(i + 1, N):
            eq = panel[i] == panel[j]
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


def pbwt_all_matches(panel: np.ndarray, L: int):
    """Durbin Algorithm 3: all maximal matches >= L sites, via PBWT blocks.

    At each site k, haplotypes matching each other back to at least k-L form
    contiguous blocks of a_k delimited by divergence values exceeding k-L.
    A pair inside a block whose alleles differ at k has a maximal match ending
    at k, starting at the running maximum of d_k across the span between them.
    """
    N, M = panel.shape
    out = set()
    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)

    for k in range(M + 1):
        thresh = k - L
        col = panel[:, k] if k < M else None
        alleles = col[cur_a] if k < M else None

        i0 = 0
        for i in range(1, N + 1):
            if i == N or cur_d[i] > thresh:
                if i - i0 > 1:
                    for p in range(i0, i):
                        m = 0
                        for q in range(p + 1, i):
                            dq = int(cur_d[q])
                            if dq > m:
                                m = dq
                            if k - m < L:
                                break
                            if k == M or alleles[p] != alleles[q]:
                                x, y = int(cur_a[p]), int(cur_a[q])
                                out.add((min(x, y), max(x, y), m, k))
                i0 = i
        if k < M:
            cur_a, cur_d = pbwt_step(cur_a, cur_d, alleles, k)
    return out
