"""
P09 incumbent: neighbour-joining start + NNI hill climbing.

This is the classical ML tree-search skeleton that RAxML-NG and IQ-TREE
elaborate on. Its stages are exactly the mutable surface:

    STAGE 1  start   distance estimate + neighbour joining
    STAGE 2  moves   the neighbourhood (NNI here; SPR/TBR are larger)
    STAGE 3  order   which moves to try, and in what order
    STAGE 4  eval    full vs lazy likelihood evaluation of a proposed move
    STAGE 5  blens   branch-length optimisation schedule
    STAGE 6  stop    termination criterion

Submission API:  search(data, budget_s) -> Tree
"""
from __future__ import annotations

import time

import numpy as np

from phylo import Tree, loglik, optimize_blens


def jc_distance(data):
    n = data.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = float(np.mean(data[i] != data[j]))
            p = min(p, 0.74)
            d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = D[j, i] = max(d, 1e-4)
    return D


def neighbor_joining(D):
    n = D.shape[0]
    nodes = list(range(n))
    left, right, blen = [-1] * n, [-1] * n, [0.1] * n
    D = D.copy()
    active = list(range(n))
    while len(active) > 2:
        m = len(active)
        r = {i: sum(D[i, j] for j in active if j != i) / (m - 2) for i in active}
        best, bi, bj = None, None, None
        for x in range(m):
            for y in range(x + 1, m):
                i, j = active[x], active[y]
                q = D[i, j] - r[i] - r[j]
                if best is None or q < best:
                    best, bi, bj = q, i, j
        left.append(bi); right.append(bj); blen.append(0.1)
        u = len(left) - 1
        newD = np.zeros((u + 1, u + 1))
        newD[:D.shape[0], :D.shape[1]] = D
        for k in active:
            if k in (bi, bj):
                continue
            newD[u, k] = newD[k, u] = 0.5 * (D[bi, k] + D[bj, k] - D[bi, bj])
        D = newD
        active = [a for a in active if a not in (bi, bj)] + [u]
    a, b = active
    left.append(a); right.append(b); blen.append(0.1)
    return Tree(n, left, right, np.array(blen, dtype=float), len(left) - 1)


def _nni_variants(tree, edge):
    """The two NNI rearrangements around an internal edge."""
    par = {}
    for u in tree.order:
        par[tree.left[u]] = u
        par[tree.right[u]] = u
    p = par[edge]
    sib = tree.right[p] if tree.left[p] == edge else tree.left[p]
    out = []
    for which in (0, 1):
        t = tree.copy()
        c = t.left[edge] if which == 0 else t.right[edge]
        if t.left[p] == edge:
            t.right[p] = c
        else:
            t.left[p] = c
        if which == 0:
            t.left[edge] = sib
        else:
            t.right[edge] = sib
        out.append(Tree(t.n, t.left, t.right, t.blen, t.root))
    return out


def search(data, budget_s=30.0):
    t0 = time.time()
    tree = neighbor_joining(jc_distance(data))
    best = optimize_blens(tree, data, passes=2)
    improved = True
    while improved and time.time() - t0 < budget_s:
        improved = False
        for e in tree.internal_edges():
            if time.time() - t0 > budget_s:
                break
            for cand in _nni_variants(tree, e):
                v = loglik(cand, data)
                if v > best + 1e-6:
                    v = optimize_blens(cand, data, passes=1)
                    if v > best + 1e-6:
                        tree, best, improved = cand, v, True
                        break
    optimize_blens(tree, data, passes=2)
    return tree
