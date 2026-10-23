"""
P09 core: exact phylogenetic likelihood under JC69, and tree utilities.

The objective is EXACTLY COMPUTABLE. Felsenstein's pruning algorithm returns
the log-likelihood of a topology + branch lengths under a fixed model to
machine precision. There is no answer key to memorize and no curated truth to
be biased -- the score is a function of the submission's own output. This is
the strongest evaluator class in the arena, the same class as exact minimizer-density
evaluation.

Trees are unrooted binary; JC69 is reversible, so we root arbitrarily at
leaf 0 and the likelihood is unchanged.
"""
from __future__ import annotations

import numpy as np

CLV_UPDATES = [0]  # implementation-independent work counter (see SPEC.md II.2)


def reset_counter():
    CLV_UPDATES[0] = 0


class Tree:
    """Rooted binary tree; leaves 0..n-1; node arrays sized 2n-2."""

    __slots__ = ("n", "left", "right", "blen", "root", "order")

    def __init__(self, n, left, right, blen, root):
        self.n, self.left, self.right = n, left, right
        self.blen, self.root = blen, root
        self.order = self._postorder()

    def _postorder(self):
        out, stack = [], [self.root]
        while stack:
            u = stack.pop()
            if self.left[u] >= 0:
                out.append(u)
                stack.append(self.left[u]); stack.append(self.right[u])
        return out[::-1]

    def copy(self):
        return Tree(self.n, self.left[:], self.right[:], self.blen.copy(), self.root)

    def internal_edges(self):
        """Edges whose removal splits the tree non-trivially -- the NNI targets."""
        par = {}
        for u in self.order:
            par[self.left[u]] = u
            par[self.right[u]] = u
        return [u for u in self.order
                if u != self.root and self.left[u] >= 0 and par.get(u) is not None]

    def splits(self):
        """Set of leaf-bipartitions -- the topology identity, for RF distance."""
        desc, out = {}, set()
        for u in self.order:
            for c in (self.left[u], self.right[u]):
                if c < self.n:
                    desc[c] = frozenset([c])
            s = desc[self.left[u]] | desc[self.right[u]]
            desc[u] = s
            if 1 < len(s) < self.n - 1:
                out.add(min(s, frozenset(range(self.n)) - s, key=sorted))
        return out


def loglik(tree: Tree, data: np.ndarray) -> float:
    """Exact JC69 log-likelihood. data: (n_leaves, n_sites) ints in 0..3."""
    n, ns = tree.n, data.shape[1]
    n_nodes = len(tree.left)
    clv = [None] * n_nodes
    for i in range(n):
        m = np.zeros((4, ns))
        m[data[i], np.arange(ns)] = 1.0
        clv[i] = m
    scale = np.zeros(ns)
    for u in tree.order:
        acc = None
        for c in (tree.left[u], tree.right[u]):
            t = max(tree.blen[c], 1e-9)
            e = np.exp(-4.0 * t / 3.0)
            L = clv[c]
            contrib = (0.25 - 0.25 * e) * L.sum(axis=0) + e * L
            acc = contrib if acc is None else acc * contrib
            CLV_UPDATES[0] += 1
        mx = acc.max(axis=0)
        mx[mx <= 0] = 1.0
        clv[u] = acc / mx
        scale += np.log(mx)
    site_lik = 0.25 * clv[tree.root].sum(axis=0)
    return float(np.sum(np.log(np.maximum(site_lik, 1e-300)) + scale))


def optimize_blens(tree: Tree, data: np.ndarray, passes: int = 3) -> float:
    """Golden-section optimisation of each branch length, a few sweeps."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    best = loglik(tree, data)
    for _ in range(passes):
        for u in range(len(tree.left)):
            if u == tree.root:
                continue
            lo, hi = 1e-6, 2.0
            c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
            orig = tree.blen[u]
            tree.blen[u] = c; fc = loglik(tree, data)
            tree.blen[u] = d; fd = loglik(tree, data)
            for _ in range(12):
                if fc > fd:
                    hi, d, fd = d, c, fc
                    c = hi - gr * (hi - lo)
                    tree.blen[u] = c; fc = loglik(tree, data)
                else:
                    lo, c, fc = c, d, fd
                    d = lo + gr * (hi - lo)
                    tree.blen[u] = d; fd = loglik(tree, data)
            cand = c if fc > fd else d
            tree.blen[u] = cand
            v = loglik(tree, data)
            if v < best:
                tree.blen[u] = orig
            else:
                best = v
    return best


def from_nested(t, n) -> Tree:
    """Build a Tree from nested tuples of leaf indices."""
    left, right, blen = [-1] * n, [-1] * n, [0.1] * n
    def rec(node):
        if isinstance(node, int):
            return node
        a, b = rec(node[0]), rec(node[1])
        left.append(a); right.append(b); blen.append(0.1)
        return len(left) - 1
    root = rec(t)
    return Tree(n, left, right, np.array(blen, dtype=float), root)


def enumerate_topologies(n):
    """Every unrooted binary topology on n leaves, as (2n-5)!! nested tuples."""
    def insert(tree, leaf):
        out = []
        def rec(node):
            res = [(leaf, node)]
            if not isinstance(node, int):
                for i in (0, 1):
                    for sub in rec(node[i]):
                        res.append((sub, node[1]) if i == 0 else (node[0], sub))
            return res
        for t in rec(tree):
            out.append(t)
        return out

    # An unrooted tree on n leaves == a ROOTED tree on leaves 1..n-1 with leaf 0
    # attached at the root as outgroup. Rooted trees on m leaves number (2m-3)!!,
    # so with m = n-1 this yields exactly (2n-5)!! unrooted topologies.
    trees = [1]
    for leaf in range(2, n):
        nxt = []
        for t in trees:
            nxt.extend(insert(t, leaf))
        trees = nxt
    return [(0, t) for t in trees]


def rf_distance(a: Tree, b: Tree) -> int:
    """Robinson-Foulds distance between two topologies."""
    sa, sb = a.splits(), b.splits()
    return len(sa - sb) + len(sb - sa)
