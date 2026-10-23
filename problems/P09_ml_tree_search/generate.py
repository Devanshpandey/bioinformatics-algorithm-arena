"""
P09 instance generator: alignments simulated along a KNOWN true tree.

Fresh seed per instance means no public alignment benchmark is involved, so
neither the true tree nor the reference alignment can have been memorized. The
scored objective (log-likelihood) does not even depend on the true tree -- the
true tree is used only for the SECONDARY topological-accuracy metric, because
the ML tree is not always the true tree and conflating the two is a classic
error in this literature.
"""
from __future__ import annotations

import numpy as np

from phylo import enumerate_topologies, from_nested

# mean_blen matters as much as size. Long branches + few sites = a rugged
# likelihood surface; that is where search is actually hard. The "small" tier
# (short branches, many sites) is SATURATED -- NJ+NNI recovers the true tree
# exactly -- and must never be used to claim a search win.
TIERS = {
    "gate":   dict(n_taxa=6,   n_sites=400,  mean_blen=0.10),  # exhaustively solvable
    "tiny":   dict(n_taxa=7,   n_sites=500,  mean_blen=0.10),
    "small":  dict(n_taxa=16,  n_sites=1000, mean_blen=0.10),  # SATURATED
    "hard16": dict(n_taxa=16,  n_sites=200,  mean_blen=0.30),  # the headroom regime
    "hard32": dict(n_taxa=32,  n_sites=300,  mean_blen=0.30),
    "hard48": dict(n_taxa=48,  n_sites=300,  mean_blen=0.40),
    "hard128": dict(n_taxa=128, n_sites=500,  mean_blen=0.40),
    "hard256": dict(n_taxa=256, n_sites=500,  mean_blen=0.40),
    "medium": dict(n_taxa=48,  n_sites=2000, mean_blen=0.10),
    "large":  dict(n_taxa=128, n_sites=5000, mean_blen=0.10),
}


def random_tree(n, rng, mean_blen=0.1):
    """A uniformly random unrooted topology with exponential branch lengths."""
    t = 1
    for leaf in range(2, n):
        edges = []
        def rec(node, path):
            edges.append(path)
            if not isinstance(node, int):
                rec(node[0], path + [0]); rec(node[1], path + [1])
        rec(t, [])
        path = edges[rng.integers(0, len(edges))]
        def ins(node, p):
            if not p:
                return (leaf, node)
            i = p[0]
            return (ins(node[0], p[1:]), node[1]) if i == 0 else (node[0], ins(node[1], p[1:]))
        t = ins(t, path)
    tree = from_nested((0, t), n)
    tree.blen = rng.exponential(mean_blen, size=len(tree.blen))
    tree.blen = np.clip(tree.blen, 0.005, 1.0)
    return tree


def simulate(tree, n_sites, rng):
    """Evolve sequences down the tree under JC69."""
    n_nodes = len(tree.left)
    seq = [None] * n_nodes
    seq[tree.root] = rng.integers(0, 4, size=n_sites)
    for u in reversed(tree.order):
        for c in (tree.left[u], tree.right[u]):
            t = tree.blen[c]
            p_change = 0.75 * (1.0 - np.exp(-4.0 * t / 3.0))
            s = seq[u].copy()
            m = rng.random(n_sites) < p_change
            s[m] = (s[m] + rng.integers(1, 4, size=int(m.sum()))) % 4
            seq[c] = s
    return np.array([seq[i] for i in range(tree.n)], dtype=np.int8)


def generate(tier: str, seed: int):
    p = TIERS[tier]
    rng = np.random.default_rng(seed)
    true_tree = random_tree(p["n_taxa"], rng, mean_blen=p.get("mean_blen", 0.1))
    data = simulate(true_tree, p["n_sites"], rng)
    return data, true_tree, dict(tier=tier, seed=seed, **p)
