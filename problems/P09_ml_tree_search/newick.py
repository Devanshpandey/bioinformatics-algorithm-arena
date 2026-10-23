"""Newick <-> arena Tree conversion, so OUR scorer evaluates every tree.

SPEC.md III: never trust a submission's self-reported objective. That applies
to incumbents too -- raxml-ng and iqtree report likelihoods under their own
model parameterisation, numerical conventions and rate heterogeneity settings.
The only apples-to-apples comparison is to parse their output tree and score it
with the same Felsenstein implementation used for every other submission.
"""
from __future__ import annotations

import re

import numpy as np

from phylo import Tree


def parse_newick(s: str, n_leaves: int) -> Tree:
    """Parse a Newick string whose taxa are named t0..t{n-1} into a Tree.

    Handles the unrooted trifurcating form that raxml-ng and iqtree emit, by
    re-rooting on the edge incident to leaf 0 (JC69 is reversible, so the
    likelihood is unchanged).
    """
    s = s.strip().rstrip(";").strip()
    pos = [0]

    def parse_node():
        if s[pos[0]] == "(":
            pos[0] += 1
            kids = []
            while True:
                kids.append(parse_node())
                if s[pos[0]] == ",":
                    pos[0] += 1
                    continue
                if s[pos[0]] == ")":
                    pos[0] += 1
                    break
            m = re.match(r"[^,\);]*", s[pos[0]:])
            label = m.group(0)
            pos[0] += len(label)
            return ("int", kids, _len(label))
        m = re.match(r"[^,\);]*", s[pos[0]:])
        tok = m.group(0)
        pos[0] += len(tok)
        name = tok.split(":")[0].strip()
        return ("leaf", int(name[1:]), _len(tok))

    def _len(tok):
        if ":" in tok:
            try:
                return float(tok.split(":")[-1])
            except ValueError:
                return 0.1
        return 0.1

    root = parse_node()

    # flatten into an undirected graph
    adj, nxt = {}, [0]

    def add(u, v, L):
        adj.setdefault(u, []).append((v, L))
        adj.setdefault(v, []).append((u, L))

    def walk(node):
        kind = node[0]
        if kind == "leaf":
            return ("L", node[1])
        me = ("I", nxt[0]); nxt[0] += 1
        for k in node[1]:
            c = walk(k)
            add(me, c, k[2])
        return me

    walk(root)

    # re-root on the edge incident to leaf 0
    start = ("L", 0)
    if start not in adj:
        raise ValueError("taxon t0 not found in Newick")
    nbr, edge_len = adj[start][0]

    left, right, blen = [-1] * n_leaves, [-1] * n_leaves, [0.1] * n_leaves

    def build_safe(u, parent):
        """Return (node_index, collapsed_extra_length).

        A degree-2 node (the artificial root of a bifurcating Newick, or any
        node with one child after excluding the parent) must COLLAPSE, and its
        edge length has to be added to the surviving edge. Dropping it silently
        changes the tree's total path lengths and therefore its likelihood --
        the round-trip test caught exactly that, a 9.2 log-unit discrepancy on a
        topologically identical tree.
        """
        if u[0] == "L":
            return u[1], 0.0
        kids = [(v, L) for (v, L) in adj[u] if v != parent]
        if len(kids) == 1:
            c, extra = build_safe(kids[0][0], u)
            return c, extra + kids[0][1]
        built = []
        for (v, L) in kids:
            c, extra = build_safe(v, u)
            while len(blen) <= c:
                blen.append(0.1)
            blen[c] = max(L + extra, 1e-9)
            built.append(c)
        cur = built[0]
        for c in built[1:]:
            left.append(cur); right.append(c); blen.append(1e-9)
            cur = len(left) - 1
        return cur, 0.0

    left, right, blen = [-1] * n_leaves, [-1] * n_leaves, [0.1] * n_leaves
    other, extra = build_safe(nbr, start)
    while len(blen) <= other:
        blen.append(0.1)
    blen[0] = max(edge_len, 1e-9)
    blen[other] = max(extra, 1e-9)
    left.append(0); right.append(other); blen.append(1e-9)
    root_idx = len(left) - 1
    return Tree(n_leaves, left, right, np.array(blen, dtype=float), root_idx)


def write_fasta(data: np.ndarray, path: str) -> None:
    bases = np.array(list("ACGT"))
    with open(path, "w") as fh:
        for i in range(data.shape[0]):
            fh.write(f">t{i}\n{''.join(bases[data[i]])}\n")
