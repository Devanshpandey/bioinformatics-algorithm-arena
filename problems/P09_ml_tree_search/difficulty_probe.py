"""
Is P09 saturated? The headroom test.

A search problem only has headroom if the incumbent heuristic demonstrably
FAILS to find the global optimum. We cannot enumerate at realistic sizes, so we
use the standard proxy: run the same hill-climber from many independent starts
and measure the spread of final log-likelihoods. If every start converges to the
same value, the surface is easy and the problem is saturated at that setting.
If starts disagree by many log-units, the incumbent is provably getting stuck,
and the gap between its typical result and the best-of-N is real headroom.

This is exactly the evidence an arena must have BEFORE spending compute on a
problem, and it is cheap to obtain.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import phylo
from baseline_search import search, jc_distance, neighbor_joining
from generate import random_tree, simulate


def random_start_search(data, rng, budget_s):
    n = data.shape[0]
    tree = random_tree(n, rng)
    best = phylo.optimize_blens(tree, data, passes=2)
    t0 = time.time()
    improved = True
    from baseline_search import _nni_variants
    while improved and time.time() - t0 < budget_s:
        improved = False
        for e in tree.internal_edges():
            if time.time() - t0 > budget_s:
                break
            for cand in _nni_variants(tree, e):
                v = phylo.loglik(cand, data)
                if v > best + 1e-6:
                    v = phylo.optimize_blens(cand, data, passes=1)
                    if v > best + 1e-6:
                        tree, best, improved = cand, v, True
                        break
    return phylo.loglik(tree, data), tree


if __name__ == "__main__":
    n_starts = 6
    print(f"{'taxa':>5} {'sites':>6} {'blen':>5} {'NJ+NNI lnL':>13} {'best random':>13} "
          f"{'spread':>8} {'NJ deficit':>11} {'RF(NJ,true)':>12}")
    for n_taxa, n_sites, mb in [(16, 1000, 0.10), (16, 200, 0.30),
                                (32, 300, 0.30), (48, 300, 0.40)]:
        rng = np.random.default_rng(7)
        true_tree = random_tree(n_taxa, rng, mean_blen=mb)
        data = simulate(true_tree, n_sites, rng)

        lnl_nj = phylo.loglik(search(data, budget_s=45.0), data)
        tree_nj = search(data, budget_s=45.0)
        rf_nj = phylo.rf_distance(tree_nj, true_tree)

        vals = []
        for s in range(n_starts):
            v, _ = random_start_search(data, np.random.default_rng(100 + s), 45.0)
            vals.append(v)
        vals = np.array(vals)
        best_rand = vals.max()
        print(f"{n_taxa:>5} {n_sites:>6} {mb:>5.2f} {lnl_nj:>13.2f} {best_rand:>13.2f} "
              f"{vals.max()-vals.min():>8.2f} {max(best_rand-lnl_nj,0):>11.2f} {rf_nj:>12}")
