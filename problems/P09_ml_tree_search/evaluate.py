"""
P09 evaluation harness.

CORRECTNESS GATE -- the strongest one in the arena. On `gate`-tier instances
(6 taxa, 105 unrooted topologies) we EXHAUSTIVELY enumerate every topology,
optimise branch lengths on each, and take the maximum. That is the provable
global optimum of the search problem. A submission must reach within `eps`
log-units of it on every gate instance, or it is not scored at all.

Nothing else in the arena has this property: for P01 we can verify the answer,
but here we can verify OPTIMALITY. A search heuristic that silently returns
local optima cannot hide.

SCORE -- at fixed CPU budget: final log-likelihood (primary), CLV updates
consumed (the implementation-independent counter), and Robinson-Foulds distance
to the true tree (secondary; reported, never gated, because the ML tree is not
always the true tree).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "harness"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from measure import emit, instance_fingerprint, peak_rss_bytes, timed  # noqa: E402
import phylo  # noqa: E402
from generate import generate  # noqa: E402


def exhaustive_optimum(data, n):
    best, best_tree = None, None
    for t in phylo.enumerate_topologies(n):
        tr = phylo.from_nested(t, n)
        v = phylo.optimize_blens(tr, data, passes=2)
        if best is None or v > best:
            best, best_tree = v, tr
    return best, best_tree


def correctness_gate(mod, n_instances, base_seed, eps=0.5, verbose=True):
    for t in range(n_instances):
        seed = base_seed + t
        data, true_tree, meta = generate("gate", seed)
        opt, _ = exhaustive_optimum(data, meta["n_taxa"])
        tree = mod.search(data, budget_s=20.0)
        got = phylo.loglik(tree, data)
        if got < opt - eps:
            if verbose:
                print(f"GATE FAILED seed={seed}: reached {got:.3f}, "
                      f"global optimum {opt:.3f} (deficit {opt-got:.3f})")
            return False
        if verbose:
            print(f"  gate seed={seed}: lnL={got:.3f}  global opt={opt:.3f}  OK")
    if verbose:
        print(f"GATE PASSED: reached the exhaustive global optimum on "
              f"{n_instances} instances (eps={eps})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--submission", default="baseline_search")
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--gate-instances", type=int, default=3)
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--log", default="results.jsonl")
    a = ap.parse_args()

    mod = importlib.import_module(a.submission)

    if not a.skip_gate:
        if not correctness_gate(mod, a.gate_instances, a.seed + 1000):
            emit({"problem": "P09", "submission": a.submission, "gate": False}, a.log)
            sys.exit(1)

    data, true_tree, meta = generate(a.tier, a.seed)
    print(f"\ninstance: {meta['n_taxa']} taxa x {meta['n_sites']} sites, "
          f"budget {a.budget}s")

    phylo.reset_counter()
    tree, secs, _ = timed(mod.search, data, a.budget, repeats=1)
    clv = phylo.CLV_UPDATES[0]
    lnl = phylo.loglik(tree, data)
    rf = phylo.rf_distance(tree, true_tree)
    lnl_true = phylo.optimize_blens(true_tree.copy(), data, passes=2)

    rec = dict(problem="P09", submission=a.submission, gate=not a.skip_gate,
               tier=a.tier, seed=a.seed, n_taxa=meta["n_taxa"],
               n_sites=meta["n_sites"], budget_s=a.budget, wall_s=secs,
               final_lnL=lnl, lnL_at_true_tree=lnl_true,
               lnL_minus_true=lnl - lnl_true, rf_to_true=rf,
               max_rf=2 * (meta["n_taxa"] - 3), clv_updates=clv,
               peak_rss_bytes=peak_rss_bytes(),
               instance=instance_fingerprint(data, tier=a.tier, seed=a.seed))
    emit(rec, a.log)
    print(json.dumps(rec, indent=2, default=float))


if __name__ == "__main__":
    main()
