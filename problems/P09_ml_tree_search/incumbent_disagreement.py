"""
Does the ML tree-search problem have ANY headroom at scale?

The scaffold-fidelity run settled the small case: raxml-ng and iqtree both find
essentially the same optimum (within 0.02-0.31 log-units) at 16-48 taxa, while
our NNI baseline trails by 0.3-20 log-units. So the "headroom" measured against
our own baseline was our baseline being weak, not the problem being open.

The remaining question is whether the two SOTA tools DIVERGE at larger taxon
counts. If they do, the gap between them is genuine, tool-independent headroom
and P09 survives as a scored problem at that scale. If they keep agreeing, P09
is saturated everywhere we can afford to evaluate and should be demoted.

This probe runs only the incumbents -- our baseline is too slow to matter here
and its result is already known.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import phylo
from generate import generate
from newick import write_fasta
from scaffold_fidelity import run_raxml, run_iqtree

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="hard32,hard48,hard128")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--raxml-mode", dest="raxml_mode", default="--search")
    ap.add_argument("--out", default="disagreement.jsonl")
    a = ap.parse_args()

    print(f"{'tier':>8} {'taxa':>5} {'seed':>5} {'raxml lnL':>14} {'iqtree lnL':>14} "
          f"{'gap':>9} {'rax_s':>8} {'iq_s':>8} {'RF(rax,iq)':>11}")
    for tier in a.tiers.split(","):
        for seed in [int(x) for x in a.seeds.split(",")]:
            data, true_tree, meta = generate(tier, seed)
            n = meta["n_taxa"]
            with tempfile.TemporaryDirectory() as wd:
                fa = os.path.join(wd, "aln.fasta")
                write_fasta(data, fa)
                r = run_raxml(fa, n, wd, a.raxml_mode, seed)
                q = run_iqtree(fa, n, wd, seed)
            if not r or "tree" not in r or not q or "tree" not in q:
                print(f"{tier:>8} {n:>5} {seed:>5}   FAILED")
                continue
            lr, lq = phylo.loglik(r["tree"], data), phylo.loglik(q["tree"], data)
            rf = phylo.rf_distance(r["tree"], q["tree"])
            row = dict(tier=tier, n_taxa=n, n_sites=meta["n_sites"], seed=seed,
                       raxml_lnL=lr, iqtree_lnL=lq, gap=abs(lr - lq),
                       raxml_s=r["wall_s"], iqtree_s=q["wall_s"], rf_raxml_iqtree=rf,
                       rf_raxml_true=phylo.rf_distance(r["tree"], true_tree),
                       rf_iqtree_true=phylo.rf_distance(q["tree"], true_tree))
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"{tier:>8} {n:>5} {seed:>5} {lr:>14.3f} {lq:>14.3f} {abs(lr-lq):>9.3f} "
                  f"{r['wall_s']:>8.2f} {q['wall_s']:>8.2f} {rf:>11}")
