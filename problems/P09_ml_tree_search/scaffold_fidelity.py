"""
P09 scaffold fidelity: our in-scaffold baseline vs the REAL incumbents.

SPEC.md II.2: "If the in-scaffold reimplementation of the incumbent is 5x slower
than the real tool, every win is an artifact of your own bad reference." P09 is
currently BLOCKED on exactly this. Its measured 11.74 log-unit headroom is over
our own NNI hill-climber, and until we know how that compares to raxml-ng and
iqtree the number means nothing.

Method: every tree -- ours, raxml-ng's, iqtree's -- is scored with the SAME
Felsenstein implementation (phylo.loglik) after parsing the incumbents' Newick
output. We never compare self-reported likelihoods across tools: they differ in
model parameterisation, rate heterogeneity defaults and numerical conventions,
so a raw lnL from raxml-ng is not commensurable with ours.

Outputs one JSON line per (tier, seed, tool) to fidelity.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

import phylo
from generate import generate
from newick import parse_newick, write_fasta
import baseline_search


def _which(*names):
    for n in names:
        p = shutil.which(n)
        if p:
            return n
    return None


def run_raxml(fasta, n_taxa, workdir, mode="--search1", seed=42, threads=1):
    exe = _which("raxml-ng")
    if not exe:
        return None
    prefix = os.path.join(workdir, "rx")
    cmd = [exe, mode, "--msa", fasta, "--model", "JC", "--threads", str(threads),
           "--seed", str(seed), "--prefix", prefix, "--redo"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    best = prefix + ".raxml.bestTree"
    if r.returncode != 0 or not os.path.exists(best):
        return {"tool": "raxml-ng", "error": (r.stderr or r.stdout)[-400:], "wall_s": dt}
    return {"tool": "raxml-ng", "mode": mode, "wall_s": dt,
            "tree": parse_newick(open(best).read(), n_taxa)}


def run_iqtree(fasta, n_taxa, workdir, seed=42, threads=1):
    exe = _which("iqtree3", "iqtree2", "iqtree")
    if not exe:
        return None
    prefix = os.path.join(workdir, "iq")
    cmd = [exe, "-s", fasta, "-m", "JC", "-T", str(threads), "-seed", str(seed),
           "--prefix", prefix, "-redo", "-quiet"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    tf = prefix + ".treefile"
    if not os.path.exists(tf):
        return {"tool": "iqtree", "error": (r.stderr or r.stdout)[-400:], "wall_s": dt}
    return {"tool": "iqtree", "wall_s": dt,
            "tree": parse_newick(open(tf).read(), n_taxa)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="small,medium")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--budget", type=float, default=60.0, help="budget for OUR baseline")
    ap.add_argument("--raxml-mode", default="--search1",
                    help="--search1 (one start) or --search (20 starts, the default users run)")
    ap.add_argument("--out", default="fidelity.jsonl")
    a = ap.parse_args()

    print(f"raxml-ng: {_which('raxml-ng') or 'NOT FOUND'}   "
          f"iqtree: {_which('iqtree2','iqtree') or 'NOT FOUND'}\n")
    print(f"{'tier':>7} {'seed':>5} {'tool':>10} {'lnL (our scorer)':>18} "
          f"{'wall_s':>9} {'RF→true':>8} {'Δ vs best':>10}")

    for tier in a.tiers.split(","):
        for seed in [int(x) for x in a.seeds.split(",")]:
            data, true_tree, meta = generate(tier, seed)
            n = meta["n_taxa"]
            rows = []
            with tempfile.TemporaryDirectory() as wd:
                fa = os.path.join(wd, "aln.fasta")
                write_fasta(data, fa)

                phylo.reset_counter()
                t0 = time.perf_counter()
                ours = baseline_search.search(data, budget_s=a.budget)
                rows.append({"tool": "arena-NNI", "wall_s": time.perf_counter() - t0,
                             "tree": ours, "clv_updates": phylo.CLV_UPDATES[0]})

                for fn in (lambda: run_raxml(fa, n, wd, a.raxml_mode, seed),
                           lambda: run_iqtree(fa, n, wd, seed)):
                    r = fn()
                    if r is not None:
                        rows.append(r)

            scored = []
            for r in rows:
                if "tree" not in r:
                    print(f"{tier:>7} {seed:>5} {r['tool']:>10} {'FAILED':>18} "
                          f"{r.get('wall_s',0):>9.2f}")
                    continue
                lnl = phylo.loglik(r["tree"], data)
                rf = phylo.rf_distance(r["tree"], true_tree)
                scored.append({**{k: v for k, v in r.items() if k != "tree"},
                               "lnL": lnl, "rf_to_true": rf, "tier": tier,
                               "seed": seed, "n_taxa": n, "n_sites": meta["n_sites"]})
            best = max((s["lnL"] for s in scored), default=float("nan"))
            for s in scored:
                s["lnL_deficit_vs_best"] = best - s["lnL"]
                print(f"{tier:>7} {seed:>5} {s['tool']:>10} {s['lnL']:>18.3f} "
                      f"{s['wall_s']:>9.2f} {s['rf_to_true']:>8} "
                      f"{s['lnL_deficit_vs_best']:>10.3f}")
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(s) + "\n")
            print()


if __name__ == "__main__":
    main()
