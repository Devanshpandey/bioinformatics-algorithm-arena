"""
P01 instance generator: coalescent haplotype panels with fresh seeds.

Why simulate rather than download 1000G: (a) the instance is generated fresh
per seed, so no public answer key can be memorized and no instance-specific
constant can be tuned; (b) panel size scales to the large tier without a
data-access agreement; (c) msprime's coalescent-with-recombination produces
exactly the long shared-haplotype structure the algorithm must exploit.

Instances are written as an .npz holding:
    panel : uint8 (N, M)  reference haplotypes
    query : uint8 (Q, M)  query haplotypes (held out of the panel)
    meta  : json blob (seed, params, msprime version)

Usage:
    python3 generate.py --tier small  --seed 42 --out inst_small_42.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# (n_haplotypes_total, sequence_length_bp, recombination_rate, mutation_rate, Ne, n_query)
TIERS = {
    "tiny":   dict(n=60,     L=2e5, rho=1e-8, mu=1.5e-8, Ne=10_000, q=5),
    "small":  dict(n=1_000,  L=2e6, rho=1e-8, mu=1.5e-8, Ne=10_000, q=20),
    "medium": dict(n=10_000, L=1e7, rho=1e-8, mu=1.5e-8, Ne=10_000, q=50),
    # Scored tiers. Below N=10,000 the no-index baseline dominates and the
    # benchmark cannot distinguish index quality (SPEC.md Finding 10).
    "n10k":   dict(n=10_000,  L=2e7, rho=1e-8, mu=1.5e-8, Ne=10_000, q=16),
    "n30k":   dict(n=30_000,  L=2e7, rho=1e-8, mu=1.5e-8, Ne=10_000, q=16),
    "large":  dict(n=100_000, L=2e7, rho=1e-8, mu=1.5e-8, Ne=10_000, q=16),
}


def generate(tier: str, seed: int):
    import msprime

    p = TIERS[tier]
    ts = msprime.sim_ancestry(
        samples=p["n"] // 2,
        ploidy=2,
        sequence_length=p["L"],
        recombination_rate=p["rho"],
        population_size=p["Ne"],
        random_seed=seed,
    )
    # BinaryMutationModel is REQUIRED, not cosmetic. msprime's default model
    # produces multi-allelic sites (~0.18% of sites at these parameters), and a
    # PBWT column update splits on allele==0 vs everything else -- which merges
    # alleles 1 and 2, corrupts the divergence arrays, and makes the index emit
    # spurious matches. Measured before this fix: 5 spurious + 1 missing out of
    # 3245 true matches. The panel MUST be biallelic for the problem as stated.
    ts = msprime.sim_mutations(ts, rate=p["mu"], random_seed=seed + 1,
                               model=msprime.BinaryMutationModel())
    G = ts.genotype_matrix().T.astype(np.uint8)  # (n_hap, n_sites)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(G.shape[0])
    q = p["q"]
    query, panel = G[idx[:q]], G[idx[q:]]
    meta = dict(tier=tier, seed=seed, msprime=msprime.__version__,
                n_panel=int(panel.shape[0]), n_query=int(query.shape[0]),
                n_sites=int(G.shape[1]), params={k: float(v) for k, v in p.items()})
    return panel, query, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small", choices=list(TIERS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    panel, query, meta = generate(a.tier, a.seed)
    np.savez_compressed(a.out, panel=panel, query=query, meta=json.dumps(meta))
    print(f"{a.out}: panel {panel.shape}, query {query.shape}, "
          f"sites={meta['n_sites']}, seed={a.seed}")
