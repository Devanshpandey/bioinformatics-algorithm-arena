"""
P02 instance generator: coalescent panels WITH exact genealogical IBD truth.

The tree sequence knows, for every pair of haplotypes and every interval, the
most recent common ancestor. tskit exposes this directly, so the truth set is
EXACT BY CONSTRUCTION -- no curated answer key, nothing an LLM could have
memorized, and unlimited fresh instances.

Truth definition (this is the load-bearing design choice; see problem.yaml):
    A pair (i, j) is IBD on [left, right) iff their MRCA over that whole
    interval is a single node at time <= MAX_TIME generations, the segment was
    inherited along one unbroken genealogical path, and the segment is at least
    MIN_CM long.

`max_time` is what makes the definition meaningful: without it, every pair of
haplotypes is IBD everywhere (they always share SOME ancestor), and the truth
set is both enormous and useless. With it, "IBD" means "recent enough to be
detectable from haplotype sharing", which is what IBD callers actually claim.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

RHO = 1e-8      # per-bp per-generation recombination rate -> 1 Mb == 1 cM
MU = 1.5e-8

# Demography: recent exponential growth over an ancestral population.
# This matters. Manufacturing IBD density by shrinking Ne (e.g. Ne=1000) works
# -- it yields ~2500 segments instead of ~250 -- but it distorts the haplotype
# structure the algorithm has to exploit, so a method tuned there would not
# transfer to real cohorts. Recent growth is WHY real human cohorts carry
# abundant recent IBD, so we buy density from the demography that actually
# produces it, and pay for it by needing a larger panel.
N_NOW, GROWTH, N_ANC, T_ANC = 500_000, 0.05, 10_000, 200

TIERS = {
    "tiny":   dict(n=60,     L=5e6),
    "small":  dict(n=600,    L=2e7),
    "medium": dict(n=2_000,  L=5e7),
    "large":  dict(n=10_000, L=1e8),
}


def _demography():
    import msprime
    d = msprime.Demography()
    d.add_population(name="pop", initial_size=N_NOW, growth_rate=GROWTH)
    d.add_population_parameters_change(time=T_ANC, population="pop",
                                       growth_rate=0, initial_size=N_ANC)
    return d

MAX_TIME = 100.0   # generations; ~ the detectability horizon for IBD callers
MIN_CM = 2.0       # segments shorter than this are not in the truth set


def bp_to_cm(bp: float) -> float:
    return 100.0 * RHO * bp


def cm_to_bp(cm: float) -> float:
    return cm / (100.0 * RHO)


def generate(tier: str, seed: int, max_time: float = MAX_TIME, min_cm: float = MIN_CM):
    import msprime

    p = TIERS[tier]
    ts = msprime.sim_ancestry(
        samples=p["n"] // 2, ploidy=2, sequence_length=p["L"],
        recombination_rate=RHO, demography=_demography(), random_seed=seed,
    )
    # BinaryMutationModel is REQUIRED, not cosmetic. msprime's default model
    # produces multi-allelic sites (~0.18% of sites at these parameters), and a
    # PBWT column update splits on allele==0 vs everything else -- which merges
    # alleles 1 and 2, corrupts the divergence arrays, and makes the index emit
    # spurious matches. Measured before this fix: 5 spurious + 1 missing out of
    # 3245 true matches. The panel MUST be biallelic for the problem as stated.
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1,
                               model=msprime.BinaryMutationModel())

    G = ts.genotype_matrix().T.astype(np.uint8)          # (n_hap, n_sites)
    pos = ts.sites_position.astype(np.float64)           # bp of each site

    segs = ts.ibd_segments(
        max_time=max_time, min_span=cm_to_bp(min_cm),
        store_pairs=True, store_segments=True,
    )
    truth = []
    for pair, seglist in segs.items():
        u, v = int(pair[0]), int(pair[1])
        if u >= G.shape[0] or v >= G.shape[0]:
            continue  # only sample nodes are haplotypes in the panel
        a, b = (u, v) if u < v else (v, u)
        for s in seglist:
            truth.append((a, b, float(s.left), float(s.right)))
    truth = np.array(sorted(set(truth)), dtype=np.float64) if truth else np.zeros((0, 4))

    meta = dict(tier=tier, seed=seed, n_hap=int(G.shape[0]), n_sites=int(G.shape[1]),
                seq_len=float(p["L"]), max_time=max_time, min_cm=min_cm,
                rho=RHO, mu=MU, N_now=N_NOW, growth=GROWTH, N_anc=N_ANC,
                n_truth=int(truth.shape[0]))
    return G, pos, truth, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="small", choices=list(TIERS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    G, pos, truth, meta = generate(a.tier, a.seed)
    np.savez_compressed(a.out, panel=G, pos=pos, truth=truth, meta=json.dumps(meta))
    tot = sum(bp_to_cm(r - l) for _, _, l, r in truth) if len(truth) else 0.0
    print(f"{a.out}: panel {G.shape}, {len(truth)} true IBD segments, "
          f"{tot:.0f} cM total, seed={a.seed}")
