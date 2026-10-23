"""
What does the index actually BUY? Query time versus panel size N.

PBWT's whole promise is a query cost independent of N after an O(NM) build.
The streaming submission deletes the index entirely (0 bytes) but regenerates
the arrays during the sweep, which costs O(N) per site. If that promise holds,
the two submissions sit on genuinely different curves and the index earns its
bytes; if it does not, the index is dead weight and the target changes.

Measured here rather than assumed, because the whole P01 beat-target rests on it.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import msprime

import baseline_pbwt
import sub_vectorized_pbwt
import sub_streaming_pbwt

RHO, MU, NE = 1e-8, 1.5e-8, 10_000


def make_panel(n_hap, seq_len, seed):
    ts = msprime.sim_ancestry(samples=n_hap // 2, ploidy=2, sequence_length=seq_len,
                              recombination_rate=RHO, population_size=NE,
                              random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1,
                               model=msprime.BinaryMutationModel())
    G = ts.genotype_matrix().T.astype(np.uint8)
    return G[8:], G[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="250,500,1000,2000,4000,8000")
    ap.add_argument("--seq-len", type=float, default=2e6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--out", default="scaling_N.jsonl")
    a = ap.parse_args()

    print(f"{'N':>7} {'M':>7} {'build_s':>9} {'idx_MB':>8} {'bits/cell':>10} "
          f"{'PBWT q/q':>10} {'stream q/q':>11} {'matches':>9}")
    for n in [int(x) for x in a.sizes.split(",")]:
        panel, qh = make_panel(n, a.seq_len, a.seed)
        N, M = panel.shape

        t0 = time.perf_counter()
        idx = sub_vectorized_pbwt.build(panel)
        build_s = time.perf_counter() - t0
        ib = idx.nbytes()

        t0 = time.perf_counter()
        nm = sum(len(baseline_pbwt.query(idx, qh[i], a.L)) for i in range(qh.shape[0]))
        q_pbwt = (time.perf_counter() - t0) / qh.shape[0]

        sidx = sub_streaming_pbwt.build(panel)
        t0 = time.perf_counter()
        nm2 = sum(len(m) for m in sub_streaming_pbwt.query_batch(sidx, qh, a.L))
        q_stream = (time.perf_counter() - t0) / qh.shape[0]
        assert nm == nm2, f"submissions disagree: {nm} vs {nm2}"

        row = dict(N=N, M=M, build_s=build_s, index_bytes=ib,
                   index_bits_per_cell=ib * 8.0 / (N * M),
                   q_pbwt_s=q_pbwt, q_stream_s=q_stream, n_matches=nm,
                   n_query=int(qh.shape[0]), L=a.L, seed=a.seed)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{N:>7} {M:>7} {build_s:>9.2f} {ib/1e6:>8.1f} "
              f"{ib*8.0/(N*M):>10.2f} {q_pbwt:>10.4f} {q_stream:>11.4f} {nm:>9,}")

    print("\nFit log(q) = a + b*log(N):")
    rows = [json.loads(l) for l in open(a.out)]
    rows = [r for r in rows if r["seed"] == a.seed]
    if len(rows) >= 3:
        x = np.log(np.array([r["N"] for r in rows]))
        for key, label in (("q_pbwt_s", "stored-index PBWT"), ("q_stream_s", "streaming")):
            y = np.log(np.array([r[key] for r in rows]))
            b, aa = np.polyfit(x, y, 1)
            print(f"  {label:>18}: exponent b = {b:+.3f}   "
                  f"({'N-independent' if abs(b) < 0.25 else 'grows with N'})")


if __name__ == "__main__":
    main()
