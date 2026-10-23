"""
Where do P01's bytes actually go?

The literature audit claimed the space is ALL in the divergence array and none
in the permutation, and quoted r = 50,712 allele-column runs on our own small
tier. That measurement came from a scratch script we no longer have, so it is
re-derived here, in-repo and reproducible, before any of it is used to justify
a design. Arena rule: a number you cannot regenerate is not a number.

Reports, per instance:
  r      total runs in the PBWT allele columns (what an RLPBWT pays for)
  r_d    total runs in the divergence arrays  (what RLE would pay for `d`)
  and the bits/cell each implies, against the bit-packed panel as the yardstick.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from generate import generate
from pbwt_core import pbwt_step


def run_structure(panel: np.ndarray):
    N, M = panel.shape
    cur_a = np.arange(N, dtype=np.int32)
    cur_d = np.zeros(N, dtype=np.int32)
    r_total = 0          # runs in the allele column, in PBWT order
    rd_total = 0         # runs in the divergence array
    per_col_runs = []
    for k in range(M):
        alleles = panel[cur_a, k]
        rc = 1 + int(np.count_nonzero(alleles[1:] != alleles[:-1])) if N else 0
        r_total += rc
        per_col_runs.append(rc)
        cur_a, cur_d = pbwt_step(cur_a, cur_d, alleles, k)
        rd = 1 + int(np.count_nonzero(cur_d[1:] != cur_d[:-1])) if N else 0
        rd_total += rd
    return r_total, rd_total, np.array(per_col_runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="small")
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--out", default="run_stats.jsonl")
    a = ap.parse_args()

    print(f"{'tier':>7} {'seed':>5} {'N':>7} {'M':>7} {'cells':>12} "
          f"{'r':>9} {'r/col':>7} {'r_d':>11} {'r_d/cell':>9}")
    for tier in a.tiers.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            panel, qh, meta = generate(tier, seed)
            N, M = panel.shape
            cells = N * M
            r, rd, per_col = run_structure(panel)
            row = dict(tier=tier, seed=seed, N=N, M=M, cells=cells,
                       r=r, runs_per_col=r / M, r_d=rd, rd_per_cell=rd / cells,
                       # what each representation would cost, in bits per cell
                       dense_a_d_bits_per_cell=8 * 8.0,        # our int32 a + int32 d
                       bitpacked_panel_bits_per_cell=1.0,
                       rlpbwt_a_bits_per_cell=r * 64.0 / cells,     # 1 word per run
                       rlpbwt_a_2word_bits_per_cell=r * 128.0 / cells,
                       rle_d_bits_per_cell=rd * 64.0 / cells)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"{tier:>7} {seed:>5} {N:>7} {M:>7} {cells:>12,} "
                  f"{r:>9,} {r/M:>7.2f} {rd:>11,} {rd/cells:>9.3f}")
            print(f"{'':>7} {'':>5}   bits/cell -> dense a+d {8*8.0:>6.1f} | "
                  f"bit-packed panel {1.0:>4.1f} | RLPBWT a (1 word/run) "
                  f"{r*64.0/cells:>5.3f} | RLE of d {rd*64.0/cells:>6.2f}")


if __name__ == "__main__":
    main()
