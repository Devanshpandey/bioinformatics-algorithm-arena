# P01 — Haplotype long-match query

The first fully implemented Arena problem. Everything needed to score a
submission is here and runs on a laptop.

| file | role |
|---|---|
| `generate.py` | msprime coalescent instance generator, fresh seed per run |
| `oracle.py` | **exact** brute-force maximal-long-match enumeration — the ground truth |
| `baseline_pbwt.py` | the incumbent: PBWT prefix+divergence arrays, virtual-insertion query |
| `sub_rlpbwt_streaming.py` | **best submission**: run-length-encoded PBWT columns, no panel, 0.251 bits/cell |
| `run_stats.py` | where the bytes are: run structure of `a` vs `d` |
| `scaling_N.py` | query time vs panel size — establishes the scored regime |
| `evaluate.py` | gate 1 (exact set equality on fresh instances) then gate 2 (cost axes) |
| `problem.yaml` | machine-readable spec: tiers, metrics, beat targets, mutable surface |

Run it:

```bash
python3 evaluate.py --tier small --seed 42 --L 300 --submission baseline_pbwt
```

A submission is any module exposing `build(panel)` and `query(index, z, L)`.
It is scored only if it reproduces the oracle exactly on 25 freshly seeded
instances. Verified status: the PBWT baseline passes 40/40 randomized
differential tests and 25/25 gate instances.

Reference numbers (panel 980 × 9330, 20 queries, L=300, seed 42, this laptop):
build 53.3 s, query 0.247 s/query, index 8.0 bytes per panel cell = **64× the
bit-packed panel**. That last number is the most obvious headroom in the
problem and is the recommended first target.


## Current standing

| submission | index bits/cell | query s/q (small) | exact |
|---|---:|---:|:--:|
| `baseline_pbwt` | 72.0 (with panel) | 0.252 | yes |
| `sub_vectorized_pbwt` | 72.0 | 0.254 | yes |
| `sub_streaming_pbwt` | 8.0 (uint8 panel) | 0.211 | yes |
| **`sub_rlpbwt_streaming`** | **0.251** | **0.172** | yes |

`sub_rlpbwt_streaming` meets the space half of the beat target — 4× below the
bit-packed panel, 287× below the dense index, exact, and 1.052× faster
(CI [1.038, 1.075]).

**Two caveats that keep this from being a scored result.** The divergence array
is still recomputed per sweep rather than stored, so a single online query is
still O(N·M) and the real open problem is untouched. And per SPEC.md Finding 10,
at N ≤ 4,000 the no-index baseline dominates, so this panel size cannot
distinguish index quality at all — the scored tier is N ≥ 10,000, on the HPC node.
