# Bioinformatics Algorithm Arena

A verifier-first testbed for **agentic algorithm discovery in genomics**. LLM agents
propose new algorithms and data structures for open problems in population and
statistical genetics. A sealed evaluator then scores every proposal against an exact
oracle, so a proposal counts only if it is provably correct and measurably better.

The design follows the FunSearch / AlphaEvolve pattern (a proposer searches, a fixed
evaluator decides), applied where biology supplies large, structured inputs and the
current solution is an algorithmic compromise. Most of the work turned out to be on
the evaluator side: deciding what to measure, at what scale, and how to stop a
proposer (or ourselves) from being credited for an artifact. Those lessons are
written up as numbered findings in [`docs/SPEC.md`](docs/SPEC.md).

> **Selection criterion.** Don't look for biological prediction tasks. Look for
> computational primitives where biology supplies enormous structured datasets, the
> current solution is a compromise, and a machine can *check* whether it did better.

## Problems

| | Problem | Ground truth | Scored objective | Status |
|---|---|---|---|---|
| [**P01**](problems/P01_haplotype_long_match) | Haplotype long-match query (PBWT index) | brute-force exact enumeration of all long maximal matches | index bits per panel cell (space); single-query latency (online) | active; agent campaign run |
| [**P02**](problems/P02_ibd_detection) | IBD segment detection | exact genealogical IBD from the simulated tree sequence (`tskit`) | cM recall/precision relative to the incumbent | evaluator and incumbent built |
| [**P09**](problems/P09_ml_tree_search) | Maximum-likelihood phylogeny search | provable optimum by exhaustive enumeration (gate); exact log-likelihood (score) | log-likelihood at matched CPU, n_taxa ≥ 128 | evaluator and baselines built; tools diverge at ≥ 128 taxa |

All instances are **simulated from a fresh seed** (msprime coalescent panels, or
alignments along a known random tree), so there is no answer key to memorise and no
data-access barrier. No controlled-access data is used anywhere in this repository.

## Results so far

### P01: an agent search beat a strong incumbent, and the cheap tier ranked it wrong

The incumbent is a run-length-encoded PBWT that never stores the panel
([`docs/RLPBWT.md`](docs/RLPBWT.md)). Six independent agent "islands" proposed
compressed index representations. The sealed evaluator scored each one, and a separate
skeptic agent then audited each survivor for cheats. **All six were exact and all six
beat the incumbent at every tier.**

| module | small (N=980) | n10k | **n30k** | gain at n30k | rank small → n30k |
|---|---:|---:|---:|---:|:--:|
| `sub_column_diff` | 0.02592 | 0.00375 | **0.00158** | **7.07×** | 2 → **1** |
| `sub_bitvector_rank` | 0.02737 | 0.00414 | 0.00174 | 6.41× | 3 → 2 |
| `sub_freeform` | 0.02567 | 0.00395 | 0.00190 | 5.87× | **1** → 3 |
| `sub_delta_varint` | 0.03529 | 0.00563 | 0.00234 | 4.76× | 4 → 4 |
| `sub_bitpack_fixed` | 0.03595 | 0.00569 | 0.00234 | 4.76× | 5 → 5 |
| `sub_eliasfano` | 0.03605 | 0.00569 | 0.00238 | 4.68× | 6 → 6 |
| *incumbent* `sub_rlpbwt_streaming` | 0.25097 | 0.02928 | 0.01115 | — | — |

Units are bits per panel cell. At n30k the best index needs one bit per ~633 panel
cells, 633× less than the bit-packed panel, and it still answers queries exactly. The top three
small-tier candidates also beat `xz -9e` on the bit-packed panel (0.0323 bits/cell),
which stores the data but cannot be queried.

Two things this table does **not** show, and that the arena was redesigned to expose:

- **Scoring only where it is cheap mis-ranks.** The small-tier winner finishes third at
  n30k. A leaderboard built at the convenient scale would have crowned the wrong method.
- **Space is not the whole problem.** Smaller indexes cost query time. In the batch
  workload the space champion `sub_column_diff` answers a query in **0.449 s vs 0.171 s**
  for the incumbent, **2.6× slower** (table below). A second sealed scorer
  ([`score_online.py`](problems/P01_haplotype_long_match/score_online.py)) measures
  single-query latency with the index built once, and there the gap widens to
  **~20×** (30.1 s vs 1.47 s at fixed M=3,000, N up to 16k;
  [`docs/RLPBWT.md` §11](docs/RLPBWT.md)), because arithmetic coding only decodes
  sequentially. The open target remains storing
  the PBWT divergence array in o(NM) space with O(1) access.

Batch query time at the small tier (N=980 haplotypes, 9,330 sites, 20 queries, L=300),
median over every sealed evaluation in
[`search_log.jsonl`](problems/P01_haplotype_long_match/search_log.jsonl):

| module | median s/query | vs incumbent | evaluations |
|---|---:|---:|---:|
| *incumbent* `sub_rlpbwt_streaming` | 0.171 | 1.00× | 9 |
| `sub_eliasfano` | 0.174 | 1.01× | 15 |
| `sub_delta_varint` | 0.187 | 1.09× | 11 |
| `sub_bitpack_fixed` | 0.193 | 1.13× | 14 |
| `sub_bitvector_rank` | 0.299 | 1.74× | 23 |
| `sub_freeform` | 0.431 | 2.51× | 20 |
| `sub_column_diff` | 0.449 | 2.62× | 17 |

These are laptop wall-clock times. Differences under ~1.2× are within that machine's
measured noise floor ([`docs/SPEC.md`](docs/SPEC.md), Tier-0 calibration), so the first
three candidates are indistinguishable from the incumbent on batch speed.

### P09: the search method matters, but only at scale

Measured against our own NNI baseline, P09 looked like it had 11.7 log-units of
headroom. Against the real incumbents (RAxML-NG 2.0.2 and IQ-TREE 3.1.3, every tree
re-scored with our own Felsenstein pruning) that gap disappeared below ~50 taxa,
because the two tools agree to within 0.31 log-units. At 128 and 256 taxa they diverge.
Once CPU budget is matched, the median gap is **6.1 and 162 log-units**, and RAxML-NG
finds the higher-likelihood tree in 5 of 6 runs.

This is a baseline comparison, run before any agent search. It demonstrates sensitivity
to search method under matched compute budgets; it does not show that either tool misses
the global optimum. P09 is scored at n_taxa ≥ 128, where the choice of search method
changes the result.

## What the evaluator got wrong, and how we found out

The proposer was rarely the weak link. These are the failures that mattered, each
recorded with its fix in [`docs/SPEC.md`](docs/SPEC.md):

- **A gate that certified nothing.** The correctness gate passed on synthetic binary
  instances while the *scored* coalescent instances contained multi-allelic sites. The
  index silently returned 5 spurious and 1 missing match per 3,245. Fix: pin
  `BinaryMutationModel`, assert preconditions in the oracle, and gate on instances drawn
  from the scored distribution. (Finding 5)
- **Wall-clock cannot tell an algorithm from an implementation.** A numpy rewrite of
  the same algorithm posted a 79× build "win" with no algorithmic content. Fix: two
  leaderboards, one ranked on machine-independent counters. (Finding 2)
- **Noise floors.** Identical code posted a 5.8% apparent speedup on a laptop
  (CI 0.98–1.11), so speedups under ~1.2× are not claimable there (Tier-0
  calibration). The size of a real engineering win is machine-dependent too:
  85× on the laptop, 12.4× on the HPC node. (Finding 9)
- **Our own baseline cheated.** `nbytes()` on the streaming baseline returned 0 while
  holding the panel alive, the exact accounting cheat the auditors were told to hunt.
  The auditors found it. Fix: audit the incumbent with the tooling you point at
  submissions. (Finding 14)
- **Trust-on-first-use lock.** A missing `immutable.lock` used to be silently recreated,
  so deleting the lock, weakening the oracle and re-running would have passed. A
  missing lock is now a hard refusal. (Result 3; found by the auditor agents)
- **Comparing the wrong problems.** "All long maximal matches", "SMEM" and "k-SMEM" are
  different problems, and about half the apparent PBWT state of the art solves one of
  the other two. We fell into this once ourselves and retracted the claim. (Finding 10)
- **Instances are not reproducible across machines.** The same seed gives different
  panels on macOS and on the Linux cluster, even at matched msprime/tskit versions. Every result row now carries an instance
  fingerprint (content SHA plus generator versions). (Finding 8)

## How a submission is scored

1. **Sealed evaluator.** The oracle, generator, PBWT core and scorers are listed with
   SHA-256 hashes in `immutable.lock` / `immutable_online.lock`. The scorer refuses to
   run if any hash differs or the lock is missing, and git history is the baseline the
   lock is checked against.
2. **Gate 1: exactness.** The submission must reproduce the oracle's output byte for
   byte on freshly seeded instances drawn from the scored distribution. Correctness is
   not a tradeable axis.
3. **Gate 2: cost.** Index bytes (measured, not self-reported), build time, query time
   and peak RSS, with thread count set by the harness rather than the candidate.
4. **Audit.** Each survivor is attacked by an independent skeptic: object-graph walks
   for retained arrays, destroying the panel in place and re-querying, bit-for-bit
   panel reconstruction, and hundreds of randomized differential tests.

A P01 submission is any module exposing `build(panel)` and `query(index, z, L)`.

## Quickstart

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate arena

cd problems/P01_haplotype_long_match
python score_submission.py --module sub_rlpbwt_streaming --json         # incumbent
python score_submission.py --module candidates.sub_column_diff --json   # best agent candidate
python evaluate.py --tier small --seed 42 --L 300 --submission baseline_pbwt

cd ../P02_ibd_detection && python evaluate.py --tier small --seed 42    # ~1 min
cd ../P09_ml_tree_search && python evaluate.py --tier small --seed 42 --budget 25
```

The small tiers run on a laptop. The scored tiers (P01 at N ≥ 10,000, P09 at
n_taxa ≥ 128) need a large-memory node; SLURM templates are in [`cluster/`](cluster).
Pin BLAS/OpenMP threads to 1 (see `cluster/activate.sh`); the harness assumes it.

## Layout

```
problems/   P01, P02, P09: generator, oracle, incumbent, evaluator, problem.yaml
            P01/candidates/   the six agent-proposed indexes
            P01/search_log.jsonl   every sealed evaluation from the campaign
harness/    resource measurement, calibration, instance fingerprinting
docs/       SPEC.md (design and numbered findings), RLPBWT.md (method report),
            design/ (scoring protocol, contamination and gaming threat model),
            specs/ (per-problem design specs, written before implementation)
cluster/    SLURM templates and environment activation for the scored tiers
tools/      instance-hash check across msprime/tskit versions
```

## About the agents

The proposer and auditor agents were LLM coding agents (Claude Code subagents),
orchestrated outside this repository. What is here is everything needed to check their
work: the sealed evaluators, the incumbent, the six candidates exactly as proposed, and
the log of every evaluation. Nothing in a candidate is trusted; it either passes the
gates or it does not score.

The design specs in `docs/specs/` were written before implementation, so a few file
names and flags there differ from the code in `problems/`.

## Author

Devansh Pandey, The University of Texas at Austin ([website](https://devanshpandey.github.io)).

## License

MIT. See [LICENSE](LICENSE).
