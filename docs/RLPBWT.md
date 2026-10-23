# Run-Length PBWT for exact long-match queries
### Method and results · Bioinformatics Algorithm Arena, problem P01 · 2026-09-02

---

## 1. The problem

Let `X ∈ {0,1}^{N×M}` be a panel of `N` haplotypes over `M` biallelic sites in a
fixed site order, `Z ∈ {0,1}^{Q×M}` a set of query haplotypes disjoint from `X`,
and `L` a length threshold in sites. Report

```
LM(X, Z, L) = { (i, j, s, e) :
      Z[i][s:e] == X[j][s:e]                  (match)
      e - s >= L                              (long)
      s == 0  or  Z[i][s-1] != X[j][s-1]      (left-maximal)
      e == M  or  Z[i][e]   != X[j][e]   }    (right-maximal)
```

This is the **all long maximal matches** problem — Durbin (2014) Algorithm 5,
the "long LEMs" of Sanaullah, Zhi & Zhang (WABI 2025).

> **Disambiguation, which is load-bearing.** *All long maximal matches*,
> *set-maximal / SMEM*, and *k-SMEM / KL-SMEM* are **different problems**, and
> roughly half of the apparent PBWT state of the art solves one of the other
> two. μ-PBWT is SMEM-only; PBML answers KL-SMEMs. Any timing or space number
> compared against the literature must first establish that the other tool
> solves the same problem. We violated this ourselves once (§7.2).

**Objective:** minimise index bits per panel cell at **fixed exact
correctness**. Correctness is not a tradeable axis — a candidate that is not
byte-exact against a brute-force oracle scores nothing at all.

---

## 2. Baseline: the dense PBWT

Durbin's positional Burrows–Wheeler transform maintains, for each site `k`, a
prefix array `a_k` (haplotypes sorted by reversed prefix) and a divergence array
`d_k`, where `d_k[j]` is the position at which `a_k[j]` and `a_k[j-1]` begin to
match. Long-match query proceeds by *virtual insertion*: sweep `k = 0…M`,
tracking where the query would sit in `a_k` and the divergence to its neighbours
above and below, reporting matches as they terminate.

Storing `a` and `d` as `int32` for every site costs

```
8 bytes/cell = 64 bits/cell
```

against a **1 bit/cell** bit-packed panel — a 64× overhead. At the arena's small
tier (980 × 9,330) that is 73 MB of index for a 1.1 MB panel.

---

## 3. The structural observation

> During a left-to-right sweep, **every** panel access the algorithm makes
> reduces to `alleles[j]` — the allele of the haplotype at position `j` of the
> *current PBWT column*.

The baseline computes that as `panel[cur_a, k]`, which is why it must retain the
panel. But the PBWT column *in `a_k` order* is exactly the run-length-structured
object the RLPBWT literature stores. So we can hold the runs and decode column
`k` on the fly; the panel is then never touched at query time at all.

`a_k` and `d_k` are not stored either — `pbwt_step` regenerates them column by
column from the decoded alleles, in O(N) per site, which the sweep pays anyway.

This yields **`sub_rlpbwt_streaming`**: per column, the run start positions as
`int32`, a per-column offset array, and one byte for the first run's allele.

---

## 4. Measured run structure

`run_stats.py`, reproducible in-repo:

| tier | seed | N | M | cells | r (runs) | runs/col | r_d/cell |
|---|---|---:|---:|---:|---:|---:|---:|
| small | 42 | 980 | 9,330 | 9,143,400 | 50,700 | 5.43 | 0.797 |
| small | 7 | 980 | 8,797 | 8,621,060 | 48,372 | 5.50 | 0.813 |

Implied cost of each representation:

| representation | bits/cell |
|---|---:|
| dense `a` + `d` (`int32`) | 64.0 |
| bit-packed panel *(yardstick)* | 1.0 |
| **RLPBWT `a`-side, 1 word/run** | **0.355** |
| **RLE of `d`** | **51.0** |

**The permutation compresses 180×; the divergence array compresses 1.25×.**
`d` is essentially incompressible by run-length encoding at 0.80 runs per cell.
That asymmetry is the central fact of this problem, and §8 returns to it.

For calibration, μ-PBWT reports ≈11 runs/column on 1000 Genomes at N = 5,008;
our 5.43 at N = 980 is the same compressibility regime, so the simulated
instances carry realistic structure.

---

## 5. Result: `sub_rlpbwt_streaming`

Small tier, 980 × 9,330, seed 42:

| submission | stored bytes | bits/cell |
|---|---:|---:|
| `sub_vectorized_pbwt` (dense a+d + panel) | 82,335,760 | 72.04 |
| `sub_streaming_pbwt` (panel only, uint8) | 9,143,400 | 8.00 |
| *(same, bit-packed)* | 1,142,925 | 1.00 |
| **`sub_rlpbwt_streaming`** | **286,778** | **0.251** |

287× below the dense index, 4× below the bit-packed panel. Exact: 40/40
randomized differential tests against the oracle, plus synthetic and coalescent
gate instances, byte-identical to every other submission. Interleaved A/B:
**1.052× faster** than the streaming baseline, 95 % CI [1.038, 1.075].

---

## 6. Scale dependence — the result that reframes everything

Run at the scored tiers on an HPC node:

| tier | N | M | bits/cell | runs/column |
|---|---:|---:|---:|---:|
| small | 980 | 9,329 | 0.2510 | 5.44 |
| n10k | 9,984 | 114,415 | **0.0293** | 6.88 |
| n30k | 29,984 | 129,048 | **0.0112** | 8.20 |

**Runs per column barely grow with N** — 5.44 → 8.20 while N grows 30× —
because `r` is governed by the number of *distinct ancestral haplotypes* at a
locus, not by how many samples were drawn. The coalescent supplies the
redundancy. Hence

```
bits/cell  ≈  32·r / (N·M)   and   r ≈ N-independent   ⟹   bits/cell ~ 1/N
```

A 30× larger panel gives a ~22× smaller per-cell index. **The metric is strongly
scale-dependent, so any bits/cell number is meaningless without its N.**

---

## 7. Agentic search over the representation

Six independent islands proposed encodings, scored *only* by the sealed
evaluator (fresh gate seeds every invocation, SHA-256–locked immutable set).
Each survivor was then audited by an independent skeptic agent.

**6/6 passed the gate. 6/6 confirmed.**

| module | method | bits/cell | vs incumbent |
|---|---|---:|---:|
| **`sub_freeform`** | context-modelled arithmetic coding | **0.02566** | **9.78×** |
| `sub_column_diff` | divergence-conditioned entropy coding | 0.02591 | 9.68× |
| `sub_bitvector_rank` | context-modelled boundary bitvector | 0.02735 | 9.17× |
| `sub_delta_varint` | delta + varint on gaps | 0.03528 | 7.11× |
| `sub_bitpack_fixed` | fixed-width bit packing | 0.03594 | 6.98× |
| `sub_eliasfano` | Elias–Fano boundaries | 0.03604 | 6.96× |
| *(incumbent)* | int32 run starts | 0.25092 | — |

### 7.1 The winning idea

The first three converged independently on the same insight, and it is the
interesting result of the campaign:

> **The divergence array is a free, strong, decoder-side predictor of where the
> next run boundary is.**

A run boundary at position `j` of column `k` is exactly the event "`a_k[j-1]`
and `a_k[j]` differ at site `k`". But `d_k[j]` says those two haplotypes have
matched over `[d_k[j], k)` — and two haplotypes that have matched for 4,000
sites almost never differ, while two that just diverged very often do. The
decoder already holds `d_k`, computed for free from columns it has *already*
decoded, so conditioning on it costs nothing to transmit.

Measured hazard `P(boundary | match-length octave)`:

| match length | 0 | 1–2 | 4–8 | 32–64 | 256–512 | 2048–4096 | 8192+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| P(boundary) | 0.168 | 0.144 | 0.080 | 0.041 | 0.0063 | 0.0010 | 0.0001 |

A **1,700× dynamic range** the incumbent discards by paying a flat 32 bits per
run. `sub_freeform` codes each column's boundary positions against a
non-homogeneous geometric with hazard `q(j) = P(boundary at j | ctx)`, context =
(octave of `k − d_k[j]`, current run's allele, octave of causal run length),
with an adaptive Krichevsky–Trofimov table so no constant is fitted to an
instance and no model is shipped.

Coding a whole run in **one** coder operation against the cumulative survival
`∏(1−q(i))·q(j)` is exactly equivalent to one binary decision per cell, but
costs 50,700 coder calls instead of 9,143,400.

Bit budget, small tier:

| component | bits | events | rate |
|---|---:|---:|---:|
| boundaries | 181,448 | 50,700 | 3.58 bits/boundary |
| non-boundary cells | 59,930 | 9.08 M | 0.0066 bits/cell |
| first alleles | 5,517 | 9,330 | 0.59 bits/column |
| **total** | **234,664** | | **0.02566 bits/cell** |

The two-pass zero-model-cost entropy of the same contexts is 230,739 bits, so
the adaptive coder runs within **1.7 %** of its own model's floor. The author
also reports the context family is *saturated*: quarter-octave contexts buy only
2 % even with zero model cost, and five other conditioning features were built,
measured, and rejected. The next real gain requires a different predictor, not a
finer one.

There is a pleasing irony here. The literature audit concluded "all the space is
in `d`." The winning submission does not store `d` at all — it *uses* `d`, which
it gets for free, to compress `a`.

### 7.2 Plausibility floor

`xz -9e` on the bit-packed panel reaches **0.03234 bits/cell**, and it is not an
index at all — merely storage. The top three candidates are *below* it. A
queryable exact index smaller than general-purpose compression of the same data
is a meaningful result and rules out "you are just compressing well."

(The check also caught a real overstatement: `sub_delta_varint`'s docstring
claims no general-purpose compressor beats it. At 0.03528 it is above xz. The
code is fine; the sentence is wrong.)

### 7.3 Verification

Auditors did not take the numbers on trust. They re-scored on independent fresh
seeds; ran `gc.get_referents` walks to prove no numpy array is reachable from
the index; checked `__slots__` so that a `__dict__` is not even representable;
pickled the index to bound its true size; **zeroed the panel in place and
re-queried**, proving results are unchanged and the panel is genuinely unused;
reconstructed the full panel bit-for-bit to prove the codec is lossless; and
re-derived correctness against a brute-force enumerator written from the
specification rather than trusting `oracle.py`. One ran 600 randomized
differential tests plus 8 adversarial edge cases.

### 7.4 Does the gain transfer? Yes — but the small-tier ranking is wrong

Measured across all three tiers (HPC node):

| module | small | n10k | **n30k** | gain@small | gain@n30k | rank@small | rank@n30k |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sub_column_diff` | 0.02592 | 0.00375 | **0.00158** | 9.68× | **7.07×** | 2 | **1** |
| `sub_bitvector_rank` | 0.02737 | 0.00414 | 0.00174 | 9.17× | 6.41× | 3 | 2 |
| `sub_freeform` | 0.02567 | 0.00395 | 0.00190 | **9.78×** | 5.87× | **1** | 3 |
| `sub_delta_varint` | 0.03529 | 0.00563 | 0.00234 | 7.11× | 4.76× | 4 | 4 |
| `sub_bitpack_fixed` | 0.03595 | 0.00569 | 0.00234 | 6.98× | 4.76× | 5 | 5 |
| `sub_eliasfano` | 0.03605 | 0.00569 | 0.00238 | 6.96× | 4.68× | 6 | 6 |
| *(incumbent)* | 0.25097 | 0.02928 | 0.01115 | — | — | — | — |

**(a) The gain transfers, attenuated as predicted.** Every candidate still beats
the incumbent at scale; the advantage falls from ~9.8× to ~5.9–7.1×. The
mechanism is understood — the incumbent pays a flat 32 bits/run while a modelled
coder pays ≈ log₂(N/r) + c, so the ratio closes logarithmically in N, matching
`sub_freeform`'s own row-subset curve (19.5× at N=61 → 9.8× at N=980).

**Best absolute result: 0.00158 bits/cell at N = 29,984 — one bit per 633 panel
cells, 633× below the bit-packed panel, exact against the oracle.**

**(b) The small-tier podium reorders.** `sub_freeform` wins at the small tier and
is **third** at n30k; `sub_column_diff` is second at small and **first** at n30k.
A benchmark run only where it is cheap to run would have crowned the wrong method
and drawn the wrong conclusion about which idea matters.

**A point improvement at one size is not a result until its curve is measured** —
and here the curve changes not just the magnitude but the ranking.

---

## 8. What remains open

Everything above compresses the **permutation**. The divergence array is still
*recomputed* during the sweep in O(N) per site, never *stored* with O(1) access.
Consequences:

1. A single online query still costs **O(N·M)** — the sweep must run.
2. The Finding-6 target is untouched:

> **Represent the divergence information in o(N·M) space with O(1) access, for
> all-long-match reporting.**

The theory says this is reachable and **has never been implemented**:

- **Sanaullah, Zhi & Zhang, WABI 2025** ([arXiv:2505.15698](https://arxiv.org/abs/2505.15698),
  DOI `10.4230/LIPIcs.WABI.2025.17`) — O(m + occ) expected time in **O(r) words**
  for exactly this problem, via a Nishimoto–Tabei move structure with O(1)
  `PLCP[i]`. The paper's own future work asks for "a practical implementation…
  possibly as a modification of MOVI or b-move."
- **Bonizzoni, Cozzi & Gao** ([arXiv:2602.13461](https://arxiv.org/abs/2602.13461), 2026) —
  O(r̃)-word RLPBWT with **constant-time** forward and backward stepping.
  Explicitly "contains no experimental validation."

And no published implementation is simultaneously (a) exact for all long maximal
matches, (b) ≤ 1 bit/cell, and (c) not slower than uncompressed PBWT. That
intersection was empty as of 2026-09-01. `sub_rlpbwt_streaming` and its six
successors occupy it for the batch workload; the online-query case does not.

**Bounds, honestly.** No lower bound specific to long-match query on haplotype
panels is published. Kempa & Kociumaka ([arXiv:2510.19820](https://arxiv.org/abs/2510.19820), 2025)
prove that in O(δ·polylog n) space over binary alphabets, LCE/SA/LCP-class
queries need Θ(log n / log log n) while BWT/LF/PLCP/Φ need Θ(log log n).
Divergence access is the positional analogue of LCP — the harder class — so O(1)
divergence access is out of reach at δ-bounded space but attainable at O(r)
words. The search should live in that gap. *Caveat stated rather than hidden:*
those theorems concern standard strings and the ordinary BWT; the PBWT is a
different, column-wise transform and no paper formally transfers them. Treat it
as a strong analogy, not a citation.

---

## 9. Reproduction

```bash
cd problems/P01_haplotype_long_match
python3 run_stats.py --tiers small --seeds 42,7      # §4
python3 evaluate.py --tier small --submission sub_rlpbwt_streaming
python3 score_submission.py --module candidates.sub_freeform --json
```

At the scored tiers (needs a large-memory node; see `cluster/README.md`):

```bash
export ARENA_ROOT=/path/to/bioinformatics-algorithm-arena
sbatch $ARENA_ROOT/cluster/slurm/p01_candidates_scored.slurm
```

Instances are generated fresh from a seed (msprime, `BinaryMutationModel`), so
no answer key exists to memorise and no constant can be tuned to an instance.
Every result row carries an `instance` fingerprint — content SHA plus generator
versions — because **instance generation is not version-stable**: the same seed
gives 9,552 matches under msprime 1.3.1 and 9,603 under 1.4.3. Never compare
rows whose `sha` differs.

---

## 10. Errors recorded

The arena's rule is that its own mistakes are part of the record.

| # | Error | Correction |
|---|---|---|
| 1 | Gate passed while the **scored tier was silently wrong** — msprime's default model made multi-allelic sites; PBWT splits on `allele==0` vs non-zero, merging alleles 1 and 2. 5 spurious + 1 missing per 3,245. | Pinned `BinaryMutationModel`; oracle asserts the precondition; gate now draws from the scored distribution. Gate panels were synthetic and binary, so they *could not* express the bug. |
| 2 | Claimed the index buys **N-independent queries**. | Retracted. Output grows as matches/query ~ N^1.31, so per-query time must grow. N-independence is a property of SMEM queries. The comparison against μ-PBWT was apples-to-oranges. |
| 3 | Predicted the dense index would hit **~8 TB and fail** at large N. | Wrong: ~29 GB at n30k, ran fine. M grows with N (9,329 → 114,415), which the estimate ignored. |
| 4 | `sub_streaming_pbwt.nbytes()` returned **0** while retaining the panel — the exact accounting cheat six auditors were told to hunt, in our own baseline. | Fixed to report `panel.nbytes`. Audit the incumbent with the tooling you point at submissions. |
| 5 | `check_lock()` was **trust-on-first-use**: a missing `immutable.lock` was silently created and reported "ok". | Missing lock is now a hard refusal; only explicit `--init-lock` may write it. The arena is now committed to git so the lock has a baseline. |

---

## 11. Addendum: the online scorer

Everything above scores **space** on a batch workload, where a submission may
sweep the panel once and amortise it over many queries. That workload cannot see
the open problem, because a sweep is O(N·M) however well the columns compress.
`score_online.py` therefore measures **single-query cost** with the index built
once, fitting `log(seconds) = a + b·log(N)` with M held fixed (M = 3,000) and a
long match threshold (L = 2,000) so that output size does not floor the exponent.

Calibration run, N = 2,000–16,000:

| module | exponent b | s per query (largest N) | bits/cell |
|---|---:|---:|---:|
| `sub_rlpbwt_streaming` (incumbent) | +0.637 | 1.472 | 0.0567 |
| `candidates.sub_column_diff` (space champion) | +0.970 | 30.137 | 0.0065 |

The best space result is about 20× **worse** online: arithmetic coding decodes
sequentially, so reaching column k means decoding columns 0…k. Extreme
compression and random access are in direct tension, and the two scorers now
expose that Pareto structure instead of hiding it. The target is b → 0:
single-query time independent of N.
