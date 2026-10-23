# P01 — Haplotype long-match query / haplotype panel indexing
### Benchmark specification v2 · 2026-09-01 · supersedes `problem.yaml` v1

> **Read this first.** While verifying the incumbent I found that the current P01 assets are **not sound**: the generator violates its own biallelic precondition, and as a consequence the PBWT baseline and both submissions emit spurious matches on the *scored* tier while passing the gate. All v1 numbers in `results.jsonl` are measurements of a program that is not exact. Details and the verified fix are in **Correctness gate** and **Confidence notes**. Everything below assumes those fixes land first.

---

## Formal statement

Let `X ∈ {0,1}^{N×M}` be a panel of `N` haplotypes over `M` biallelic sites in a fixed site order, `Z ∈ {0,1}^{Q×M}` a set of query haplotypes disjoint from `X`, and `L ∈ ℕ` a length threshold in sites.

Report the set

```
LM(X, Z, L) = { (i, j, s, e) :
      Z[i][s:e] == X[j][s:e]                      (match)
      e - s >= L                                  (long)
      s == 0    or  Z[i][s-1] != X[j][s-1]        (left-maximal)
      e == M    or  Z[i][e]   != X[j][e]     }    (right-maximal)
```

This is the **long-match query** problem of Durbin (2014) Algorithm 5 / Naseri et al. (2019) L-PBWT-Query / Wang et al. (2023) Syllable-Query. In the compressed-indexing literature the same objects are the **long LEMs** of Sanaullah, Zhi & Zhang (WABI 2025).

**Precise disambiguation — this is the axis on which half the "PBWT SOTA" literature is not comparable:**

| Object | Definition | Tools that compute it |
|---|---|---|
| **All long maximal matches (this problem)** | every left- and right-maximal match of length ≥ L, all `(i,j)` pairs | Durbin Alg 5, L-PBWT-Query, Syllable-Query, Dynamic μ-PBWT long-match query, WABI-2025 long-LEM algorithm (theory only) |
| **Set-maximal / SMEM** | for each query position, only the *longest* match; `j` reported only if maximal | Durbin Alg 5 (set-maximal variant), μ-PBWT, PBML |
| **k-SMEM / KL-SMEM** | SMEM occurring in ≥ k haplotypes and of length ≥ L | PBML |

μ-PBWT and PBML **do not solve this problem**; they solve the SMEM variant, whose output is asymptotically smaller. Any headline comparison against their numbers is invalid unless the output object is matched. Syllable-PBWT and Dynamic μ-PBWT are the only *implemented* compressed methods that solve the stated problem.

Lower bounds inherent to the statement: `Ω(M)` per query (the query must be read), `Ω(occ)` output, `Ω(NM)` build time from an uncompressed panel (every cell must be read at least once). Any claim below these is a claim that the input format changed.

---

## Input / output / submission API

A submission is a Python module exposing:

```python
build(panel: np.ndarray[uint8, (N, M)]) -> Index
    # Index MUST expose:
    #   .nbytes() -> int   total bytes of ALL state retained after build,
    #                      INCLUDING any retained reference/copy/view of `panel`.
    #                      Returning 0 while holding the panel is a scoring violation.
    #   .words()  -> int   same quantity in 64-bit words (theory-facing)

query(index: Index, z: np.ndarray[uint8, (M,)], L: int) -> list[(j, s, e)]
    # every maximal match of z against the panel, e - s >= L, in any order,
    # no duplicates

# OPTIONAL, declared capability:
query_batch(index, Z: np.ndarray[uint8, (Q, M)], L: int) -> list[list[(j, s, e)]]
    # a submission that only implements query_batch is scored on the BATCH track;
    # it is not comparable on the single-query track (see Beat target).

# REQUIRED in counted mode (see Implementation-independent counters):
build_counted(panel, ctr) -> Index
query_counted(index, z, L, ctr) -> list[(j, s, e)]
    # identical algorithm, all index/panel reads routed through `ctr`-instrumented
    # accessors supplied by the harness
```

`L` is in **sites**, not cM. Sites are the panel's own column indices; no genetic map is supplied at the simulated tiers (a map is supplied only in the real-data external-validity run, and only to convert an L in cM to a per-instance L in sites).

Determinism: `build` and `query` must be deterministic given `(panel, z, L)`. Randomised algorithms are permitted only if seeded from a harness-supplied seed and re-run reproducibly; a randomised algorithm with a false-negative rate is **not admissible on the exact track** (see Mutable surface, approximation clause).

Submissions live in `problems/P01_haplotype_long_match/` and are selected by module name.

---

## Ground truth (EXACT vs CURATED vs PROXY, and justify)

**EXACT.** The truth is a *function of the input*, computed by direct enumeration of the definition, not a downloaded answer key. There is nothing to memorise: fresh seeds produce fresh instances, and the oracle recomputes the truth for each one.

Two structural properties make this the strongest evaluator class in the arena:

1. **Pairwise decomposability.** The predicate for `(i, j, s, e)` depends only on rows `Z[i]` and `X[j]`. Therefore restricting the panel to any row subset `J ⊆ [N]` yields the *exact* truth for all pairs in `Z × J` — not an estimate. This is what makes the large tiers verifiable at all: a 256-row subsample is an exact partial oracle, and any systematic omission by a candidate shows up as a set-difference, not a statistic.
2. **Cheap certification of soundness.** Every reported tuple can be verified independently in `O(e - s)` — segment equality plus two boundary tests. Soundness is therefore checkable at 100% coverage at every tier including `large`; only completeness needs sampling.

Not CURATED: there is no consortium truth set, no GIAB analogue, no circularity risk. Not PROXY: no statistical surrogate is involved anywhere.

The one thing simulation *cannot* certify is that the instance distribution resembles real panels. That is handled separately by an unscored external-validity run on 1000 Genomes (below), which checks the *ranking* of candidates is preserved on real data — never the correctness, which is instance-independent.

---

## Instance generation and tiers

**Generator:** `msprime` — verified locally at **1.3.1**; current release as of 2026-09-01 is **1.3.3** (bugfix, Beta/Dirac coalescent only; pin either, record the version). Source: <https://github.com/tskit-dev/msprime/releases>.

### The mandatory generator fix

The v1 generator calls `msprime.sim_mutations(ts, rate=1.5e-8, random_seed=...)` with the default (JC69, discrete-genome) model. **This produces sites with three alleles**, violating the "biallelic" precondition. Measured on the current assets:

| tier / seed | shape | max allele | sites with allele > 1 |
|---|---|---|---|
| tiny / 42 | 60 × 546 | 1 | 0 (0.00 %) |
| small / 42 | 1000 × 9330 | **2** | 17 (0.18 %) |
| small / 7 | 1000 × 8797 | **2** | 9 (0.10 %) |

The tiny tier is accidentally clean, which is exactly why the gate never caught it. Verified fix:

```python
ts = msprime.sim_mutations(
        ts, rate=1.5e-8, random_seed=seed + 1,
        model=msprime.BinaryMutationModel(state_independent=False))
assert ts.genotype_matrix().max() <= 1
```

After this change the PBWT baseline is exact on **0/8 failures at each of n=120, n=200, n=400** over 8 fresh msprime seeds each — versus **4/8, 2/8, 1/8 failures** before. (An alternative fix — dropping non-biallelic sites post hoc — is *not* acceptable: it makes `M` seed-dependent in a way that is invisible in the instance metadata.)

### Tiers

Parameters below are `sim_ancestry(samples=n//2, ploidy=2, sequence_length=Lbp, recombination_rate=1e-8, population_size=10_000, random_seed=seed)` then `sim_mutations(rate=1.5e-8, model=BinaryMutationModel(state_independent=False), random_seed=seed+1)`; `q` queries held out by a seeded permutation.

| tier | n (hap) | seq len (bp) | measured M (sites) | q | L sweep (sites) | wall to generate | fits laptop? |
|---|---|---|---|---|---|---|---|
| **gate** | 120–200 | 4e5–5e5 | ~1 280 – 1 720 | 5–8 | 60, 80 | 0.1–0.3 s | yes |
| **small** | 1 000 | 2e6 | 9 330 (seed 42) | 20 | 100 / 300 / 1200 | **0.82 s** | yes (M1, 16 GB) |
| **medium** | 10 000 | 1e7 | **57 465** (seed 42) | 50 | 300 / 1000 / 4000 | **8.0 s** | yes, ~1 GB peak |
| **large** | 100 000 | 2e7 | ~115 000 (est.) | 100 | 1000 / 4000 | est. 3–8 min | HPC |

All measured on the target laptop (Apple M1, macOS 26.6). Exact command:

```bash
python3 generate.py --tier small --seed $RANDOM --out inst.npz     # after the fix
```

Note: v1's `problem.yaml` records `medium` as `n_hap: 10000, seq_len_bp: 1.0e7`; the realised panel is `9950 × 57465`, i.e. **six times more sites than the yaml's descriptive text implies**. Record realised shapes in every result record; do not quote nominal tier parameters as if they were shapes.

### Real data (external-validity run only, never in the search loop)

Verified live 2026-09-01, exact filenames and sizes from the directory listing:

```
https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/
  working/20220422_3202_phased_SNV_INDEL_SV/
    1kGP_high_coverage_Illumina.chr20.filtered.SNV_INDEL_SV_phased_panel.vcf.gz   655M
    1kGP_high_coverage_Illumina.chr21.filtered.SNV_INDEL_SV_phased_panel.vcf.gz   407M
    1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz   425M
```

3 202 samples = 6 404 haplotypes, phased, GRCh38. Extract biallelic SNVs with `bcftools view -m2 -M2 -v snps` → uint8 matrix. This is the panel Syllable-PBWT and μ-PBWT both benchmark on (they use the older 2 504-sample / 5 008-haplotype release), so it is the natural bridge to published numbers. **UK Biobank must never be a scored tier** — it is not redistributable, so no third party can reproduce a UKB-scored result.

---

## Correctness gate

Binary. A candidate that fails is not scored; there is no partial credit.

**G0 — Precondition assertion (on the instance, not the submission).** `panel.max() <= 1` and `query.max() <= 1`. Aborts the run, not the candidate. This is the check whose absence produced the v1 defect.

**G1 — Exact set equality on fresh gate instances.** `n_gate = 8` instances drawn from the **same msprime generator as the scored tier**, at `n ∈ {120, 200}`, `seqlen ∈ {4e5, 5e5}`, `q ∈ {5, 8}`, `L ∈ {60, 80}`, with seeds drawn from the run's own entropy. Submission output must equal `oracle(panel, query, L)` as a set of `(i, j, s, e)` tuples, including no duplicates.

> **Why this replaces the v1 gate.** The v1 gate used a hand-rolled founder-mosaic generator at `N ∈ [6,30]`, `M ∈ [10,50]`, `L ∈ [2,8]`. I ran that generator up to `N=400, M=4000, L=100`, 30 seeds per configuration: **0/30 failures at every size.** The same submission fails **4/8** msprime seeds at `n=120`. A gate whose instance distribution differs from the scored distribution is not a gate. Cost of the replacement: **2.7 s** at n=120, **7.1 s** at n=200.

**G2 — Soundness at 100 % coverage, every tier.** Every reported `(i, j, s, e)` is independently verified: `(X[j][s:e] == Z[i][s:e]).all()`, `e - s >= L`, `s == 0 or X[j][s-1] != Z[i][s-1]`, `e == M or X[j][e] != Z[i][e]`. Vectorised, cost `O(Σ (e-s))`. This alone would have caught the v1 defect: 48 of 9 600 reported tuples at small/seed 42/L=300 fail `seg_all_equal`.

**G3 — Completeness at the scored tier.**
- `small`: full oracle, exact, **2.65 s** with the vectorised oracle below.
- `medium` / `large`: **row-subsampled exact oracle**. Draw `R = 256` panel rows without replacement; require the candidate's output restricted to those rows to equal the oracle's output on the subsampled panel *with the original row indices restored*. Exact for the sampled pairs (pairwise decomposability). Measured cost at medium (`9950 × 57465`, 50 queries, 256 rows): **9.3 s**. A full medium oracle is ≈ 6 min — run it once per milestone, not per candidate.

**G4 — Anti-collusion.** The gate seeds are drawn per evaluation run from OS entropy and are not visible to the candidate. The candidate never sees the oracle module.

---

## Accuracy metric

There isn't one, by design. **Accuracy is a gate, not a score.** `LM(X,Z,L)` is a set with an exact definition; a candidate either reproduces it or it does not. Reporting an F1 would invite a leaderboard on which a 0.999-recall method outranks an exact one, which is scientifically wrong for this problem — the downstream consumers (IBD segment calls, imputation reference selection, genealogical search) need soundness.

There is **no external evaluation tool**. The evaluator is `oracle.py`, and it must be replaced by the vectorised form, which I implemented and verified produces byte-identical output:

```python
def long_matches(panel, query, L):
    N, M = panel.shape
    out = set()
    for i in range(query.shape[0]):
        eq  = (panel == query[i])
        pad = np.zeros((N, 1), dtype=bool)
        d   = np.diff(np.concatenate([pad, eq, pad], 1).view(np.int8), axis=1)
        rs, cs = np.nonzero(d ==  1)      # run starts
        re, ce = np.nonzero(d == -1)      # run ends (exclusive)
        keep = (ce - cs) >= L
        out.update((i, int(j), int(s), int(e))
                   for j, s, e in zip(rs[keep], cs[keep], ce[keep]))
    return out
```

Verified against the v1 reference oracle at small/42/L=300: identical sets (9 552 tuples), **31.20 s → 2.65 s, 11.8×**. Timing is flat in `L` (2.5–2.8 s across L ∈ {100, 300, 1200}), so the L-sweep is nearly free.

A separate **approximate track** exists for declared-inexact methods, scored on recall at fixed precision ≥ 0.999 with the exact oracle as truth. It is a different leaderboard and a candidate may never migrate a result between them.

---

## Resource metrics

Four axes, all first-class, all reported per `(tier, seed, L)`.

**Build**
- `build_s` — min over 3 repeats (min, not mean: scheduling noise is one-sided).
- `build_peak_rss_bytes` — measured in a **child process** running build only. `measure.py`'s own docstring specifies child-process measurement; `evaluate.py` v1 uses `resource.RUSAGE_SELF` in the evaluator process, which (a) includes the numpy/instance/gate footprint and (b) is monotone, so it cannot separate build peak from query peak. Fix: two child invocations, `--phase build` and `--phase query`, each reporting its own `ru_maxrss`.

**Query**
- `query_s_per_query` — min over 3 repeats of the total, divided by `Q`. **First query timed separately** from steady state, to expose work deferred out of `build` into a lazy first call.
- `query_peak_rss_bytes` — child process, index loaded from disk, panel *not* pre-resident.

**Index size — the axis with the loophole**
- `index_bytes = Index.nbytes()`, defined as **every byte retained after build, including any retained view of the panel.** v1's `PBWTIndex.nbytes()` docstring says "excluding the panel itself" while the object holds `self.panel`; and `sub_streaming_pbwt` reports `index_bytes = 0` while holding the panel. Under that accounting the index axis is trivially winnable by doing nothing. Conservation rule: `index_bytes ≥ ceil(N*M/8)` unless the submission genuinely compresses the panel, in which case it must be able to reconstruct any `panel[j][k]` on demand and is spot-checked on 10 000 random cells.
- `index_bits_per_panel_cell = index_bytes * 8 / (N*M)` — the literature-comparable unit.
- `index_words_per_run = Index.words() / r`, where `r` is the harness-computed RLE-PBWT run count of the instance — the theory-comparable unit.
- On-disk serialised size is reported separately from in-memory size; they diverge for compressed structures and both matter.

**Amortisation (closes the streaming loophole without banning streaming)**

A build-free streaming sweep is a *legitimate* algorithm, but it wins the index axis by construction. Score the amortised cost at a declared query load:

```
amortized_s(Qref) = build_s / Qref + query_s_per_query
```

Report the curve at `Qref ∈ {1, 10², 10⁴, 10⁶}`. The **headline track uses Qref = 10⁴**, because the motivating application (genealogical search against a fixed biobank panel) queries a built index millions of times. Streaming wins at `Qref = 1` and should be shown winning there; it must not be allowed to win the headline.

**Output size** is a cost axis too: `output_bytes` in the harness's canonical encoding. A method that reports matches as run-intervals rather than per-pair tuples is doing something real and should be credited — but only if it can enumerate the tuples on demand for G2/G3.

---

## Implementation-independent counters

This is the arena's defence against ranking a numpy rewrite above an algorithm. The v1 harness proved the danger on its own assets: `baseline_pbwt` and `sub_vectorized_pbwt` produce **byte-identical output with identical asymptotics and identical index bytes**, and differ 79× on build wall-clock.

Counters must be emitted by **harness-supplied instrumented accessors**, never self-reported. The submission implements `build_counted` / `query_counted` against a `TrackedArray` wrapper the harness injects; counted mode is slow and is run once per candidate at the small tier only.

| counter | definition | what it measures |
|---|---|---|
| `build_cell_reads` | reads of `panel[j][k]` during build | build work; `Ω(NM)` is the floor |
| `index_words` | 64-bit words retained after build (incl. panel) | space, language-free |
| `query_index_probes` | random reads of index words during query: `a[k][i]`, `d[k][i]`, run headers, rank/select calls, hash lookups, move-table entries | the query's real cost |
| `query_cell_reads` | reads of `panel[j][k]` during query | block-scan cost |
| `report_ops` | tuples emitted | equals `occ`, an instance constant |

**Derived quantities — these are the actual scoreboard for algorithmic novelty:**

```
build_work_ratio = build_cell_reads / (N * M)                       # incumbent ≈ 1
query_work_ratio = (query_index_probes + query_cell_reads) / (M + occ)
space_ratio      = index_words * 64 / (N * M)                       # bits per panel cell
run_ratio        = index_words / r                                  # words per RLE run
```

`query_work_ratio` is the key one. PBWT's claimed bound is `O(M + occ)` **independent of N**; the falsifiable prediction is that `query_work_ratio` is flat in `N`. Fit `log(query_work_ratio) ~ β log N` across the four tiers at fixed `occ/M`. A candidate that reduces `β`, or reduces the constant at equal `β`, has made an algorithmic advance regardless of what language it is written in. A candidate that only reduces seconds at unchanged counters has made an engineering advance and is ranked on the separate engineering leaderboard.

`r` is computed by the harness per instance and stored in the instance metadata (RLE run count of the PBWT allele columns, `Σ_k runs(y_k)`).

---

## Incumbent(s) to beat

| tool | version / date | what it computes | index size (bits per panel cell) | query time | Apple Silicon |
|---|---|---|---|---|---|
| **Arena `baseline_pbwt`** | this repo | all long maximal matches | **64.0** measured (a+d int32) | 0.247 s/q at N=980, M=9330, L=300 | runs — but see caveat |
| **Durbin `pbwt`** | `richarddurbin/pbwt`, pushed 2025-11-20, no license file | `-longWithin <L>` (within-panel), `-matchDynamic` (query vs panel) | run-length compressed panel; index not separately reported | linear in query, independent of panel size (claimed) | **very likely yes**, unverified: pure C, `CFLAGS=-g -O3`, no x86 intrinsics in the Makefile, sole dependency htslib (arm64-clean) |
| **L-PBWT-Query** (Naseri et al. 2019) | Bioinformatics 35:i233 | all long matches, query vs panel | **64** (paper: "roughly 8NM memory"; 4.7 TB max RSS on UKB) | **6 s/query** over all 22 UKB autosomes (974 818 hap × 658 720 sites, L=700) vs ~20 s for PBWT-Query | source at genome.ucf.edu/pbwt-query; not tested |
| **Syllable-PBWT / Syllable-Query** (Wang, Naseri, Zhang, Zhi 2023) | `ZhiGroup/Syllable-PBWT`, MIT, pushed 2023-08-23 | **all long maximal matches — the direct incumbent** | **2.02** (B=64) / **1.14** (B=128) on UKB; **1.99** / **1.03** on 1000G sequence | "faster than existing solutions" — reported as figures, not a runtime table | README says Linux + GCC≥5; no SIMD or x86 intrinsics documented → likely builds, unverified |
| **Dynamic μ-PBWT** (Shakya, Sanaullah, Zhi, Zhang; RECOMB 2025 / J. Comput. Biol. 2026) | `ucfcbb/Dynamic-mu-PBWT`, MIT, pushed 2025-05-05 | all long maximal matches + dynamic insert/delete | **~12** (UKB, 1.5 B/site/hap); **~14.7** (1KGP, 1.84 B/site/hap) | UKB chr21, M=9793, L=2000: **0.2325 s/q at N=100 000** (μ-PBWT: 1.0774 s) | C++/SDSL; unverified |
| **μ-PBWT** (Cozzi et al. 2023) | `dlcgold/muPBWT`, GPL-3.0, pushed 2026-09-01 | **SMEM only — not this problem** | **0.021** on UKB chr20 WGS (11.06 GB for 300 238 × 13 780 193) | up to 6× slower than Durbin at 1000 queries; up to 80× less memory | bioconda recipe exists |
| **PBML** (Islam, Cozzi, Gagie, Varki, Colonna, Garrison, Bonizzoni, Boucher 2025/2026) | `uwaiseibna/PBML`, pushed 2026-04-07 | **KL-SMEM only — not this problem** | 14× less than Durbin PBWT; 22 % more than μ-PBWT | 63.8 % faster than μ-PBWT single-threaded; 6.4 % slower than Durbin; 15.9× over μ-PBWT at 16 threads | unverified |
| **RaPID2** (Tang, Naseri, Zhi, Zhang, *Bioinf. Advances* 2026-03-17) | `ucfcbb/RaPID2`, MIT, pushed 2026-04-08 | approximate all-vs-all IBD (P02's problem, not P01's) | — | UKB chr20, 974 818 hap, 17 197 sites, 2 cM: 65 513 s / 822 GB (32.5× faster than RaPID) | — |

**Bits-per-cell arithmetic, shown so it can be checked.** UKB panel used by Syllable-PBWT: 974 818 × 658 720 = 6.4213e11 cells. 10.1 TB → 8.08e13 bits / 6.4213e11 = **125.8 bits/cell** (= 4 × int32: a, d, u, v — confirming the paper's Table 1 "4 units of MN 32-bit integers"). 162 GB → **2.018**; 91.4 GB → **1.139**. 1000G sequence 4 908 × 6 468 094 = 3.1745e10 cells: 7.9 GB → **1.991**; 4.1 GB → **1.033**. μ-PBWT UKB chr20: 300 238 × 13 780 193 = 4.1374e12 cells; 11.06 GB → **0.0214**.

**Sources.** Syllable-PBWT: <https://academic.oup.com/bioinformatics/article/39/1/btac734/6849513> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC9805553/>. μ-PBWT: <https://academic.oup.com/bioinformatics/article/39/9/btad552/7265394> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC10502237/>. Dynamic μ-PBWT: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11838588/>. L-PBWT-Query: <https://academic.oup.com/bioinformatics/article/35/14/i233/5529240>. d-PBWT: <https://academic.oup.com/bioinformatics/article/37/16/2390/6149123>. PBML: <https://www.biorxiv.org/content/10.64898/2025.12.01.691644v1.full>. RaPID2: <https://academic.oup.com/bioinformaticsadvances/article/6/1/vbag078/8526998>. GBZ: <https://academic.oup.com/bioinformatics/article/38/22/5012/6731924>.

**Is the 64× figure representative?** Partly. `8.0 bytes/cell = 64 bits/cell` is **exactly** the published footprint of the L-PBWT-Query family ("roughly 8NM memory", 4.7 TB on UKB), so it is not an arena artifact — it is the real cost of the uncompressed a+d formulation, and d-PBWT Algorithm 3 is worse still at 126 bits/cell. **But it is 60–110× above published SOTA for the same problem**: Syllable-Query is at 1.0–2.0 bits/cell. So "cut 8 bytes/cell to below 1 byte" (the v1 beat target) is not a win — it is roughly 8× *worse* than a 2023 paper. The beat target has to be recalibrated against 1 bit/cell, not against 8 bytes.

**Apple Silicon caveat on the arena's own numbers.** `uname -m` reports `arm64` (Apple M1) but `platform.machine()` reports `x86_64` and Python is 3.8.8 (EOL) — the interpreter is running under **Rosetta 2**. Every timing in `results.jsonl` is an emulated-x86 number on an ARM host. Rebuild the environment on native arm64 Python ≥3.11 before any timing is quoted.

**GBWT/GBZ**: correctly out of scope for P01. It indexes paths in a *graph*, so the coordinate system is nodes not sites, and neither its space nor its query is expressible in bits per panel cell. Table 1 of the GBZ paper gives 1000GP (5 008 haplotypes, 3.2 Gbp): GBWT 7.97 GiB, GBZ 16.84 GiB — no per-cell normalisation is derivable. Keep it as prior art for run-length-compressed FM-indexing, not as a scored incumbent.

---

## Where the real headroom is

**The incumbent family is demonstrably not near-optimal, and I can localise the gap to a single data structure.**

### 1. The measured evidence: it is the divergence array, and nothing else

I computed the run structure of the arena's own simulated instances (re-derivable with `problems/P01_haplotype_long_match/run_stats.py`):

| tier / seed | N × M | cells | `r` (PBWT allele-column runs) | runs/column | `r_d` (divergence-array runs) | `r_d`/cell |
|---|---|---|---|---|---|---|
| small / 42 | 980 × 9 330 | 9 143 400 | **50 712** | 5.4 | 7 288 990 | **0.797** |
| small / 7 | 980 × 8 797 | 8 621 060 | **48 372** | 5.5 | 7 006 373 | **0.813** |

For calibration, μ-PBWT reports "average number 11 of runs per column" on 1000 Genomes (5 008 haplotypes). Our 5.4 at N=980 is the same regime — the simulated instances have the right compressibility structure.

The consequence is stark. An `O(r)`-word index at a generous 2 words/run costs `50 712 × 128 bits / 9 143 400 cells = 0.71 bits/cell` — **90× smaller than the arena baseline's 73 MB, and below the 1.0 bits/cell bit-packed panel itself.** Meanwhile the **divergence array is essentially incompressible by run-length encoding: 0.80 runs per cell.** All of the space is in `d`, none of it is in `a`.

That is not a coincidence — it is the exact statement the theory literature has been circling. Syllable-PBWT wins by *not storing d at all* (rolling hashes over syllables reconstruct positional comparisons). μ-PBWT wins by storing Φ/threshold samples instead of d — and Φ alone is ~40 % of its memory. The WABI 2025 long-LEM paper's entire technical contribution is a constant-time `PLCP[i]` in `O(r)` space. **The open engineering problem for P01 is: represent the divergence information in `o(NM)` space with `O(1)` access, for all-long-match reporting.** An evolutionary search has a precisely named target.

### 2. The theory has outrun every implementation, by two papers

- **Sanaullah, Zhi & Zhang, WABI 2025** (LIPIcs 344, 17:1–17:25, DOI `10.4230/LIPIcs.WABI.2025.17`; arXiv:2505.15698, 2025-05-21): an `O(m + occ)` expected-time algorithm for **exactly this problem** ("long LEMs") in **`O(r)` words**, via an adaptation of the Nishimoto–Tabei move structure, supporting `PLCP[i]` in `O(1)`. The paper's own Discussion states the future work: *"a practical implementation of the structures and algorithms described here, possibly as a modification of MOVI or b-move."* **There is no implementation.** I checked the arXiv HTML, the Dagstuhl landing page, and searched for a repository — none exists.
- **Bonizzoni, Cozzi & Gao, arXiv:2602.13461 (2026-02-13), "Optimal-Time Mapping in Run-Length Compressed PBWT":** an `O(r̃)`-word structure with **constant-time forward and backward stepping** in the RLPBWT, improving the prior `O(log log_w(ℓ/r_j))` per step. Explicitly a theory paper: *"contains no experimental validation."*

So: the optimal space/time point for P01 — `O(r)` words, `O(m + occ)` query, `O(1)` stepping — is *proved achievable* and *has never been built*. That is the single largest piece of headroom in the arena, and it is not speculative.

### 3. The published SOTA is itself inconsistent, which means nobody has consolidated the wins

Syllable-PBWT gets to **1.0–2.0 bits/cell** with fast queries but is static and hash-based (no `r`-sensitivity). μ-PBWT gets to **0.021 bits/cell** on WGS but only answers SMEMs and is **up to 6× slower than Durbin**. Dynamic μ-PBWT answers the right query but sits at **12–15 bits/cell**, 8–12× worse than Syllable, and is 60–160× slower than d-PBWT on updates. PBML is fast and parallel but answers KL-SMEMs. **No single implementation is simultaneously (a) exact for all long maximal matches, (b) at or below 1 bit/cell, and (c) not slower than uncompressed PBWT.** That intersection is empty in the literature as of 2026-09-01. Filling it is a publishable result and is precisely what a multi-objective search is for.

### 4. Lower bounds — what is actually known, honestly

There is **no published lower bound specific to long-match query on haplotype panels.** What exists:

- **Trivial and tight:** `Ω(M)` per query, `Ω(occ)` output, `Ω(NM)` build from an uncompressed panel.
- **Space:** the panel's own `δ` / `γ` (substring-complexity and smallest-string-attractor measures) lower-bound any representation from which the panel is recoverable — the standard `δ ≤ γ ≤ b ≤ z ≤ g` chain (Kempa & Prezza; Navarro, *Indexing Highly Repetitive String Collections*, arXiv:2004.02781). An index that also stores the panel cannot go below `δ`. Nothing forbids `o(δ)` for an index that answers queries *without* recovering the panel, and no such bound is known here.
- **Query time in compressed space — the sharpest relevant result:** Kempa & Kociumaka, *Tight Lower Bounds for Central String Queries in Compressed Space* (arXiv:2510.19820, 2025-10-22). In `O(δ log^{O(1)} n)` space and for binary alphabets, they prove a clean dichotomy: **LCE, SA, SA⁻¹ and LCP queries require `Θ(log n / log log n)`**, while **BWT, LF, LF⁻¹, PLCP, Φ, Φ⁻¹ require `Θ(log log n)`**.

The precise reading for P01: divergence-array access is the positional analogue of LCP/LCE, the class with the *harder* bound. If a candidate targets `δ`-bounded space, `O(1)` divergence access is provably out of reach. If it targets `O(r)` words — a strictly larger regime — Bonizzoni–Cozzi–Gao 2026 shows `O(1)` stepping **is** attainable. **The barrier and the achievability result bracket the design space, and the gap between them is where the search should live.** Caveat I am flagging rather than hiding: these are theorems about standard strings and the ordinary BWT; the PBWT is a different (column-wise, positional) transform, and I found no paper that formally transfers the Kempa–Kociumaka bounds to it. Treat it as a strong analogy, not a citation.

### 5. Cheap headroom that is real but is engineering, not science

The arena's own baseline is ~100× off compiled implementations in the `N` dimension. Published Dynamic μ-PBWT answers a long-match query on UKB chr21 (M = 9 793, L = 2000) at **N = 100 000 in 0.2325 s**; the arena baseline needs **0.247 s at N = 980, M = 9 330** — nearly identical wall time for a 100× smaller panel. Since both algorithms are supposed to be `N`-independent per query, most of that gap is interpreter + Rosetta overhead plus a worse constant, not asymptotics. This should be booked on the **engineering** leaderboard and must not be allowed to masquerade as a discovery.

---

## Beat target

A win is one of the following, on **≥ 5 fresh seeds**, at the stated tier, with G0–G3 passed on every seed, counters reported, and the environment fingerprint recorded.

**W1 — Match published SOTA on space (the entry bar, not a discovery).**
`index_bits_per_panel_cell ≤ 1.20` including all retained panel bytes, at `query_s_per_query ≤ 1.25 ×` the uncompressed baseline and `build_s ≤ 4 ×` it, at the **medium** tier. Rationale: Syllable-PBWT is at 1.03–1.14 bits/cell on real panels. Reaching this is reproducing 2023, and should be logged as such.

**W2 — Beat published SOTA on space at no query cost (a real result).**
`index_bits_per_panel_cell ≤ 0.75` — strictly below the bit-packed panel — at `query_s_per_query ≤ 1.25 ×` baseline and `query_work_ratio` no worse than baseline, at **medium and large**. No implemented method achieves this for all-long-match query today.

**W3 — Implement the `O(r)` bound (the headline target).**
`run_ratio = index_words / r ≤ 8` **and** `query_work_ratio ≤ 4` at the medium tier, with `query_work_ratio` flat in `N` (fitted `β ≤ 0.05` across small→large). This is a working implementation of the WABI 2025 `O(r)`-words / `O(m+occ)` result, which does not exist in any repository as of 2026-09-01. Publishable on its own.

**W4 — Change the scaling curve (the target a point measurement cannot capture).**
Fit `log(query_work_ratio) ~ β log N` at fixed `occ/M` across all four tiers. Baseline `β` is measured first and frozen as `β₀`. A win is `β < β₀ − 0.10` with non-overlapping bootstrap CIs over seeds. Because `β₀` should already be ≈ 0 in theory, a positive measured `β₀` is itself a finding — it means the incumbent's claimed `N`-independence does not hold in practice, and the search target becomes recovering it.

**W5 — Build (engineering track, explicitly labelled).**
`build_s ≤ 0.25 ×` baseline at equal `index_bits_per_panel_cell` and equal `query_work_ratio`, medium tier. `build_work_ratio < 1` is **not** achievable from an uncompressed panel (`Ω(NM)` to read the input); a candidate reporting it has changed the input format and must declare it.

**Not a win:** any result at the `small` tier alone; any result at one seed; any result whose gain vanishes when counters are substituted for seconds; any reimplementation of Syllable-PBWT, μ-PBWT, Dynamic μ-PBWT, PBML, RaPID2 or the WABI-2025 construction **presented as novel** — all six are on GitHub or arXiv and are near-certainly in pretraining. Novelty verification against this exact list is a scored gate before any claim leaves the repo.

---

## Mutable surface

The incumbent decomposed into replaceable stages. An agent replaces **one**; the harness supplies the rest.

| # | Stage | Incumbent's choice | Live alternatives in the literature / open |
|---|---|---|---|
| **S1** | **Panel encoding** | dense `uint8` | bit-packing; syllabification into B-bit words (Syllable-PBWT, B ∈ {64,128}); column RLE; per-column wavelet tree; delta-of-columns |
| **S2** | **Positional sort (the `a` recurrence)** | full `a[k]` for all `k`, int32 | run-boundary-only `a`; syllable-level radix sort with rolling hash; block-parallel construction (HP-PBWT); move-table permutation |
| **S3** | **Divergence representation** ← **the space bottleneck; `r_d`/cell = 0.80 measured** | full `d[k]`, int32 | thresholds + Φ sampling (μ-PBWT); sampled `d` + LCE recomputation; PLCP in `O(r)` with `O(1)` access (WABI 2025, unimplemented); hash-based positional comparison replacing `d` entirely (Syllable) |
| **S4** | **Checkpointing / sampling policy** | every column materialised | every `2^t`-th column + replay; run-boundary checkpoints; zero checkpoints (`sub_streaming_pbwt`'s degenerate policy); adaptive by local `r_k` |
| **S5** | **Step / rank-select primitives** | `zeros[k]` + linear scan for `u`/`v` | bitvector rank/select; wavelet tree; **move structure** (Nishimoto–Tabei; Movi; b-move; Move-r) — the constant-time `O(r̃)` stepping of arXiv:2602.13461 lives here |
| **S6** | **Query positioning** | virtual insertion, one site at a time | matching statistics then LEM extraction (WABI 2025); Boyer–Moore–Li skipping (PBML); syllable-level jumps with hash verification |
| **S7** | **Match extension & maximality certification** | bidirectional divergence scan around `t`, early-terminated at `k − cur < L` | block scan with SIMD equality; interval-tree over run boundaries; certify-on-report from matching statistics |
| **S8** | **Output encoding** | per-pair `(j,s,e)` tuples | run-interval batches `(run_id, s, e)` expanded lazily; sorted-by-`s` streaming with deduplication |
| **S9** | **Query scheduling** | one query at a time | batched sweep with regenerated arrays (the streaming variant); query-sorted co-traversal; SIMD across queries |
| **S10** | **Parallelisation / blocking** | none | panel-block decomposition with merge (HP-PBWT: 4× at 30 threads on UKB array data, 22× at 60 cores on 8M randomised haplotypes); PBML-style thread-parallel query (15.9× at 16 threads) |

**Immutable:** the oracle, the generator, the counters, the problem definition, `L` semantics, and the requirement of exactness on the exact track.

**Approximation clause:** a submission may be inexact only if it (a) declares `exact = False` at module scope, (b) is scored solely on the approximate leaderboard against recall at precision ≥ 0.999, and (c) never has a number moved between leaderboards. Silent inexactness is a disqualification, not a low score — this is exactly the failure mode the v1 assets exhibited.

---

## Gameability risks

| # | Attack | Status | Guard |
|---|---|---|---|
| 1 | **Report the panel as a zero-byte index and brute-force at query time.** | **Live and already realised** — `sub_streaming_pbwt` reports `index_bytes = 0`. | `nbytes()` must include every retained byte incl. panel; conservation floor `≥ ceil(NM/8)` unless panel reconstruction is demonstrated on 10 000 spot-checked cells; headline track uses `amortized_s(Qref=10⁴)`. |
| 2 | **Numpy/C rewrite of the identical algorithm scored as discovery.** | **Live and already realised** — 79× "win" with zero novelty. | Counter-based algorithmic leaderboard separate from the seconds-based engineering one; both reported. |
| 3 | **Silent inexactness that the gate cannot see.** | **Live and already realised** — 48/9 600 spurious tuples pass the v1 gate. | G1 uses the scored generator at reduced size; G2 soundness at 100 % coverage every tier; G3 exact row-subsampled completeness. |
| 4 | **Tuning to a fixed instance or a fixed `L`.** | Latent. | Seeds drawn per run from OS entropy; `L` swept over three values per tier; instance never written to a stable path the candidate can cache against. |
| 5 | **Deferring build work into the first query.** | Latent. | First query timed separately from steady state; build and query RSS measured in separate child processes. |
| 6 | **Under-counting the counters.** | Structural. | Counters emitted only by harness-injected accessors in `--counted` mode; a candidate whose counted-mode output differs from its timed-mode output is disqualified. |
| 7 | **Reimplementing a published method as novel.** | **Certain** — Syllable-PBWT (MIT), μ-PBWT (GPL-3), Dynamic μ-PBWT (MIT), PBML, RaPID2, GBWT are all public and in pretraining; the WABI-2025 construction is in arXiv HTML. | Mandatory novelty check against a named prior-art list before any claim; structural-similarity review by a skeptic agent, not a self-report. |
| 8 | **Exploiting msprime's specific model** (constant `Ne`, uniform recombination, infinite-sites-like mutation) rather than repetitiveness in general. | Real. | Randomise `Ne ∈ [5e3, 2e4]`, `ρ ∈ [5e-9, 2e-8]`, and add a demography (bottleneck / expansion) variant per run; confirm rank-preservation on the 1000G chr20/21/22 external-validity run before any claim. |
| 9 | **Overfitting to `occ/M`.** | Real: measured `occ/M` at small tier spans 0.51 (L=100) → 0.003 (L=1200). | L sweep is mandatory; report metrics per `L`, never pooled. |
| 10 | **Output-format shrinkage that hides missing matches.** | Latent, once S8 is opened. | Run-interval output must be expandable to tuples on demand, and G2/G3 run on the expansion. |

---

## Evaluation cost per candidate at the small tier

All measured on the target laptop (Apple M1, macOS 26.6, Python 3.8.8 x86_64 under Rosetta — native arm64 will be faster).

| step | cost | notes |
|---|---|---|
| instance generation, small/seed | **0.82 s** | cacheable per `(tier, seed)` across a generation |
| G1 gate, 8 msprime instances at n=120 (incl. oracle + build + query) | **2.7 s** | at n=200: 7.1 s |
| full exact oracle, small tier | **2.65 s** (vectorised) — was 31.2 s | cacheable per `(tier, seed, L)`; flat in `L` |
| G2 soundness on ~9 600 tuples | < 0.2 s | vectorised |
| candidate build (vectorised PBWT reference) | **0.71 s** | pure-Python reference: 53.0 s |
| candidate query, 20 queries × 3 repeats | **~16 s** | dominated by Python; a C/Rust candidate is far cheaper |
| counted-mode run (once, small tier) | ~5–10× timed mode | budget ~2 min |

**Cold candidate, uncached:** ≈ **25 s** with a vectorised submission. **Warm** (oracle + instance cached across a generation, three `L` values): ≈ **20 s**, of which the candidate's own build+query is the bulk. Counted mode adds ~2 min and is run once per surviving candidate, not per proposal.

Throughput: **~150 candidates/hour/core**; ~1 200/hour on 8 M1 cores. A 5 000-candidate evolutionary run is under 5 hours on the laptop. This clears the arena's "cheap enough to run thousands of times" bar with room.

Medium tier (promotion gate for survivors only): generation 8.0 s, row-subsampled oracle 9.3 s, plus the candidate — budget **2–10 min/candidate**. Large tier is HPC, milestone cadence only.

---

## Readiness scores

| axis | score | rationale |
|---|---|---|
| **Evaluator fidelity** | **5** *(currently 2)* | The truth is a computable function with a 100 %-coverage soundness check and an *exact* row-subsampled completeness check — the strongest evaluator class available in genomics. Scored 2 as-built: G0 is missing, the generator violates its own precondition, and both submissions are silently wrong on the scored tier. Every defect has a verified one-line fix. |
| **Evaluator cost** | **5** | 25 s cold / 20 s warm per candidate at small tier after the 11.8× oracle speedup; ~1 200 candidates/hour on the target laptop; gate is 2.7 s. |
| **Headroom** | **5** | Two 2025–2026 theory papers prove an `O(r)`-words / `O(m+occ)` / `O(1)`-stepping optimum for exactly this problem and neither has an implementation (the WABI authors say so explicitly). Measured on our own instances: `r`/cell = 0.0055 while the index is 64 bits/cell — a 90× gap — and the entire gap is the divergence array (`r_d`/cell = 0.80). No published implementation occupies the intersection of {exact all-long-match, ≤1 bit/cell, not slower than uncompressed}. |
| **Infra fit** | **4** | Small and medium tiers both run on the M1 laptop (medium generation 8.0 s, peak ~1 GB); large is a natural HPC job. Pure-Python/numpy assets, no external tool install required to *score*. Docked one point: the current interpreter is x86_64 under Rosetta on Python 3.8.8 (EOL), so every published timing must be regenerated on native arm64; and none of the C++ incumbents has been *confirmed* to build on Apple Silicon. |

---

## Confidence notes

**Verified by direct measurement on this machine (highest confidence):**
- Generator emits non-biallelic sites: max allele 2 at 17/9 330 sites (small/42) and 9/8 797 (small/7); tiny tier is clean. Reproduced twice.
- `baseline_pbwt` and `sub_vectorized_pbwt` both emit **48 spurious tuples** out of 9 600 at small/42/L=300; oracle truth is **9 552**; zero missing, zero duplicates; the spurious intervals fail segment equality. `sub_streaming_pbwt` reproduces the identical 48.
- Root cause isolated by instrumented state comparison: at k=808 in the smallest failing case the virtual-insertion position is 101 vs true 97 and `d_above` is 665 vs true 800, on a column where allele counts do not sum to `N` — i.e. the binary `allele == 0 else` branch mis-sorts a three-allele column.
- Fix verified: `msprime.BinaryMutationModel(state_independent=False)` → **0/8 seed failures** at each of n=120, 200, 400 (previously 4/8, 2/8, 1/8).
- v1 gate generator produces **0/30 failures** up to N=400/M=4000 — it cannot see the bug.
- Run statistics: `r` = 50 712 (5.4/column) and `r_d`/cell = 0.797 at small/42; 48 372 / 5.5 / 0.813 at small/7.
- Vectorised oracle: identical output, **31.20 s → 2.65 s**.
- Medium tier realised shape `9950 × 57465`; generation 8.0 s; 256-row exact oracle 9.3 s.
- Host is Apple M1 `arm64`; Python reports `x86_64` → Rosetta 2, Python 3.8.8.
- `https://ftp.1000genomes.ebi.ac.uk/.../20220422_3202_phased_SNV_INDEL_SV/` listing fetched live with exact filenames and sizes.
- Repository liveness via GitHub API: `richarddurbin/pbwt` pushed 2025-11-20 (no license file); `ZhiGroup/Syllable-PBWT` 2023-08-23 MIT; `dlcgold/muPBWT` 2026-09-01 GPL-3.0; `ucfcbb/Dynamic-mu-PBWT` 2025-05-05 MIT; `ucfcbb/RaPID2` 2026-04-08 MIT; `uwaiseibna/PBML` 2026-04-07 (no license); `jltsiren/gbwt` 2026-08-19 MIT. Durbin's `pbwt` CLI confirmed to expose `-longWithin`, `-maxWithin`, `-matchNaive`, `-matchIndexed`, `-matchDynamic`; Makefile is `-g -O3`, htslib only, no x86 intrinsics.
- arXiv metadata verified for 2505.15698 (Sanaullah/Zhi/Zhang, 2025-05-21), 2602.13461 (Bonizzoni/Cozzi/Gao, 2026-02-13), 2510.19820 (Kempa/Kociumaka, 2025-10-22).

**Read from primary sources, not independently reproduced (high confidence, but they are the authors' numbers on their hardware):** all memory and runtime figures in the incumbent table — Syllable-PBWT 162/91.4 GB and 7.9/4.1 GB; μ-PBWT 11.06 GB on UKB chr20; Dynamic μ-PBWT 1.5/1.84 bytes-per-site-per-haplotype and the 0.2325 s vs 1.0774 s long-match-query table; L-PBWT-Query 6 s/query and 4.7 TB; RaPID2 65 513 s / 822 GB; PBML's percentage speedups; GBZ Table 1.

**Computed by me from published inputs (arithmetic shown above, check it):** every bits-per-panel-cell figure. These are divisions of a published byte count by a published cell count; they are not published as such by any of the papers.

**Explicitly could NOT confirm:**
1. **Apple Silicon buildability of any C/C++ incumbent.** Syllable-PBWT's README says Linux + GCC ≥ 5 and documents no intrinsics; Durbin's `pbwt` is plain C with htslib. Both *should* build on arm64 macOS. **Neither was compiled.** Do this before quoting any head-to-head timing.
2. **Whether Syllable-PBWT's runtime advantage is tabulated anywhere.** The paper presents runtime as figures (Fig. 4), not a table; I have its memory numbers precisely and its timing only qualitatively ("faster than existing solutions").
3. **The 2026 Shakya/Sanaullah/Zhi/Zhang J. Comput. Biol. paper** (`10.1177/15578666251385570`) returned HTTP 403. I used the bioRxiv/PMC version of the same work (Dynamic μ-PBWT, RECOMB 2025) instead; the journal version may contain updated numbers.
4. **HP-PBWT's numbers** (bioRxiv 2025.02.04.636317) come from a search-result summary — the PDF would not extract. The "4× at 30 threads / 22× at 60 cores" figures are second-hand and should be re-verified before being cited.
5. **GB vs GiB** in μ-PBWT's "11.06 GB". If GiB, the derived figure is 0.0230 rather than 0.0214 bits/cell. Does not change any conclusion.
6. **Formal transfer of the Kempa–Kociumaka lower bounds to the PBWT.** Their theorems are about standard-string primitives (LCE/SA/LCP vs BWT/LF/Φ/PLCP) in `O(δ polylog n)` space. I found no paper that proves the analogous statement for the positional transform. I have framed it as a strong analogy and flagged it in-line; do not cite it as a PBWT lower bound.
7. **Whether the arena's per-site query throughput is genuinely comparable to Dynamic μ-PBWT's.** The arena baseline is 26.5 µs/site at N=980; the published Dynamic μ-PBWT figure is 23.7 µs/site at N=100 000. Both algorithms are nominally `N`-independent per query, so the comparison is meaningful in principle — but the arena's `occ/M` (0.05) may differ from theirs, and it is unclear whether their number includes index load. Treat as suggestive, not as a measurement.

**Files that must change before this spec is live:**
`problems/P01_haplotype_long_match/generate.py` (BinaryMutationModel + assertion), `.../oracle.py` (vectorised form), `.../evaluate.py` (G0–G3, child-process RSS, phase split, amortisation curve, counted mode), `.../baseline_pbwt.py` (assert biallelic; honest `nbytes()`), `.../sub_streaming_pbwt.py` (honest `nbytes()`), `.../problem.yaml` (recalibrated beat targets, realised tier shapes), `harness/measure.py` (`TrackedArray` accessors backing the counters). `.../results.jsonl` should be truncated: every record in it is a measurement of a non-exact program on invalid instances.