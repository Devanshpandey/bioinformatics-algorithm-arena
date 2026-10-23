I have everything verified. Writing the spec.

## Formal statement

**P02 — Long approximate haplotype match discovery (IBD segment detection).**

Given a phased haplotype panel `H` of `N` binary haplotype vectors over `M` ordered biallelic markers, together with a strictly increasing genetic map `g: {1..M} → ℝ₊` (cM), find the set

```
S = { (i, j, a, b) : i < j,  g(b) − g(a) ≥ L_min,
                     H[i, a..b] ≈ H[j, a..b] under an error model }
```

and report each element's endpoints. `≈` is not fixed by the problem: it is the thing the algorithm gets to define. The scientific target is not string matching but **inference**: the reported `(i, j, a, b)` should coincide with the maximal interval over which haplotypes `i` and `j` descend from a single common ancestral haplotype no more than `T_cap` generations back.

Two properties make this a real algorithms problem rather than a tuning problem:

1. **The pair space is quadratic.** `C(N,2)` grows to 1.2 × 10¹¹ pairs at UK Biobank scale, but the number of pairs sharing any ≥2 cM segment is a vanishing fraction of that. Every serious method is an *output-sensitive candidate-generation* algorithm: the science is in avoiding pairs, not in comparing them.
2. **The match is approximate and the errors are structured.** Genotyping error is roughly i.i.d. per marker; *phase switch error* is not — it swaps the continuation of a true segment onto the homologous haplotype, fragmenting one true segment into two half-length pieces on two different haplotype pairs. Mutation and gene conversion inject isolated discordances. An exact-match seeding strategy is brittle to all three.

The submitted artifact is a program, not a segment list. It is run on instances it has never seen, generated from fresh seeds.

---

## Input / output / submission API

### Instance directory (read-only, hash-pinned)

```
instance_<tier>_<seed>/
  panel.vcf.gz          # phased, biallelic, no missing genotypes, ploidy 2
  panel.map             # PLINK format: chrom, marker_id, cM, bp
  manifest.json         # N_haplotypes, M_markers, contig_len_bp, L_min_cM,
                        # T_cap_generations, error_model, instance_sha256
```

`panel.vcf.gz` is the *observed* panel — it already carries the injected genotyping and phase-switch error. The truth files live outside the sandbox and are not readable by the candidate (enforced in the correctness gate).

### Submission

A directory with an executable entry point and a declared interface:

```
submission/
  run.sh                # required: ./run.sh <instance_dir> <out.ibd> <threads> <mem_gb>
  build.sh              # optional: compile step, timed separately
  meta.json             # {"name", "language", "stages_replaced": [...],
                        #  "declared_min_output_cM", "deterministic": true}
  counters.json         # written by run.sh; see Implementation-independent counters
```

Any language. Stdout/stderr are captured. The process is run with a wall-clock cap, an RSS cap, and no network.

### Output format

Tab-separated, one segment per line, matching hap-IBD's column order so that the incumbent needs no adapter:

```
sample1  hap1  sample2  hap2  chrom  start_bp  end_bp  length_cM
```

`hap ∈ {1,2}`. `length_cM` must equal `g(end_bp) − g(start_bp)` under the supplied map to within 1e-6 (checked). Segments may be reported in any order. Gzipped output is accepted and its compressed size is what counts toward the output-size metric.

### Reference harness

The harness supplies each candidate with a loader (`panel.vcf.gz` → an in-memory `M × N` bit-packed matrix plus the cM vector) and a writer, both instrumented. Candidates may bypass them; bypassing forfeits the automatic counter instrumentation and requires the candidate to populate `counters.json` itself, which the gate cross-checks (see below).

---

## Ground truth (EXACT vs CURATED vs PROXY, and justify)

**EXACT, by construction, with one non-obvious correction that the rest of this section is about.**

The truth is genealogical IBD read directly off the msprime coalescent simulation that generated the panel. No caller, no consensus, no downloaded answer key. Fresh seeds per evaluation round mean there is no public artifact an LLM could have memorized.

### The definitional problem, stated precisely

The audit flagged the right thing. `tskit.TreeSequence.ibd_segments()` does **not** return what a practitioner means by an IBD segment. Verified locally against tskit 0.5.7, the docstring reads:

> A pair of nodes `(u, v)` has an IBD segment with a left and right coordinate `[left, right)` and ancestral node `a` iff the most recent common ancestor of the segment `[left, right)` in nodes `u` and `v` is `a`, **and the segment has been inherited along the same genealogical path (ie. it has not been broken by recombination)**. The segments returned are the longest possible ones.
>
> Note that this definition is purely genealogical — allelic states *are not* considered here. **If used without time or length thresholds, the segments returned for a given pair will partition the span of the contig.**

Two consequences, both fatal if ignored:

**(1) Without a time cap, every pair is IBD everywhere.** I confirmed this: for 100 haplotypes over 10 Mb, `ibd_segments()` with no thresholds returns 31,934,223 segments across 4,950 pairs, and for pair `(0,1)` the segment spans sum to exactly 10,000,000 bp — the whole contig. "Genealogical IBD" with no cap is a vacuous truth set.

**(2) "Same genealogical path" is edge-wise, not node-wise, and it fragments true segments.** tskit splits a segment whenever the inheritance from ancestor `a` is represented by two edges, even when the MRCA node is unchanged and the intervals are contiguous. I measured the damage:

| | max_time = 100 g, 200 haplotypes, 20 Mb |
|---|---|
| adjacent segment boundaries that are **not** MRCA changes | **532 of 615 (86.5%)** |
| raw segment count | 3,230 |
| after merging adjacent same-MRCA-node segments | 2,156 (**−33.3%**) |
| segments ≥ 2 Mb (≈2 cM), raw | 175 |
| segments ≥ 2 Mb, merged | **250 (+43%)** |

A truth set built from raw `ibd_segments()` output would contain **43% fewer long segments than actually exist**, and would score a perfectly correct caller as massively over-calling and over-extending. This is the single most likely way to build a silently wrong evaluator for this problem — and note that a naive implementation gives no error, just wrong science.

### The resolution

**Definition (scored truth).** A true IBD segment for haplotype pair `(u,v)` is a maximal interval `[l, r)` such that at every point in `[l, r)` the MRCA of `u` and `v` is the *same node* `a`, with `time(a) ≤ T_cap`, and `g(r) − g(l) ≥ L_min`.

This is node-wise, not path-wise. It is recovered exactly by post-processing tskit's output:

```python
segs = ts.ibd_segments(max_time=T_cap, store_segments=True)
# merge adjacent segments with identical MRCA node and touching coordinates
```

**I verified this merge is exactly equivalent to the definition**, not approximately. Against a brute-force scan over every tree computing `tree.mrca(u,v)` and accumulating maximal same-node runs under the time cap: 112 pairs, **0 mismatches, 114 segments by both routes, identical coordinates**. The merge is not a heuristic cleanup; it is the correct reduction, and it runs at C speed instead of the `O(N² · trees)` Python scan.

**This also means the published benchmark's truth generator and mine agree in definition but not in fidelity.** The GigaScience benchmark (`ZhiGroup/IBD_benchmark`, `Simulation/msprime_simulation.py`) never calls `ibd_segments()`. It scans trees calling `tree.mrca()` for all pairs — correctly node-wise, thus dodging the fragmentation trap — but it (a) **subsamples trees, only examining a tree every 5,000 bp**, so boundaries are quantized and short segments can be missed entirely, and (b) **applies no MRCA time cap at all**, relying solely on a 1 cM length filter. Our evaluator is exact where theirs is quantized, and is orders of magnitude faster.

### Setting `T_cap`: it must be non-binding, not definitional

The temptation is to declare "true IBD = MRCA within 25 generations" and be done. That is wrong, and I can show it. Measuring the MRCA-time distribution of merged segments as a function of their length (300 haplotypes, 30 Mb, `max_time=500`):

| segment length | median MRCA time | 95th pct MRCA time |
|---|---|---|
| 2–3 cM | 54.9 g | **130.3 g** |
| 3–5 cM | 35.8 g | 89.7 g |
| 5–10 cM | 17.6 g | 43.1 g |
| ≥10 cM | 6.0 g | 19.6 g |

A 2 cM segment routinely comes from an ancestor 100+ generations back. Capping at 25 g — the value suggested by the textbook `E[length] = 100/(2g)` cM relation — would delete most genuine 2 cM segments from the truth set and score every caller that finds them as a liar.

**So: `L_min` defines the truth set; `T_cap` is only a computational filter, and must be chosen so that it is provably non-binding.** For `L_min = 2.0 cM` we set `T_cap = 250` generations (95th percentile is 130 g, so 250 g has ~2× headroom). The evaluator *self-audits* this: it computes the fraction of scored truth segments whose MRCA time falls within 10% of `T_cap`, and **fails the round** if that fraction exceeds 0.1%. If you raise `L_min`, `T_cap` may drop; if you lower it, `T_cap` must rise, and the audit enforces it automatically rather than trusting the spec author.

Measured impact of getting this wrong, same panel:

| `max_time` | merged segments | frac ≥2 cM | frac ≥5 cM |
|---|---|---|---|
| 25 g | 1,851 | 51.2% | 20.5% |
| 50 g | 6,863 | 29.9% | 7.0% |
| 100 g | 26,252 | 11.3% | 1.9% |
| 200 g | 100,667 | 3.3% | 0.5% |

### Genealogical truth vs. what practitioners want — the honest answer

The audit's framing was: a caller can be "wrong" against genealogical truth while being right about what practitioners want. My position is that this is true only for the *precision* side, and it is fixable by construction rather than by fudging the metric.

- **Recall side:** no conflict. A ≥2 cM segment that is genealogically IBD at `T_cap = 250 g` is exactly what every downstream application (relatedness, IBD mapping, `IBDNe`, founder-effect detection) wants to find. Missing it is a real miss.
- **Precision side:** real conflict. A caller reporting a genuine 1.7 cM segment when `L_min = 2.0` is making a *length* error, not fabricating. A caller reporting a segment whose MRCA is 300 generations back is reporting something real that our cap happens to exclude.

The evaluator therefore uses **two truth sets**, which removes the conflict rather than papering over it:

- `T_min` — merged, `time ≤ T_cap`, length `≥ L_min`. **Recall is scored against this.**
- `T_any` — merged, `time ≤ 2·T_cap`, **no length filter**. **Precision is scored against this.** Reported cM overlapping anything in `T_any` is not counted as a false positive.

A false positive is then unambiguous and defensible: reported sequence that is identical-by-state but shares no common ancestor within 500 generations. Nobody wants those. Conversely, no caller is punished for being slightly off on a length threshold or for finding a slightly-too-old real segment.

**Why simulated, not curated.** Real-data alternatives are all proxies: parent–offspring chromosome coverage (only validates one trivially-detectable configuration), caller consensus (circular — it enshrines the incumbents' shared blind spots, notably their shared fragmentation under phase error), and pedigree cohorts (small, private, memorizable). Simulation gives the exact ARG for free, and the error model is a knob rather than a confound.

**What simulation does *not* give**, stated plainly: the coalescent is a model. Real panels have population structure, cryptic relatedness, non-crossover gene conversion, reference bias, and array ascertainment that no simulator fully reproduces. A method that wins here has demonstrated an algorithmic advantage under a specified generative model; it has not been validated for production. The arena scores algorithms, and the promotion criterion to "real" is a separate, later gate.

---

## Instance generation and tiers

All generators are deterministic given `(tier, seed)`. Verified on this machine: **tskit 0.5.7, msprime 1.3.1, macOS 26.6, arm64**.

```
pip install "msprime==1.3.1" "tskit==0.5.7" "numpy>=1.24"
```

> **Practical note found while testing:** the default `python3` on the development laptop (an x86 Anaconda install) reports `platform.machine() == 'x86_64'`, i.e. Python 3.8.8 running under Rosetta even though `uname -m` is `arm64`. All timings below are therefore *pessimistic*. Build the arena on a native arm64 interpreter (`conda create -n arena python=3.12` from an arm64 Miniforge, or Homebrew `python@3.12`) or every runtime number will be silently inflated.

### Demography

The Gutenkunst *et al.* out-of-Africa model, sampling from CEU. This matches the GigaScience benchmark exactly, so numbers are comparable to the published literature. The parameterization is reproduced verbatim in `ZhiGroup/IBD_benchmark/Simulation/msprime_simulation.py` (lines building `out_of_africa_model_demography`), which I read in full; the arena vendors its own copy rather than importing it, pinned by hash.

### Genetic maps — all verified live (HTTP 200, 2026-09-01)

| Map | URL |
|---|---|
| HapMap II GRCh37 chr20 (benchmark-identical) | `https://raw.githubusercontent.com/ZhiGroup/IBD_benchmark/main/Simulation/genetic_map_GRCh37_chr20.txt` |
| PLINK-format GRCh37 maps, all chromosomes (Browning lab) | `https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh37.map.zip` |
| deCODE hg19 + hg38 liftover | `https://github.com/ZhiGroup/RaPID/tree/master/genetic_maps` |

### Tier ladder

| | **Small** (laptop inner loop) | **Medium** (laptop / 1 node) | **Large** (HPC) |
|---|---|---|---|
| diploid samples | 500 | 5,000 | 50,000 → 250,000 |
| haplotypes `N` | 1,000 | 10,000 | 100,000 → 500,000 |
| sequence | 20 Mb, uniform `r = 1.2e-8` | chr20, HapMap II GRCh37 (~63 Mb) | chr20, then chr1 |
| markers `M` | ~82,000 (measured) | ~600,000 seq / ~17,200 array | ~10⁶ seq / ~60,000 array |
| `L_min` | 2.0 cM | 2.0 cM | 2.0 and 7.0 cM |
| `T_cap` | 250 g | 250 g | 250 g |
| instance + truth build | **22.1 s measured** | ~8 min | node-hours, built once |
| candidate wall cap | 120 s | 30 min | 6 h |
| RSS cap | 8 GB | 32 GB | 200 GB |

The small tier is the search loop. Everything about it was measured, not estimated:

```
sim = 3.5 s | vcf_write = 4.0 s (167 MB) | gzip = 5.0 s (7 MB)
ibd_segments = 0.33 s | merge = 9.3 s | TOTAL = 22.1 s
raw segments 976,405 → merged 382,614 | 82,089 sites | 1,000 haplotypes
```

The 9.3 s merge is pure-Python per-pair looping and is the only slow step; a vectorized NumPy merge over the flattened segment arrays removes essentially all of it. Instances are built once per seed and cached, so this cost is amortized over every candidate scored on that seed.

### Exact generator command

```bash
python arena/p02/generate.py \
    --tier small --seed 20260901 \
    --demography ooa_gutenkunst --sample-pop CEU \
    --sequence-length 20e6 --recombination-rate 1.2e-8 \
    --mutation-rate 1.38e-8 \
    --L-min-cM 2.0 --T-cap-generations 250 \
    --genotype-error 0.001 --phase-switch-error 0.0017 \
    --out instances/small_20260901/
```

Mutation rate `1.38e-8` and genotype error `0.001` are the GigaScience benchmark's values. The `0.0017` switch-error rate is their reported "0.17% average switching error" condition. The generator emits `panel.vcf.gz`, `panel.map`, `manifest.json`, and — written outside the instance directory — `truth_min.parquet`, `truth_any.parquet`, `truth_audit.json`.

**Error injection order matters and is fixed:** simulate ancestry → simulate mutations → extract truth from the *clean* tree sequence → downsample to array sites (medium/large only) → inject genotyping error → inject phase switch error → write VCF. Truth is always read off the genealogy, never off the corrupted panel.

**Conditions swept per tier** (each a separate instance, scored separately and in aggregate): `{sequence, array} × {error-free, 0.1% geno, 0.4% geno, 0.17% switch, 0.1% geno + 0.17% switch}`. The error-free sequence condition is the sanity floor — every tool should be near-perfect there, and a candidate that is not is broken.

---

## Correctness gate

Binary. Every check must pass before any score is computed. A failure returns `INVALID` with a reason code and no score — not a low score, no score, so that failure carries no gradient a search process could exploit.

**G1 — Well-formedness.** Parses; all sample IDs present in `panel.vcf.gz`; `hap ∈ {1,2}`; `start_bp < end_bp`; both endpoints are marker positions present in `panel.map`; `chrom` matches the manifest.

**G2 — Length consistency.** For every row, `|length_cM − (g(end_bp) − g(start_bp))| < 1e-6`. Catches candidates reporting inflated lengths against a private map.

**G3 — Non-overlap.** For a given ordered haplotype pair, reported intervals must be pairwise disjoint. Blocks the "report nested and shifted copies of the same segment to maximize overlap mass" attack.

**G4 — Threshold honesty.** No reported segment shorter than `declared_min_output_cM` in `meta.json`, and `declared_min_output_cM ≥ L_min`. A candidate may not quietly report 0.5 cM segments to inflate recall.

**G5 — Determinism.** Two independent runs on the same instance with the same thread count produce byte-identical sorted output. Multi-threaded candidates must be deterministic across thread counts too, or must declare `"deterministic": false`, which bars them from the leaderboard and routes them to a separate exploratory pool.

**G6 — Isolation.** The candidate process runs with the truth directory unmounted and the network disabled. The instance directory is verified by `instance_sha256` before and after the run — a candidate that writes into it fails.

**G7 — Planted positive control.** Every instance carries injected controls invisible in the manifest: (a) 5 duplicated haplotype pairs (exact copies, so IBD over the entire contig), (b) 5 pairs with a synthetically spliced segment of exactly known cM length. The candidate must report ≥95% of the duplicate span and must recover each spliced segment to within 20% of its true length. This catches a caller that scores well on aggregate mass while being structurally broken.

**G8 — Planted negative control.** 5 haplotype pairs drawn from a deeply diverged population (split at `T_AMH`) are injected. Reporting any segment ≥ `L_min` on more than one of these fails the gate. This is the anti-hallucination check.

**G9 — Resource compliance.** Wall time and peak RSS under the tier caps, measured externally (see below), not self-reported.

**G10 — Counter plausibility.** `counters.json` present and internally consistent: `n_pairs_touched ≥` the number of distinct pairs actually reported, and `n_allele_ops ≥ n_pairs_touched`. On a dedicated 64-haplotype micro-instance where the exact counts are derivable by exhaustive enumeration, a candidate's reported counters must match the harness's independent instrumentation within 5%.

---

## Accuracy metric

The core question — *a true 10 cM segment reported as two 4 cM segments is neither a hit nor a miss* — is answered with a closed-form, threshold-free, parameter-free measure rather than an arbitrary 50%-overlap rule.

### Contiguity-weighted recall and precision

Let `t` range over `T_min` (true segments, per haplotype pair) and `r` over the candidate's reported segments for that same pair. All lengths in cM under the supplied map. `|t ∩ r|` is the cM overlap.

```
                Σ_t  Σ_r |t ∩ r|² / |t|                    Σ_r  Σ_t' |t' ∩ r|² / |r|
       R_c  =  ─────────────────────────          P_c  =  ──────────────────────────
                      Σ_t |t|                                     Σ_r |r|
```

where `t'` ranges over `T_any` (the no-length-filter, `2·T_cap` truth set) for the precision term.

**Primary accuracy score:  `F1_c = 2·P_c·R_c / (P_c + R_c)`.**

### Why this is the right answer to the fragmentation question

Rewrite the inner recall sum. If piece `r` covers fraction `p_r = |t ∩ r| / |t|` of true segment `t`, then that segment's contribution is `|t| · Σ_r p_r²` — the **Simpson concentration index** of how the true segment's mass is distributed across reported pieces. It is a standard, well-understood quantity, not an invention.

Behaviour on the audit's exact scenario, a true 10 cM segment:

| reported as | union coverage | `Σ p_r²` | credit |
|---|---|---|---|
| one 10 cM segment | 1.00 | 1.00 | **100%** |
| one 8 cM segment | 0.80 | 0.64 | 64% |
| **two 4 cM segments** | **0.80** | **0.32** | **32%** |
| two 5 cM segments | 1.00 | 0.50 | 50% |
| ten 1 cM segments | 1.00 | 0.10 | 10% |

Properties that matter:

- **Neither a hit nor a miss.** Two 4 cM pieces score 0.32 — strictly better than a miss, strictly worse than the 0.64 earned by a single contiguous 8 cM call. Exactly the requested behaviour, and it falls out of the formula rather than being bolted on.
- **Reduces to coverage when unfragmented.** With one reported piece, `R_c` is just fractional coverage. The metric adds nothing when there is nothing to add.
- **No free parameters.** No 50% overlap threshold, no merge-gap tolerance, no matching rule. Nothing for an agent to reverse-engineer and nothing for the spec author to get wrong.
- **Splitting can never help.** `Σ p_r²` is maximized by a single piece for any fixed union coverage, so there is no reported-segment partition that beats the contiguous call. Symmetrically for `P_c`, over-reporting dilutes.
- **Merging beyond truth can never help either.** A candidate that stitches two genuinely separate true segments into one long call gains on `R_c` for each but is charged on `P_c` for the intervening span, which lies outside `T_any`.

### Secondary metrics (reported, not scored)

Computed alongside so results are comparable to the published literature and so failure modes are legible:

- **GigaScience-compatible power, accuracy, recall, length accuracy, length discrepancy** using their definitions (50% overlap threshold, best-matching segment, RMSE in cM), binned by true segment length `[2,3), [3,5), [5,7), [7,10), [10,∞)`. Reported so an arena result can be placed next to Tang *et al.* 2022 directly.
- **Breakage index** `B` = mean number of reported segments contributing to each ≥50%-covered true segment. hap-IBD's characteristic failure under phase error shows up here as `B > 1`.
- **Signed length bias** in cM, which separates over-extension (IBS running past IBD, the failure `ibd-ends` was built to fix) from under-extension.
- **Haplotype-level vs individual-level** `F1_c`. The gap between them isolates phase-switch handling specifically.

### Evaluation tool

No published tool implements `F1_c`. The arena ships `arena/p02/score.py`, which is hash-pinned and frozen at the start of each search round. It also emits the GigaScience metrics.

The existing tool, for cross-checking only, is `IBD_Benchmark` from `https://github.com/ZhiGroup/IBD_benchmark` — a C# program (`IBD_Benchmark/*.cs`, precompiled binaries `IBD_BM.exe`, `IBD_BM_u18`, `IBD_BM_u22` in `IBD_Benchmark_ExperiencePackage/`) that natively parses FastSMC, hap-IBD, iLASH, RaPID and TPBWT output and emits *Accuracy, Length Accuracy, Length Discrepancy, Recall, Power, Accumulative Recall, Accumulative Power* binned by cM. The arena runs it as an independent second opinion on the secondary metrics. The shipped Linux binaries are x86-64; on Apple Silicon use `dotnet` from source or run it on the cluster.

```bash
python arena/p02/score.py \
    --truth-min instances/small_20260901/truth_min.parquet \
    --truth-any instances/small_20260901/truth_any.parquet \
    --map       instances/small_20260901/panel.map \
    --pred      runs/cand_0413/out.ibd \
    --L-min-cM 2.0 --report-bins 2,3,5,7,10 \
    --out runs/cand_0413/score.json
```

---

## Resource metrics

Accuracy alone is never the score. Four resource axes are first-class, all measured externally by the harness — never self-reported.

**Build vs query, and why the split is real here.** IBD detection is conventionally a single batch job, but the interesting algorithms are not. RaPID, iLASH and TPBWT all construct a reusable index (random projections, LSH tables, templated PBWT) whose cost is separable from querying it. The arena measures the split explicitly and rewards it, because a panel-index that supports incremental addition of new samples is worth far more to a biobank than a monolithic rerun:

- **`t_build`** — wall time from process start to the point where the candidate signals `INDEX_READY` on a named pipe (optional; candidates that do not signal are charged everything to query).
- **`t_query`** — `INDEX_READY` to exit.
- **`t_total`** — the number that gates the beat target.
- **`t_incremental`** — a bonus measurement: time to add 10% more haplotypes and report only the new pairs. No incumbent supports this. A candidate that does earns a separate leaderboard entry.

**RAM.** Peak RSS of the process tree, sampled at 20 Hz by the harness via `psutil` on macOS and read from `cgroup.memory.peak` under Slurm on the HPC node. Reported as `peak_rss_gb`. Java candidates are additionally charged their `-Xmx`, since a JVM that requires a 110 GB heap has that requirement whether or not it touches every page. Memory is where the incumbent is weakest and where the tier ladder bites hardest.

**Index size.** Bytes on disk after `INDEX_READY`, plus peak scratch. For candidates with no persisted index this is 0 and the axis is inert.

**Output size.** `gzip -9` compressed bytes of `out.ibd`. This is not a throwaway axis: hap-IBD emits **231.5 billion autosomal segments** at 2 cM on UK Biobank, and at ~40 bytes per line uncompressed that is multiple terabytes. Output representation is an untouched part of the design space — a candidate that emits a compressed pair-graph with implicit intervals, plus a decoder, competes here legitimately as long as the decoder is included in the timing and the decoded stream passes the correctness gate.

**Scaling exponents.** For each of `t_total`, `peak_rss_gb`, `output_bytes`, fit `log(y) = α·log(N) + c` across the tier ladder (≥4 sizes) with bootstrap CIs on `α`. The exponent — not any single point — is the primary object of interest. A candidate that is 1.3× slower at the small tier with `α = 1.1` against the incumbent's `α = 1.8` is the win this arena exists to find.

---

## Implementation-independent counters

The quantity that measures algorithmic work for this problem, independent of language, compiler, SIMD width and constant-factor tuning, is **how much of the quadratic pair space the algorithm touched**.

**C1 — `n_pairs_touched`.** The number of distinct unordered haplotype pairs `{i,j}` for which the candidate performed *any* operation whose result depends on data from both `i` and `j`. This is the fundamental counter.

The interpretable form is **pair selectivity**:

```
selectivity = n_pairs_touched / |{ pairs with ≥1 true segment ≥ L_min }|
```

- Oracle floor: `1.0` — touch only pairs that are genuinely related.
- Naive ceiling: `C(N,2) / |related pairs|`, which at the small tier is 499,500 / (a few thousand) ≈ 10²–10³, and at UK Biobank scale is ~10⁶.

This is the number that decides whether a method scales, and it cannot be improved by rewriting in C.

**C2 — `n_allele_ops`.** Total `(pair, marker)` allele-equality tests: every time the candidate compares haplotype `i`'s allele to haplotype `j`'s at a marker, in seeding, extension, or verification. Vectorizing 64 comparisons into one word operation still counts 64. This is what separates "cheap candidate generation, expensive verification" from the reverse.

**C3 — `n_symbol_ops`.** Per-symbol work that does *not* involve a pair: PBWT prefix/divergence array updates, hash or sketch evaluations, sort keys. Counted separately because it is the sub-quadratic budget an algorithm spends to buy a lower C1.

**Headline derived quantity — work per unit of correct output:**

```
work_per_true_cM = (n_allele_ops + n_symbol_ops) / (total cM in T_min correctly recovered)
```

A method that halves runtime by loosening thresholds and dropping recall does not improve this. A method with a genuinely better candidate-generation scheme does.

**Enforceability — stated honestly.** These counters are instrumented automatically only for candidates using the supplied loader and match primitives. A candidate that rolls its own inner loop must increment them itself, and nothing stops it lying. Three mitigations, none complete:

1. G10 cross-checks against the harness's independent instrumentation on a 64-haplotype micro-instance where exact counts are derivable by exhaustive enumeration.
2. Counters must be *consistent with observed scaling*. A candidate claiming `C1 ∝ N` while its wall time scales as `N^1.9` across four tiers is flagged for manual review.
3. Counters are **diagnostic and reported, never part of the beat target**. The enforceable objectives are wall time, RSS and output size, measured externally. The counters exist to explain *why* a candidate won and to distinguish a real algorithmic advance from a constant-factor engineering win — which is the distinction this whole arena is for.

---

## Incumbent(s) to beat

### Primary incumbent: hap-IBD

**Why it and not the others.** It is the current default at biobank scale, it is the only one that completed the full UK Biobank chromosome 20 at a 2 cM threshold in the head-to-head that its own authors ran, and it is the one whose authors published a *separate tool* to fix its endpoints — which is an unusually clean admission of where the headroom is.

**Distribution format (verified 2026-09-01).** A single precompiled Java JAR:

```
https://faculty.washington.edu/browning/hap-ibd.jar
HTTP 200 · 270,829 bytes · Last-Modified: Fri, 16 Jun 2023 00:29:09 GMT
```

Source at `https://github.com/browning-lab/hap-ibd` (Java, 55 stars, last push 2024-06-11, README last updated 2023-06-14). **The repository has no GitHub releases and no tags** — the JAR on the faculty page is the distribution channel, and versioning is by the string printed in the first line of the output log. Pin by SHA-256 of the JAR, not by a version number.

**Apple Silicon: yes, natively, no Rosetta.** The JAR is architecture-independent bytecode requiring Java ≥ 1.8; native execution needs only an aarch64 JVM. Homebrew's `openjdk` is at **26.0.2.1 with an `arm64_tahoe` bottle** (verified via `brew info --json=v2`), and Eclipse Temurin ships macOS aarch64 builds. **Caveat: I could not execute hap-IBD on this machine — no JVM is installed** (`java -version` → "Unable to locate a Java Runtime"). The architecture-independence argument is sound but the end-to-end run is unverified; install `brew install openjdk` and confirm before freezing the baseline.

**Defaults** (from the README): `min-seed=2.0` cM, `min-extend=min(1.0, min-seed)`, `min-output=2.0` cM, `min-markers=100`, `min-mac=2`, `max-gap=1000` bp, `nthreads`=core count.

```bash
java -Xmx8g -jar hap-ibd.jar \
     gt=panel.vcf.gz map=panel.map out=hapibd_out \
     min-seed=2.0 min-output=2.0 nthreads=8
```

**Algorithm, as pipeline stages** — read directly from the preprint text, since this defines the mutable surface:

1. **Ingest / compression.** VCF → binary reference format v3: low-frequency variants stored as the haplotype indices carrying the minor allele; higher-frequency variants as unique allele sequences plus a mapping vector.
2. **Seeding.** PBWT sweep maintaining Durbin's prefix and divergence arrays; the divergence array yields *all* exact-IBS segments ending at each marker (Durbin's Algorithm 3) that exceed the minimum seed length. Seeds are **exact matches** — zero mismatches tolerated inside a seed.
3. **Seed filtering.** `min-markers` (default 100) and `min-mac` (default 2) gate seeds, guarding low-marker-density regions.
4. **Extension.** Each seed is extended left and right across a non-IBS gap of at most `max-gap` bp to an adjacent IBS segment of at least `min-extend` cM; repeated until no further extension is possible.
5. **Deduplication.** If an extending IBS segment is itself a valid seed, the extension is abandoned, since the same maximal segment will be produced by an earlier seed.
6. **Parallel decomposition.** `T` overlapping marker windows of length `≈ ((L − S)/T + S)` cM with `≈ S` cM overlap, where `T` is *the thread count* and `S` the minimum seed length. Duplicate segments from overlapping windows are discarded.
7. **Output.** Buffered, flushed to disk when full, so few segments are resident at once.

**Published numbers, with sources.**

From Zhou, Browning & Browning, *AJHG* 106(4):426–437, doi `10.1016/j.ajhg.2020.02.010`; numbers below grepped from the preprint text at `https://www.biorxiv.org/content/10.1101/2019.12.12.874685v1.full`:

- 485,346 UK Biobank samples, 12 threads: **231.5 billion autosomal IBD segments ≥2 cM in 24.4 hours** wall-clock.
- Same cohort, **chromosome 20 alone: 3.43 billion segments at 2 cM; 106 million at 5 cM**.
- **GERMLINE** could not analyze subsets ≥50,000 (>128 GB RAM). **TRUFFLE** could not analyze subsets ≥150,000. **iLASH** could not analyze ≥150,000 at 2 cM (>128 GB) but completed the full set at 5 cM. **RaPID** could not complete full chr20 at 2 cM within a 2-day cap but completed at 5 cM.
- Head-to-head: single-threaded **RaPID 223.6 min** vs **hap-IBD 13.4 min on 12 threads** for >5 cM on full chr20 (≈17×; the paper separately notes 12-core parallelization gives ≈10× over single-threaded, so the algorithmic gap is ≈1.7×).
- Parent–offspring chr20 coverage, 50,000 samples, ≥2 cM: **hap-IBD 0.987, iLASH 0.978, RaPID 0.994, TRUFFLE 1.0.**

From Tang, Naseri, Wei, Zhang & Zhi, *GigaScience* 2022, doi `10.1093/gigascience/giac111`, PMC9724555 (correction: PMC9798273 — a supplementary file was withdrawn "to remove the benchmarking of TPBWT on the UK Biobank dataset due to potential licensing conflicts"):

Peak memory, UK Biobank chr1, 2 cM cutoff, by sample size (15K/31K/62K/125K/250K/500K):

| tool | 15K | 31K | 62K | 125K | 250K | 500K |
|---|---|---|---|---|---|---|
| hap-IBD | 1.76 MB* | 7.88 GB | 8.79 GB | 22.81 GB | 59.03 GB | **110.15 GB** |
| RaPID | 0.73 GB | 1.03 GB | 1.54 GB | 2.47 GB | 4.14 GB | **7.25 GB** |
| iLASH | 7.85 GB | 20.05 GB | 58.73 GB | 191.46 GB | — | — |
| TPBWT | 7.70 GB | 30.59 GB | 122.16 GB | 488.40 GB | — | — |

*The "1.76 MB" is what the table prints; the unit is almost certainly a typo for GB given the rest of the row. Do not propagate it without re-checking the published table.

- CPU utilization: **hap-IBD 45.27 of 60 cores (≈75%)**; iLASH 9.23 of 60 (≈15%).
- Qualitative accuracy: FastSMC and iLASH high accuracy at 2 cM but low power; hap-IBD slightly lower accuracy at 2 cM but *significantly higher power*; RaPID high power and competitive accuracy on long segments; RaPID and TPBWT lowest length discrepancy. At ≥10 cM with no genotyping error all tools are near-perfect. **Phase switch error at 0.17% costs 4–7% power on short segments and 5–20% on ≥15 cM segments.**

### Secondary incumbents and Apple Silicon status

| Tool | Version / source | Apple Silicon |
|---|---|---|
| **hap-IBD** | JAR, 2023-06-16, `faculty.washington.edu/browning/hap-ibd.jar` | **Yes** — arch-independent bytecode + aarch64 JVM. Not executed here (no JVM installed). |
| **RaPID** v1.7 | `github.com/ZhiGroup/RaPID`, C++, **no tags/releases**, repo pushed 2026-05-07 | **Unverified.** No arm64 statement anywhere. Plain C++; likely compiles. Must be tested. |
| **iLASH** 1.0.2 (2021-01-11) | `github.com/roohy/iLASH`, C++11, CMake ≥3.5 | **Unverified/doubtful.** README lists tested compilers: Apple LLVM 10 on *Intel* MacBook Pro Mid-2015, GCC 4.8.5, GCC 5.0. No arm64 testing claimed. |
| **FastSMC** (in ASMC) | **v1.4.0, 2025-10-22** | **Yes, newly.** PyPI `asmc-asmc` 1.4.0 ships `macosx_14_0_arm64` wheels for cp39–cp314. **v1.3.1 and earlier were x86_64-only** — an arena built a year ago could not have included FastSMC natively on this laptop. Wheel existence verified from the PyPI JSON; not installed or run here. |
| **TPBWT** | `github.com/23andMe/phasedibd`, Cython, pushed 2026-02-23, no releases | **Unverified.** Note the licensing constraint that forced the GigaScience correction — check terms before using it as a published-number baseline. |
| **Refined IBD / TRUFFLE** | Browning lab JAR / C | Included for historical reference only; superseded at these scales. TRUFFLE not independently researched here. |

The arena's frozen baselines are **hap-IBD (accuracy and power reference)** and **RaPID v1.7 (memory and scaling reference)**. RaPID is the memory bar precisely because it is 15× leaner than hap-IBD at 500K — beating hap-IBD on accuracy while using hap-IBD-like memory is a weak result.

---

## Where the real headroom is

**Verdict: hap-IBD is fast but algorithmically crude, and I can support that with evidence from its own authors rather than assertion.** The claim in the problem brief holds up.

**1. Its authors shipped a second tool because its endpoints are wrong.** `ibd-ends` (`github.com/browning-lab/ibd-ends`) exists to take hap-IBD output and re-estimate a *posterior distribution* over each endpoint, accounting for genotype error, recent mutation and gene conversion. hap-IBD reports the IBS boundary as the IBD boundary. The preprint says so directly:

> "There is some degree of error in determination of segment end points. The IBS interval can extend beyond the end-points of a contained IBD segment."

and closes with:

> "Development of IBD segment detection methods that are robust to genotype error, recent mutation, and gene conversion that occur near the ends of IBD segments is an area for future research."

That is the incumbent's authors naming the open problem. Under `F1_c`, endpoint over-extension is charged on both `P_c` and the length-error diagnostic, so this headroom is directly addressable and directly measurable.

**2. Exact-match seeding is a brittle, load-bearing choice.** hap-IBD requires an *exact* IBS seed of ≥1 cM (default `min-extend`). A single uncorrected error inside every candidate 1 cM window destroys the seed and the whole segment is lost — no partial credit, no recovery. Every competitor made a different choice at this exact stage: iLASH uses minhash + LSH banding over haplotype slices, RaPID uses random projections with `-r` runs and `-s` successes, TPBWT uses subsampled position templates. Nobody has systematically searched this design space against an exact evaluator, and the fact that four independent groups picked four different approximate-matching schemes is a strong signal that none of them is known to be right.

**3. Fragmentation under phase error is a documented, unfixed, first-order failure.** The GigaScience benchmark measures **5–20% power loss on ≥15 cM segments** at only 0.17% switch error — and the loss is *worse on long segments*, which is the opposite of what you want, because long segments are the ones downstream applications care about. The TPBWT paper (Freyman *et al.*, *MBE* 38(5):2131) reports that hap-IBD, Refined IBD and iLASH "produced short fragments of IBD that covered only a small portion of chromosome 6 that was truly IBD," while TPBWT "stitched" fragments back across phase switch errors. So the fix is known to be possible and hap-IBD does not implement it. hap-IBD has *no cross-haplotype stitching stage at all* — its `max-gap`/`min-extend` extension operates strictly within one haplotype pair, and a phase switch by construction moves the continuation to a *different* pair. This is a missing pipeline stage, not a mistuned parameter. `F1_c` punishes exactly this (two 4 cM pieces → 32% credit), and `B` (breakage index) reports it directly.

**4. Memory is 15× worse than a contemporary and scales badly.** 110.15 GB vs RaPID's 7.25 GB at 500K samples on chr1. hap-IBD's memory grows ~1.9× per doubling from 62K onward (8.79 → 22.81 → 59.03 → 110.15 GB) while RaPID's grows ~1.7× from a 12× lower base. For a solo researcher this is the difference between a workstation and a large-memory node.

**5. Parallel decomposition is tied to thread count, and it under-uses the machine.** hap-IBD partitions into exactly `T` windows of equal *genetic* length where `T` is the thread count. IBD density is wildly non-uniform across the genome — the paper reports centromeric inflation of "40 to 3000 times greater than the baseline level" — so equal-cM windows carry grossly unequal work and the last thread dominates. The predicted symptom is exactly what the independent benchmark measured: **45.27 of 60 cores, 75% utilization** (iLASH, worse-designed here, gets 15%). Work-stealing over many small tasks, or partitioning by estimated segment density rather than genetic length, is a textbook fix that no IBD tool implements. A ~1.33× wall-clock win is available at zero accuracy cost, which is a real result even though it is not a new asymptotic.

**6. Output representation is completely unexplored.** 231.5 billion segments is terabytes of TSV. Every tool emits flat text. Nobody has published a compressed or implicit representation, despite enormous redundancy (segments cluster on the same pairs and the same loci; endpoints are marker indices with limited entropy). This is an unclaimed axis with no incumbent at all.

**7. Incremental updating does not exist.** Adding 10,000 samples to a 500,000-sample panel requires a full rerun in every tool listed. Biobanks grow monotonically. An index supporting incremental insertion would be a qualitative advance, and `t_incremental` is measured for exactly this reason.

**Where the incumbent is genuinely strong — do not expect wins here.** At ≥10 cM with no genotyping error, all five benchmarked tools are near-ceiling; there is no room. hap-IBD's PBWT seeding is already `O(NM)` for the sweep itself, which is optimal up to constants for the scan. Its buffered output keeps resident segments low. And its 12-thread throughput on real UK Biobank data is genuinely excellent. **The headroom is not in the PBWT sweep — it is in stages 2 (approximate seeding), 4/6 (extension and the missing stitching stage), 6 (work partitioning) and 7 (output), plus the memory profile.** A candidate that reimplements the PBWT sweep faster and claims victory has found a constant factor, and the implementation-independent counters exist to expose that.

**The counter-argument, stated fairly.** hap-IBD won its own benchmark on ROC at 2 cM and won the independent benchmark on power. It is possible that its accuracy is near the information-theoretic limit for a phased panel at this marker density and error rate, and that everything above is a constant-factor and engineering story. **The arena should measure this rather than assume it.** The instrument is an *oracle bound*: run the scoring pipeline on the true genealogy's own IBS realization — i.e. compute the best `F1_c` achievable by any method that only sees the error-injected panel, approximated by an exhaustive `O(N²)` dynamic-programming matcher with access to the true error rates, run at the small tier where quadratic is affordable. The gap between hap-IBD and that oracle is the honest headroom number. **I have not computed it**, and it should be the first thing built after the evaluator, before any search runs. If that gap is under 3 points of `F1_c`, the accuracy axis is closed and the arena should be re-pointed at memory, scaling and output size — which axes 4–7 above suggest are wide open regardless.

---

## Beat target

A win must be falsifiable, multi-objective, and measured on **fresh seeds the candidate has never been evaluated on**. Search runs against a dev seed pool; promotion is scored once against a sealed test pool of 20 seeds per condition.

Let `F1_c*`, `t*`, `R*`, `O*` be hap-IBD's measured `F1_c`, wall time, peak RSS and gzipped output size on the same instances, same thread count, same hardware.

**Win type A — accuracy win.**
`ΔF1_c ≥ +0.02` absolute, aggregated over ≥20 sealed seeds, with the paired bootstrap 95% CI on `ΔF1_c` excluding zero, **and** `t_total ≤ 1.25·t*` **and** `peak_rss ≤ 1.25·R*`. No accuracy win is accepted if it is bought with a large resource regression.

**Win type B — Pareto win.** Strict improvement on ≥1 of `{F1_c, t_total, peak_rss, output_bytes}` with no axis worse by more than 2% relative, CIs excluding zero on the improved axis. This is the realistic first win, and the memory axis (110 GB vs RaPID's 7 GB) is the softest target.

**Win type C — scaling win (the real prize).** Fitted exponent `α` for `t_total` or `peak_rss` vs `N`, over ≥4 tier sizes: `α_candidate ≤ α_hapIBD − 0.15` with non-overlapping bootstrap CIs, at `F1_c ≥ F1_c* − 0.01`. This is the win that changes the curve rather than the point, and it is the one worth a paper.

**Win type D — capability win.** Correctness gate passed while providing something no incumbent provides: a working `t_incremental`, or calibrated endpoint posteriors that beat `ibd-ends` at lower cost. Judged separately; no `F1_c` threshold, but the gate is mandatory.

**Robustness requirement, applying to all types.** The win must hold in ≥4 of the 5 error conditions, and must not be driven by a single length bin. A candidate that wins only in the error-free sequence condition has found a parameter, not an algorithm.

**Anti-overfitting.** Sealed-pool seeds are used once. A candidate promoted on the sealed pool is re-run on a second, freshly generated pool before any claim leaves the lab. Evaluator hash, tool SHA-256s, and all seeds are recorded per run in the JSONL archive.

---

## Mutable surface

The stages below are hap-IBD's actual architecture, read from the preprint. An agent replaces **one** stage, keeping the rest as the reference implementation, so that a win is attributable. Replacing multiple stages is allowed but scored in a separate pool.

| # | Stage | Reference behaviour (hap-IBD) | What an agent may propose |
|---|---|---|---|
| **S1** | **Panel encoding** | bref3-style: rare variants as carrier index lists, common variants as unique allele sequences + mapping vector | Bit-packed transposes, succinct/wavelet-tree indexes, run-length or PBWT-native compression. Directly attacks the 110 GB. |
| **S2** | **Candidate generation / seeding** | PBWT prefix + divergence arrays; **exact**-match IBS seeds ≥ `min-extend` cM (Durbin Alg. 3) | **The richest stage.** Mismatch-tolerant seeding: LSH/minhash (iLASH), random projection (RaPID), position templates (TPBWT), spaced seeds, k-mer sketches, learned or adaptive projections. Success measured by pair selectivity (C1). |
| **S3** | **Seed filtering** | `min-markers=100`, `min-mac=2` | Density-adaptive thresholds; local marker-density models; an MAC/entropy-aware seed score replacing hard cutoffs. Targets the 40–3000× centromeric inflation. |
| **S4** | **Extension / gap bridging** | Greedy bidirectional across gaps ≤ `max-gap=1000` bp to IBS ≥ `min-extend` | Likelihood-ratio or HMM extension; explicit error/mutation/gene-conversion models; posterior stopping rules. |
| **S5** | **Endpoint estimation** | Reported endpoint = IBS boundary (known to over-extend) | Probabilistic endpoints in the spirit of `ibd-ends` but *inline* rather than as a post-hoc pass. Scored by signed length bias and `P_c`. |
| **S6** | **Cross-pair stitching** | **Does not exist in hap-IBD** | A genuinely missing stage. Detect phase switches and rejoin fragments across the homologous haplotype pair (TPBWT-style). Highest expected `F1_c` payoff under switch error; directly moves `Σ p_r²`. |
| **S7** | **Deduplication** | Discard extensions whose extending IBS is itself a valid seed | Canonical-seed selection; interval-tree or union-find dedup. |
| **S8** | **Work partitioning** | `T` overlapping equal-cM windows, `T` = thread count | Density-estimated partitioning, work-stealing over fine-grained tasks, pair-blocked or cache-aware tiling. Targets the measured 75% utilization. |
| **S9** | **Output serialization** | Buffered flat TSV, flushed when full | Columnar/delta/varint encodings, implicit interval representation, compressed pair-graph + decoder. Unclaimed axis. |

**Frozen and off-limits:** the instance generator, the truth extraction (including the same-MRCA-node merge and the `T_cap` audit), `score.py`, the correctness gate, and the resource harness. Agents cannot read the truth files, cannot see sealed-pool seeds, and cannot modify anything under `arena/p02/eval/`. The evaluator hash is recorded with every scored run.

---

## Gameability risks

| # | Attack | Defence |
|---|---|---|
| 1 | **Report every pair over the whole contig.** Maximizes recall. | `P_c` collapses (nearly all reported cM lies outside `T_any`); `F1_c` → ~0. Output-size axis explodes. Wall-time cap. |
| 2 | **Shotgun many tiny segments** to blanket true segments. | `Σ p_r²` is quadratic in piece size: ten 1 cM pieces on a true 10 cM segment earn 10%. G4 blocks sub-threshold reporting outright. |
| 3 | **Nested/overlapping duplicates** to multiply overlap mass. | **G3** requires reported intervals to be pairwise disjoint per pair. Hard fail. |
| 4 | **Over-merge** across gaps to inflate contiguity. | Merged span covering non-IBD sequence is charged against `P_c`. Self-limiting; there is an interior optimum at the truth. |
| 5 | **Memorize a public answer key.** | Truth is generated from fresh seeds at evaluation time and never published. No downloadable artifact to memorize. This is why simulation beats curation here. |
| 6 | **Fit the metric's free parameters.** | `F1_c` has none — no overlap threshold, no merge gap, no matching rule. |
| 7 | **Read the truth files.** | **G6**: truth unmounted, network disabled, instance dir hash-verified pre/post. |
| 8 | **Overfit to a fixed instance set.** | Sealed test pool, fresh seeds, single use; re-validation on a second pool before any claim. |
| 9 | **Falsify the counters.** | **G10** cross-check on a micro-instance with exactly derivable counts; scaling-consistency check. Acknowledged as imperfect — which is precisely why counters are diagnostic and the beat target rests on externally measured time and RSS. |
| 10 | **Self-report resources.** | All resource measurement is external (harness sampling / cgroup), never trusted from the candidate. |
| 11 | **Exploit truth-set artifacts** — e.g. learn tskit's edge-boundary splits and mimic them. | Eliminated at the source by the same-MRCA-node merge, which I verified is exactly equivalent to the node-wise definition (0 mismatches). The artifact does not exist in the truth set. |
| 12 | **Exploit the `T_cap` boundary** by tuning to the exact cap. | The cap is audited to be non-binding (<0.1% of scored truth within 10% of it), and `T_any` uses `2·T_cap`, so the boundary carries no exploitable gradient. |
| 13 | **Ace the planted controls only.** | Controls are pass/fail gates, not scored; they cannot raise a score, only disqualify. |
| 14 | **Nondeterminism as a lottery** — resubmit until a lucky run. | **G5** byte-identical determinism across runs and thread counts. |
| 15 | **Tune to one error condition.** | Win must hold in ≥4 of 5 conditions and across length bins. |

**Residual risk I cannot fully close:** a candidate could implicitly learn the msprime/Gutenkunst generative model — the coalescent's segment-length distribution, the specific demography — and exploit it as a prior rather than discovering a general algorithm. This is legitimate statistical inference and illegitimate benchmark-gaming at the same time, and the metric cannot distinguish them. Partial mitigations: randomize demography across seeds (vary `N_e`, add bottlenecks, sample from CEU/CHB/YRI/admixed), vary the recombination map by chromosome, and — for any candidate that gets promoted — run a held-out check under `stdpopsim`'s HomSap models with a *different* engine (SLiM rather than msprime). I have not built this and it should be treated as an open item, not a solved one.

---

## Evaluation cost per candidate at the small tier

Measured on this machine (Apple Silicon, macOS 26.6) — **and note these are inflated**, because the Python interpreter available here is x86-64 3.8.8 running under Rosetta.

**One-time per seed (amortized over every candidate scored on it):**

| step | time |
|---|---|
| `sim_ancestry` + `sim_mutations` | 3.8 s |
| VCF write (167 MB) + gzip (7 MB) | 9.0 s |
| `ibd_segments(max_time=250)` | 0.33 s |
| same-MRCA-node merge (pure Python; ~0.5 s vectorized) | 9.3 s |
| **total instance + truth** | **22.1 s** |

**Per candidate:**

| step | time |
|---|---|
| candidate run (hap-IBD-class, 1,000 haplotypes, 20 Mb, 8 threads) | est. 2–15 s |
| `score.py` (`F1_c` + secondary metrics) | est. 1–3 s |
| gate checks G1–G10 | < 1 s |
| **total** | **≈ 5–20 s, cap 120 s** |

I did not time hap-IBD itself — no JVM on this machine — so the candidate-run figure is an estimate extrapolated from the published "~10 minutes CPU time for 5,000 UK Biobank samples on chr20," scaled down by sample count and sequence length. Verify it before setting the wall cap.

**Throughput.** With a warm instance cache, roughly **180–700 candidates/hour single-process**, or a few thousand per hour across 8 cores. An overnight unattended run on the laptop scores on the order of 10⁴ candidates — comfortably in FunSearch territory, which is the whole point of insisting the small tier fit on a laptop.

**Storage.** ~15 MB per seed (7 MB gzipped VCF + truth parquets + map). A 200-seed pool is ~3 GB. Trivial.

**Recommended pool.** 20 dev seeds × 5 error conditions = 100 instances ≈ 37 min to build, ~1.5 GB. Sealed test pool the same size, built separately and never opened during search.

---

## Readiness scores

**Evaluator fidelity: 5/5.** Truth is exact by construction from the coalescent — not a proxy, not a consensus. The one serious trap (tskit's edge-wise path splitting, which inflates long-segment counts by 43%) was found and closed, and the fix was *proven* equivalent to the node-wise definition rather than assumed (112 pairs, 0 mismatches). `T_cap` is empirically calibrated and self-auditing. `F1_c` is closed-form and parameter-free. Docked nothing on fidelity; the coalescent-model-realism caveat is a scope limit, not a fidelity defect.

**Evaluator cost: 5/5.** 22.1 s to build an instance plus exact truth; ~5–20 s per candidate; thousands of evaluations per hour on a laptop even under Rosetta. `ibd_segments` itself is 0.33 s — the C implementation does the hard part, and the published benchmark's `O(N²·trees)` Python truth scan is avoided entirely.

**Headroom: 4/5.** Strong documented evidence: the authors published `ibd-ends` because endpoints are wrong and named error-robustness at segment ends as "an area for future research"; 5–20% power loss on ≥15 cM segments under 0.17% switch error, with *no* cross-pair stitching stage in hap-IBD at all; 110 GB vs RaPID's 7.25 GB at 500K; 75% CPU utilization from thread-count-tied windowing; output representation and incremental updating entirely unclaimed. Not 5/5 because at ≥10 cM error-free all tools are already at ceiling, and the oracle bound that would settle whether the *accuracy* axis has room is not yet computed.

**Infra fit: 5/5.** Small tier verified running on this exact laptop: msprime 1.3.1 and tskit 0.5.7 already installed, whole loop measured end to end. hap-IBD is a 270 KB architecture-independent JAR needing only an aarch64 JVM (Homebrew `openjdk` 26.0.2.1, `arm64_tahoe` bottle). FastSMC gained native arm64 wheels in ASMC v1.4.0 (2025-10-22). Medium tier fits a laptop; large tier maps cleanly onto a SLURM cluster. All genetic-map and JAR URLs returned HTTP 200.

---

## Confidence notes

### Verified by execution on this machine

- `tskit.TreeSequence.ibd_segments` signature — `(*, within, between, max_time, min_span, store_pairs, store_segments)` — and full docstring, read from tskit 0.5.7 via `inspect`. The "purely genealogical," "same genealogical path," and "will partition the span of the contig" language is quoted from that docstring, not from a web summary.
- No-threshold behaviour: 31,934,223 segments / 4,950 pairs on 100 haplotypes × 10 Mb; pair (0,1) spans sum to exactly the full contig.
- Edge-wise fragmentation: 532 of 615 adjacent boundaries are not MRCA changes; merging cuts segments 33.3%; ≥2 Mb segments 175 → 250 (+43%).
- **Merge-equivalence proof:** `ibd_segments(max_time) + merge-adjacent-same-node` is byte-identical to a brute-force per-tree `tree.mrca()` scan with a time cap. 112 pairs, 0 mismatches, 114 segments both ways.
- MRCA-time-by-length table (2–3 cM → q95 130.3 g; ≥10 cM → q95 19.6 g) and the `max_time` sensitivity table.
- All timing and cost figures, including the 22.1 s small-tier build.
- Environment: macOS 26.6 arm64; tskit 0.5.7; msprime 1.3.1; **no JVM installed**; **default `python3` is x86-64 3.8.8 under Rosetta**.
- URL liveness (HTTP 200): `hap-ibd.jar` (270,829 bytes, Last-Modified 2023-06-16), `plink.GRCh37.map.zip`, `genetic_map_GRCh37_chr20.txt`, `beagle.27Feb25.75f.jar`.
- Repo metadata via `gh api`: hap-ibd (Java, no releases/tags, pushed 2024-06-11), RaPID (no tags, v1.7 directory, pushed 2026-05-07), iLASH (tag 1.0.2, 2021-01-11), ASMC (v1.4.0, 2025-10-22), phasedibd (no releases, pushed 2026-02-23).
- ASMC arm64 wheels: PyPI JSON confirms `macosx_14_0_arm64` for cp39–cp314 at 1.4.0, and x86_64-only at 1.3 / 1.3.1.
- Homebrew `openjdk` 26.0.2.1 with an `arm64_tahoe` bottle.

### Verified by reading primary source text

- hap-IBD algorithm stages, defaults, and all quoted limitations — grepped from the preprint PDF text (`biorxiv.org/content/10.1101/2019.12.12.874685`), not from a summarizer.
- hap-IBD numbers: 231.5 billion autosomal segments in 24.4 h at 12 threads; 3.43 billion on chr20 at 2 cM; 106 million at 5 cM; RaPID 223.6 min vs hap-IBD 13.4 min; the competitor failure modes; the `T`-overlapping-window parallelization.
- The GigaScience truth generator source (`ZhiGroup/IBD_benchmark/Simulation/msprime_simulation.py`), read in full: confirms node-wise MRCA comparison, 5,000 bp tree subsampling, **no time cap**, 1 cM length filter, Gutenkunst OOA demography, and the exact invocation `python -u ./msprime_simulation.py 20 ./genetic_map_GRCh37_chr20.txt 1.38e-8 56 4000 17197 5000 1.0 ./output/`.
- The GigaScience correction notice (PMC9798273): TPBWT UK Biobank benchmarking withdrawn over licensing.
- iLASH README compile requirements and the Intel-only tested-compiler list.
- RaPID v1.7 usage and output format.

### Lower confidence — extracted via page summarization from a figure-heavy paper

- **GigaScience memory tables.** Reported above as printed. The hap-IBD "1.76 MB" at 15K is almost certainly a unit typo for GB; I flag it rather than silently correcting it. **Re-read the published Table 1/Table 2 before using these as baselines.**
- **GigaScience runtime "hap-IBD ~0.5 hours for UK Biobank chr1 at 2 cM."** I could not reconcile this with the hap-IBD paper's own 24.4 h autosomal figure, and the thread/core counts differ (60 cores vs 12). **Treat as unverified.**
- **CPU utilization 45.27/60 and 9.23/60.** From the text summary, not seen in the table directly.
- **Per-tool, per-bin power and accuracy values.** These live in figures and were not text-extractable. Only the qualitative ordering is reported above.

### Could NOT confirm

- **hap-IBD executing on Apple Silicon.** No JVM on this machine. The bytecode-plus-aarch64-JVM argument is sound and the arm64 JDK exists, but the end-to-end run is untested. Confirm before freezing the baseline.
- **RaPID v1.7 and iLASH 1.0.2 compiling on arm64.** No arm64 statement in either project; iLASH's README lists only Intel Macs as tested. Plain C++11/CMake, so likely fine, but **must be built and tested**.
- **FastSMC actually running from the arm64 wheel.** Wheel filenames verified on PyPI; not installed or executed.
- **TPBWT licensing terms.** The GigaScience correction removed its UK Biobank benchmarking for "potential licensing conflicts." Check terms before using it as a baseline.
- **TRUFFLE.** Named in the brief; I did not research it beyond hap-IBD's report that it failed at ≥150,000 samples. No version, distribution or Apple Silicon claim is made here.
- **hap-IBD's version string.** The repo has no tags; versioning is via the output log's first line. Pin by JAR SHA-256.
- **The oracle bound on `F1_c`.** Not computed. This is the single most important open item: until it exists, the size of the *accuracy* headroom is argued from the incumbent's own stated limitations rather than measured. Build it before running search.
- **hap-IBD's measured `F1_c`, `α`, and small-tier runtime.** All beat targets are expressed relative to these because they do not exist yet — no published number uses this metric. They must be measured as step one.