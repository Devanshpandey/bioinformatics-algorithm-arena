# The Bioinformatics Algorithm Arena — v1 specification
### 2026-09-01
### Active roster: **P01 haplotype long-match · P02 IBD detection · P09 ML tree search**

## The selection criterion

> Do not look for biological prediction tasks. Look for **computational primitives
> where biology supplies enormous structured datasets and the current solution is
> an algorithmic compromise.**

Those are the places where a search process can discover a better ordering, index,
sketch, dynamic program, graph traversal, pruning rule, or approximation scheme —
and, critically, where a machine can *check* whether it did.

Problems are organized by the search operator they invite, not by biological
discipline:

| Class | Task shape | Invites | Problems |
|---|---|---|---|
| **A** | find things efficiently | indexes, orderings, hashes, sketches | P01, P02, P05, P06, P12 |
| **B** | represent things efficiently | succinct structures, compression | P04, P07, P10, P11 |
| **C** | search a combinatorial space | heuristics, pruning, move sets | P08, P09, P13, P14 |
| **D** | approximate an expensive probabilistic computation | state reduction, sparsification | P03, P15 |

---

## Part I — The three findings that reshape the design

Three independent audits (scoring design, adversarial contamination, feasibility)
produced three conclusions that matter more than the problem list itself.

### Finding 1 — The problems are three threat surfaces, not one

Conflating them is the design error that sinks arenas.

| Class | Truth provenance | Contamination risk | Dominant threat | Problems |
|---|---|---|---|---|
| **Self-verifying** | truth is *computed* by an oracle from the input | **~zero** — there is no answer key, only a function | degenerate output; resource-accounting fraud | P01, P04, P06, P07, P09, P10, P11, and P08 under an objective score |
| **Simulation-derived** | generated from a seed | **zero if regenerated, total if downloaded** | fitting the simulator's model | P02, P03(sim), P05, P12, P13, P15(sim) |
| **Curated human truth** | consortium curation over real samples | **real and unfixable by regeneration** | memorization, retrieval, circularity | P14 (GIAB), P03(real), P08 (BAliBASE), P15 (trio) |

Only 3–4 of 15 problems have genuine truth-set contamination exposure. That is the
good news. The bad news, which is badly under-appreciated: **published tool source
code is a guaranteed contamination channel for all fifteen.** minimap2, KMC, BCALM2,
RAxML-NG, Beagle, SHAPEIT5, hap-ibd, GBWT, PBWT, Syllable-PBWT, μ-PBWT — all on
GitHub, all near-certainly in pretraining. The arena's *truth* claim is threatened in
a few places; its **novelty claim is threatened everywhere.** Novelty verification is
therefore a scored gate, not an afterthought. No sandbox fixes this — it is in the weights.

### Finding 2 — Wall-clock scoring cannot tell an algorithm from an implementation

This is the crux of the whole enterprise, and we demonstrated it on our own P01
harness before writing a line of this spec. Three submissions, all producing
**byte-identical output** (9,600 matches, exact against the oracle):

| submission | build | query/q | index B/cell | what it actually is |
|---|---|---|---|---|
| `baseline_pbwt` | 53.0 s | 0.261 s | 8.00 | Durbin's PBWT, readable Python |
| `sub_vectorized_pbwt` | **0.67 s** | 0.264 s | 8.00 | **same algorithm**, numpy column update |
| `sub_streaming_pbwt` | 0.00001 s | 0.200 s | **0.00** | **different algorithm**: no stored index |

Ranked on build wall-clock, the vectorized submission is a **79× win** — and it
contains *zero* algorithmic novelty: identical index, identical query path,
identical asymptotics. The streaming submission's win is structural (it deletes the
73 MB index entirely by regenerating the PBWT arrays during a batched sweep) yet
looks far less impressive on the same axis.

**A scoring rule that cannot separate those two columns is not measuring science.**
Hence the dual-leaderboard design in Part II.

### Finding 3 — Feasibility eliminates a third of the list before science does

The binding constraints for a solo researcher are laptop RAM, macOS/arm64 build
availability, and data-access regimes — not scientific interest.

- **`vg` has no macOS conda build and no macOS release binary**; its indexes are
  26–38 GB. P04 and P05 are deferred on platform grounds alone, which is genuinely
  painful because `vg sim` offers one of the cleanest exact-truth signals in the set.
- **metaSPAdes needs 100–500 GB RAM per sample.** P13 cannot host a fast inner loop
  on any laptop; every candidate becomes a batch job.
- **The panels that define imputation SOTA are all unusable as arena assets:** HRC is
  EGA-controlled, TOPMed offers no panel download, **UKB cannot be redistributed.**
  P03 is deferred on access *and* saturation grounds.
- **UKB constraint, general:** any arena task whose reproducibility depends on UKB is
  not reproducible by anyone else, which defeats the purpose. Design every UKB-touching
  problem so the *public* tier (1000G + msprime) is the scored one and UKB is a
  private confirmatory run only.
- Conversely, **eight problems (P01, P02, P06, P07, P08, P09, P12, P15) admit exact
  planted or exactly-computable truth with no download whatsoever.** Every
  recommended starting problem is drawn from those eight. That correlation is the
  selection criterion, not a coincidence.

---

## Part II — Scoring

### II.1 Admissibility gate, then one objective per track

Multi-objective scoring makes "beat the incumbent" ill-defined. Resolve it with an
ε-constraint gate followed by single-objective ranking. **Strict Pareto dominance is
used for the archive, never for the leaderboard** — it almost never fires, it is
fragile to noise, and it yields a set rather than a ranking, so it cannot drive
evolutionary selection.

**Gate (hard, pre-declared).** A candidate is admissible only if all four hold:
1. **Validity** — passes a *semantic* validator, not a format check. Failure scores
   zero; there is no partial credit for a fast wrong answer.
2. **Accuracy non-inferiority** — accuracy ≥ incumbent − δ, δ pre-declared per task.
3. **Memory cap** — peak RSS ≤ incumbent (κ = 1.0 default).
4. **Index cap** — on-disk index bytes ≤ incumbent's, where an index exists.

**Three tracks, run per task:** **T** (minimize cost at fixed accuracy and memory —
the default), **M** (minimize peak RSS at fixed accuracy and time), **A** (maximize
accuracy at fixed time and memory). Track A is where a genuinely more sensitive
algorithm lands, and it is the track most arenas forget to build.

**The "within noise" test.** Accuracy differences are a *sampling* problem, not a
seed problem: for deterministic tools on fixed data, run-to-run variance is zero and
the uncertainty is over the finite sample of evaluation units.
- Define the evaluation unit per task (read; (sample, marker) genotype; alignment
  column; replicate search).
- **Paired block bootstrap, 10,000 resamples**, blocking by contig / LD block / gene
  family — never resampling units independently.
- Pass iff the **lower** bound of the 95% CI on Δ exceeds −δ. The burden of proof
  sits on the challenger, deliberately.
- **Sizing rule:** the sealed test must be large enough that δ ≥ 3 × the bootstrap SE
  of Δ. If it fails, enlarge the test set — never loosen δ.
- Stochastic algorithms get s = 20 seeds and **must report the worst seed**. An
  algorithm that is fast on average and catastrophic 1-in-20 is not deployable.
- Multiplicity: **no family-wise correction on non-inferiority gates** (it would make
  the gate easier to pass, in the wrong direction); Benjamini–Hochberg at q = 0.05
  across superiority claims only.

### II.2 Dual leaderboards — the answer to Finding 2

- **Leaderboard A — Algorithmic.** Mandated scaffold, one language for all entrants,
  immutable I/O layer, mutable region delimited by `EVOLVE-BLOCK` sentinels. Cost = a
  designated **implementation-independent counter**, incremented *inside the immutable
  API layer* so a candidate cannot under-report. **This is the scientific claim.**
- **Leaderboard B — Engineering.** Any language, SIMD, GPU. Cost = wall clock and peak
  RSS against the real released incumbent binary on the same node. **This is the
  usability claim.**

The headline result is the *pair*. A-win + B-loss = "promising algorithm,
unengineered" — a real, publishable result. B-win + no A-movement = "engineering win,
no algorithmic advance" — also published, also labeled. This names the confound
instead of burying it. (MLPerf solves the same problem with Closed vs Open divisions.)

**Primary counters per task:** DP cells entered (mapping, MSA); **Li–Stephens HMM
state-transitions evaluated** (imputation/phasing — this is exactly the quantity a
PBWT state reduction attacks); bytes spilled to and re-read from temp storage (k-mer
counting); conditional-likelihood-vector updates (phylogenetics); index probes and
bits per element (P01/P11). Plus a universal set: retired instructions (`perf stat
-e instructions:u`, or cachegrind when determinism matters), LLC misses, I/O volume,
bits/element, cgroup v2 `memory.peak`.

**Two guards this creates a need for:**
- **Counter-avoidance detection.** "Fewer DP cells" is trivially achieved by not
  aligning; counters are meaningful only behind the accuracy gate. Flag for human
  review any candidate whose primary counter drops >10× while accuracy is flat —
  that is either a breakthrough or a leak, and you want to know which. Published
  harness work found **stronger models hack more** (16.6% hack rate for the strongest
  model vs 0% for the mini model), so budget for this review pass.
- **Engineering headroom constant.** Implement the incumbent's algorithm both naively
  and optimally in the scaffold; publish the ratio. If a Leaderboard-B win is smaller
  than that constant, you cannot attribute it to the algorithm, and you must say so.

**Never use language handicap factors** ("divide Python by 30"). Fix language by
mandating the scaffold. Also mandatory: **scaffold fidelity.** If the in-scaffold
reimplementation of the incumbent is 5× slower than the real tool, every "win" is an
artifact of your own bad reference. Require the scaffold reference within **1.5×** of
the real incumbent before a task goes live, and publish that ratio in every table.

### II.3 Measurement normalization

A runtime number without its machine fingerprint is not comparable, and the harness
must treat it as unreported. Pinned cores on one NUMA node, SMT off, turbo off,
governor `performance` with drift asserted <2%, ASLR **left on** with setup
randomization instead (measurement bias from link order and environment size is large
and ubiquitous), thread count set by the harness and verified from cgroup cpu/wall
ratio, cold-cache headline for I/O-bound tasks and warm for compute-bound.

**Interleaved same-node A/B is mandatory** — run `I C I C I C I C I C`, never
`IIIII CCCCC`, with randomized starting arm. This is the cheapest and highest-value
protection against thermal drift, and the rule most often skipped.

r = 5 repeats (escalate to 11 if IQR > 3% of median); report min, median and raw
times; decide on the **median** with a bootstrap CI on the ratio. **Win threshold:
the entire 95% CI on the speedup ratio must lie above 1.0 and the point estimate
must be ≥ 1.05.** Sub-5% differences are not claims. Peak RAM from cgroup v2
`memory.peak` (not `/usr/bin/time -v`, which mishandles child processes and mmap),
r = 3, take the **max** — worst case is the deployable number.

### II.4 Scaling-curve scoring

Three tiers is three points, and a power law fit to three points is numerology.

Use a **dedicated size ladder: ≥7 log-spaced sizes over ≥2.5 decades.** If the top
end is unaffordable, **extend the bottom, not the top** — you need the lever arm and
small points are nearly free. Fit the deterministic counter first, wall clock second.

Three mandatory diagnostics published with every exponent:
1. **Curvature test** — also fit a quadratic in log n; if the CI on the quadratic term
   excludes zero, a single exponent is a lie. This is the most important guard,
   because most real bioinformatics algorithms have a regime change (in-cache →
   out-of-cache, in-RAM → external-memory) that masquerades as a better exponent when
   you sample only one side of it.
2. **Leave-one-size-out stability.**
3. **Alternative-model comparison** against n log n and n log² n; if ΔAIC < 4, report
   **"no reliable exponent"** rather than a number.

Score by **extrapolated cost ratio one decade past the largest measured size, with
propagated CI, explicitly labeled a prediction** — then once per round actually run
that point for the top candidate and publish predicted-vs-actual. That table is how
the arena earns the right to talk about asymptotics. Award an "asymptotic
improvement" badge only when the CI on Δb excludes 0, curvature passes for both arms,
**and the fitted crossover n lies within or below the measured range.** An algorithm
that only wins beyond every size you tested is a conjecture, not a result. Always
report the crossover in plain language: "faster above ~2M reads."

---

## Part III — The harness contract (anti-gaming)

Ranked by (probability × damage × difficulty-to-detect):

| # | Exploit | Guard |
|---|---|---|
| 1 | **Truth in the input.** Simulators (ART, pbsim3, Badread, wgsim, CAMISIM) write the true origin into the read name: `@sim_chr7_11823991_+`. An agent parsing it scores perfectly and is nearly invisible in a diff. | **Rename every record to an opaque ID before the candidate sees it**; keep the mapping outside the container. ~10 lines. Highest value per line of code in the whole arena. |
| 2 | Correctness-for-speed trades the metric doesn't price (silently drop 3% of hits, get 4× faster) | Exact-output gate where an oracle exists; recall floor at *every* threshold, not just the headline one |
| 3 | **Reimplementation as invention** — the Erdős failure mode transposed. Text-similarity tools do not catch it. | Mandatory novelty gate: mechanism statement → literature search → an independent agent argues the candidate *is* a known method |
| 4 | Resource-accounting fraud: background threads, lazily deferred output, mmap-hidden RAM, warm page cache | cgroup accounting of the whole process tree; cpu/wall ratio check; forced cold cache; output flushed and `sync`ed inside the timed region |
| 5 | Compression self-reference (decompressor embeds the data or reads an uncounted reference) | Total accounting: decompressor + all resources it reads count toward the size |
| 6 | **Network retrieval of a published answer** (a good HG002 SV call set is a `wget` away — far likelier than memorization) | **No network egress from the evaluation container**, denied at the namespace level |
| 7 | Metric-definition exploits (switch error over phased sites only; r² pooled across MAF) | Pre-declared stratified metrics: MAF-binned r², SER with a completeness denominator |
| 8 | Seed shopping, best-of-N, tier cherry-picking, leaderboard probing | Pre-registration ledger; Ladder-thresholded feedback; fixed tier reporting |

**The structural fix that makes reading the answers impossible rather than merely
forbidden:** ground truth never enters the candidate's container. The candidate
writes output; a separate scoring pass in a different container compares it to truth.
Layered with: evaluator lives outside the agent's repo; no egress from the eval
container; non-root, read-only rootfs, seccomp, no ptrace, quota'd tmpfs scratch;
SHA-256 verification of every immutable scaffold file at eval time, with expected
hashes stored outside the container.

**Governance.** DEV (open, ~30% of budget) / VAL (labels withheld, rate-limited) /
TEST (sealed, **different generative source** — a different sequencing run, species,
or simulator parameterization; a different random subset of the same source tests
nothing). **N = 5 sealed evaluations per lineage per task per round, no rollover, a
run consumes budget even if the candidate fails to compile.** Sealed feedback is
**Ladder-thresholded**: return the improved score only if it beats the lineage's
previous best by more than η, else return the previous best — with η = δ, so the
leaderboard structurally cannot resolve differences already declared meaningless.
Return only the primary scalar plus per-gate pass/fail; rich diagnostics live on DEV.

**Pre-registration** before every sealed run: candidate SHA, container digest,
declared track, declared δ, **expected VAL result**, and a one-paragraph mechanism
claim. Hash-chained append-only JSONL, published whether or not the run succeeds.
The point is that **the number of attempts becomes publicly countable** — that is
what makes the sealed-test statistics honest, and it is what almost every ML
leaderboard fails to do.

---

## Part IV — Focused roster: three problems

The full 15-problem feasibility survey is not part of this release; the three problems below are the ones it selected.
The active arena is three problems, chosen for attackability rather than appeal.

| | Problem | Class | Why this one |
|---|---|---|---|
| **P01** | Haplotype long-match query | A+B | Exact oracle. Highest downstream leverage in the whole roster: a better haplotype index propagates into IBD, imputation and panel compression. Directly transferable to biobank-scale haplotype analysis. |
| **P02** | IBD segment detection | A | Exact genealogical truth from tskit. **Built on P01's PBWT**, so the leverage claim becomes a measured dependency. `hap-ibd` is a noarch JAR — the incumbent runs natively with zero compilation. |
| **P09** | ML phylogenetic tree search | C | The cleanest evaluator: an exactly computable objective, no answer key, and — uniquely — a **provable global optimum** on small instances. Zero data access; six incumbents install natively in one command. |

**Why these three and not others.** P01 is required and already built. P02 shares
P01's infrastructure and its truth is exact, so the marginal cost of adding it is
small and it tests the transfer hypothesis directly. P09 is the purest
combinatorial-search problem available with a math-grade scorer, and it is the
one where an evolutionary loop has the least excuse to fail.

Deferred with reasons (from the feasibility audit): **P04/P05 pangenome** — no
macOS `vg` build, 26–38 GB indexes; **P13 metagenome assembly** — 100–500 GB RAM
per sample, and CAMI III assembly closed 2026-10-14; **P03 imputation** — every
SOTA panel is access-restricted and UKB cannot be redistributed, and the
incumbents are near-ceiling on 1000G; **P10 compression** — canonical benchmark
site down; **P14 SV calling** — usable only with the regenerate-and-blend recipe
(v5.0q minus a private 15% hold-out, plus a secret-coordinate simulation arm and
a null arm), and it carries disclosed circularity.

**Tier-0 calibration still applies.** Before scoring anything, run the
incumbent-against-itself check and confirm the harness reports speedup 1.00 with
a CI straddling 1. On this roster P01 is the natural calibration target, since
its oracle makes correctness unambiguous.

## Part V — What is built, and what it already caught

All three problems have a runnable generator, an exact truth source, a verified
incumbent, and a measurement harness. None of them is scoreable yet, and the
reasons are the point of the exercise.

| | truth | incumbent verified | measured | blocking issue |
|---|---|---|---|---|
| P01 | exact brute-force oracle (+ precondition assert) | 16 synthetic + 2 coalescent gate instances, all 3 submissions exact at 9,552 matches | build 56.2 s → 0.66 s vectorized; index 8.0 B/cell; streaming variant 0.0 B/cell | none — Tier-0 calibration **passed** (ratio 1.058, CI [0.976, 1.110]); target now precisely named (Finding 6) |
| P02 | exact (tskit genealogical) | Durbin Alg 3 = brute force, 40/40 | recall 0.332, precision 0.791, 64.3 s; stage 1 = **99.98%** of runtime | genealogical ≠ detectable IBD; score relative only |
| P09 | exact (computed objective) | reaches exhaustive global optimum | RF=0 at 16×1000; 54,900 CLV updates | **DEMOTED — headroom refuted by raxml-ng/IQ-TREE (Finding 7)** |

The harness has already earned its keep five times, each time catching something
a wall-clock leaderboard would have rewarded:

1. **The implementation-vs-algorithm confound, demonstrated** (Finding 2): a
   submission with *zero* algorithmic novelty posted a 79× build-time win.
2. **A real correctness bug**: P01's gate caught a `np.maximum.reduceat` final
   segment running past the last selected index, producing plausible but subtly
   wrong divergence values.
3. **Cross-problem module shadowing**: P02 silently loaded *P01's* generator
   because a shared `sys.path` let one problem's `generate.py` shadow another's.
   Fixed by explicit-path loading. A benchmark suite that lets problems
   contaminate each other's inputs is not a benchmark suite.
4. **A silently wrong index on the scored tier** (Finding 5): multi-allelic
   sites from msprime's default mutation model corrupted PBWT divergence arrays,
   producing 5 spurious and 1 missing match per 3,245. The gate could not see it
   because gate panels were synthetic and binary. Now fixed three ways, and the
   gate draws from the scored distribution. *(P02's generator carried the same
   latent defect but happened to produce no multi-allelic sites at its
   parameters — its numbers are unchanged; the fix makes it structurally
   impossible rather than luckily absent.)*
5. **A saturated tier, caught before it was scored**: P09 at 16 taxa × 1000 sites
   recovers the true tree exactly, so it has no headroom. The difficulty probe
   found the one regime that does — and then found that the measured headroom is
   against our own weak baseline, not the real incumbent. That is the
   scaffold-fidelity rule from §II.2 biting on day one, exactly as designed.

### Finding 5 — the gate passed while the scored tier was broken

The most important thing the arena caught, and it caught it only because an
independent audit re-derived the problem's preconditions from scratch.

`msprime`'s default mutation model produces **multi-allelic sites** — 17 of 9,330
sites (0.18%) in P01's scored small tier carried allele value 2. A PBWT column
update splits on `allele == 0` versus everything else, which **merges alleles 1
and 2**, corrupting the divergence arrays. Measured consequence on a coalescent
instance: **5 spurious and 1 missing match out of 3,245** — a silently wrong
index returning plausible answers.

The gate did not catch it because **the gate's instance distribution did not
cover the scored instance distribution**: gate panels were synthetic and binary
by construction, so they could not express the failure. Every P01 number
recorded before this fix was a measurement of a program that was not exact.

Three fixes, all now in place:
1. Both generators pin `msprime.BinaryMutationModel()`, which makes panels
   strictly biallelic (verified: allele values `[0, 1]`, zero multi-allelic sites).
2. `oracle.py` **asserts the biallelic precondition** and raises. A precondition
   that scored instances can violate must be *checked* by the evaluator, never
   assumed by it — a silently violated precondition is worse than a crash.
3. The correctness gate now runs a second half (**gate 1b**) on instances drawn
   from the *scored* generator, not just synthetic panels.

**The general rule this establishes, which applies to every problem in the arena:**
a gate whose instance distribution does not cover the scored distribution
certifies nothing. Post-fix, all three P01 submissions pass 16 synthetic +
2 coalescent gate instances and agree exactly (9,552 matches).

### Finding 6 — P01's headroom is localized, and the theory is unimplemented

The independent spec audit measured the run structure of our own instances and
narrowed the target from "the index is too big" to a single data structure:

| tier/seed | N × M | allele-column runs `r` | runs/column | divergence-array runs `r_d`/cell |
|---|---|---:|---:|---:|
| small/42 | 980 × 9,330 | 50,712 | 5.4 | **0.797** |
| small/7 | 980 × 8,797 | 48,372 | 5.5 | **0.813** |

(For calibration, μ-PBWT reports ~11 runs/column on 1000 Genomes at N=5,008 — our
5.4 at N=980 is the same compressibility regime, so the simulated instances have
realistic structure.)

An `O(r)`-word index at 2 words/run costs **0.71 bits/cell — 90× smaller than our
baseline and below the bit-packed panel itself.** But the divergence array is
**essentially incompressible by run-length encoding at 0.80 runs/cell.** All of
the space is in `d`, none in `a`. That is the precise target:

> **Represent the divergence information in `o(NM)` space with `O(1)` access,
> for all-long-match reporting.**

This is exactly what the field has been circling: Syllable-PBWT wins by *not
storing `d` at all*; μ-PBWT stores Φ/threshold samples instead (Φ alone is ~40%
of its memory); the WABI 2025 long-LEM paper's whole contribution is `O(1)`
`PLCP[i]` in `O(r)` space.

**Two results prove the target is reachable and neither has been built:**
- Sanaullah, Zhi & Zhang, WABI 2025 (`10.4230/LIPIcs.WABI.2025.17`,
  arXiv:2505.15698): `O(m + occ)` expected time in `O(r)` words for *exactly this
  problem*, via a Nishimoto–Tabei move structure. The paper's own future work
  asks for "a practical implementation... possibly as a modification of MOVI or
  b-move." **No implementation exists.**
- Bonizzoni, Cozzi & Gao, arXiv:2602.13461 (2026): `O(r̃)`-word structure with
  **constant-time** forward and backward RLPBWT stepping. Explicitly "contains no
  experimental validation."

**And the published SOTA does not intersect.** Syllable-PBWT reaches 1.0–2.0
bits/cell but is static and hash-based; μ-PBWT reaches 0.021 bits/cell on WGS but
answers only SMEMs and is up to 6× slower than Durbin; Dynamic μ-PBWT answers the
right query at 12–15 bits/cell; PBML answers KL-SMEMs. **No implementation is
simultaneously (a) exact for all long maximal matches, (b) ≤1 bit/cell, and
(c) not slower than uncompressed PBWT.** That intersection is empty as of
2026-09-01, and filling it is a publishable result.

**Critical disambiguation before any novelty claim:** "all long maximal matches"
(this problem), "set-maximal/SMEM", and "k-SMEM/KL-SMEM" are *different problems*.
Roughly half the apparent PBWT SOTA is not comparable to P01 because it solves a
different one. Any claim must state which.

**Known bounds, honestly.** There is no published lower bound specific to
long-match query on haplotype panels. Kempa & Kociumaka (arXiv:2510.19820, 2025)
prove a dichotomy in `O(δ polylog n)` space for binary alphabets: LCE/SA/LCP-class
queries need `Θ(log n / log log n)`, while BWT/LF/PLCP/Φ need `Θ(log log n)`.
Divergence access is the positional analogue of LCP — the *harder* class — so
`O(1)` divergence access is provably out of reach at `δ`-bounded space, while at
`O(r)` words it is attainable. **The barrier and the achievability result bracket
the design space, and the search should live in the gap.** Caveat flagged rather
than hidden: those are theorems about standard strings and the ordinary BWT; the
PBWT is a different, column-wise transform and no paper formally transfers them.
Treat it as a strong analogy, not a citation.

**Engineering, not science:** our baseline is ~100× off compiled implementations
(Dynamic μ-PBWT answers a long-match query on UKB chr21 at N=100,000 in 0.2325 s;
ours needs 0.247 s at N=980). Since both are supposed to be `N`-independent per
query, that gap is interpreter and constant-factor overhead. It belongs on
Leaderboard B and must never masquerade as a discovery.

### Finding 11 — P09 is UN-demoted: the incumbents diverge sharply at scale

A run on the HPC node settled the question Finding 7 left open. raxml-ng 2.0.2 vs
IQ-TREE 3.1.3, 5 seeds per tier, every tree re-scored by our own Felsenstein:

| tier | taxa | median lnL gap | max gap | median RF(raxml, iqtree) |
|---|---:|---:|---:|---:|
| hard32 | 32 | 0.002 | 0.307 | 2 |
| hard48 | 48 | 0.238 | 0.767 | 2 |
| **hard128** | **128** | **34.93** | **51.00** | **106** |
| **hard256** | **256** | **231.42** | **322.04** | **306** |

Below ~50 taxa the two tools are indistinguishable — which is why Finding 7,
measured only at 16–48 taxa, concluded "saturated." At 128 taxa they separate by
~35 log-units and disagree on 106 splits; at 256 taxa, by ~231 log-units and 306
splits. Two independently-implemented state-of-the-art searches disagreeing that
violently means **at least one of them is far from the optimum**, and the gap
between them is a *lower bound* on the distance from optimal.

By the criterion pre-registered in Finding 7 — "if they diverge, the gap is
genuine tool-independent headroom and P09 survives at that scale only" — **P09
is reinstated, with its scored regime bounded to n_taxa ≥ 128.** The demotion was
correct on the evidence available; the evidence changed.

**One confound, and it is serious enough to gate the claim.** IQ-TREE was often
the *faster* tool (46 s vs 112 s at hard128/seed42) while also scoring worse.
That is exactly the signature of a tool given less work by its defaults, not of a
weaker algorithm. A gap that is really a flag mismatch is not headroom. A
matched-budget sweep varies each tool's search effort and compares the
two lnL-vs-CPU curves at **matched CPU**; the claim stands or falls there.

### Finding 12 — at scale the dense index stays dominated, and a prediction of ours was wrong

HPC node, 250 GB RAM:

| N | M | index | bits/cell | stored q/q | streaming q/q | matches |
|---:|---:|---:|---:|---:|---:|---:|
| 992 | 9,329 | 74 MB | 64.0 | 0.142 s | 0.149 s | 10,211 |
| 3,992 | 10,694 | 342 MB | 64.0 | 0.514 s | 0.512 s | 51,260 |
| 15,992 | 12,405 | 1,587 MB | 64.0 | 2.559 s | 2.475 s | 243,994 |
| 31,992 | 12,914 | 3,306 MB | 64.0 | 5.793 s | 5.666 s | 542,721 |

Fitted exponents: stored-index b = **+1.075**, streaming b = **+1.053** — both
essentially linear in N, consistent with output growing as N^1.31.

At the scored tiers both submissions are exact and agree exactly (n10k:
9,984 × 114,415, 1,373,136 matches; n30k: 5,203,780 matches). And the
zero-byte streaming submission still **wins at every scale**: 10.82 vs 12.42 s/q
at n10k, 39.98 vs 44.89 s/q at n30k. Finding 10's conclusion does not merely
survive the move to scale — it strengthens: **the dense PBWT index is dominated
by having no index at all everywhere we have tested, up to N = 32,000.**

**A prediction we got wrong, recorded rather than quietly dropped.** The job
script asserted the dense index would be ~8 TB and fail outright at large N. It
did not fail; at n30k it was ~29 GB and ran fine. The error was assuming M stays
~10,000 while N grows. It does not: more haplotypes means more segregating
sites, so M grew from 9,329 at N=992 to 114,415 at N=9,984. The dense index is
therefore Θ(N·M) with *both* factors growing — worse asymptotically than we said,
but far smaller at the specific sizes tested. The lesson is the arena's own: a
back-of-envelope figure that was never measured does not belong in a spec.

**Still missing:** `sub_rlpbwt_streaming` was written after that run was
submitted, so the only submission that actually attacks the space target has no
scored-scale numbers. A later scored-tier run supplies them (below).

### Finding 15 — the gain transfers, but the small-tier leaderboard MIS-RANKS

A follow-up run scored all six candidates plus the incumbent at all three tiers.

| module | small | n10k | n30k | gain@small | gain@n30k | rank@small | rank@n30k |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sub_column_diff` | 0.02592 | 0.00375 | **0.00158** | 9.68× | **7.07×** | 2 | **1** |
| `sub_bitvector_rank` | 0.02737 | 0.00414 | 0.00174 | 9.17× | 6.41× | 3 | 2 |
| `sub_freeform` | 0.02567 | 0.00395 | 0.00190 | **9.78×** | 5.87× | **1** | 3 |
| `sub_delta_varint` | 0.03529 | 0.00563 | 0.00234 | 7.11× | 4.76× | 4 | 4 |
| `sub_bitpack_fixed` | 0.03595 | 0.00569 | 0.00234 | 6.98× | 4.76× | 5 | 5 |
| `sub_eliasfano` | 0.03605 | 0.00569 | 0.00238 | 6.96× | 4.68× | 6 | 6 |
| *(incumbent)* | 0.25097 | 0.02928 | 0.01115 | — | — | — | — |

**Two results, and the second is the important one.**

**(a) The gain transfers, attenuated.** Every candidate still beats the incumbent
at the scored tier; the advantage shrinks from ~9.8× to ~5.9–7.1×, exactly as
`sub_freeform` predicted from its own row-subset curve (19.5× at N=61 → 9.8× at
N=980). The mechanism is understood: the incumbent pays a flat 32 bits/run while
a modelled coder pays ≈ log₂(N/r) + c, and the ratio between those closes
logarithmically in N. Best absolute result: **0.00158 bits/cell — 633 panel
cells per bit, 633× below the bit-packed panel**, exact.

**(b) The small-tier ranking is WRONG.** `sub_freeform` wins at the small tier
and is **third** at n30k; `sub_column_diff` is second at small and **first** at
n30k. The podium reorders. Had we scored only where it was cheap to score — which
is what every laptop-scale benchmark does — we would have crowned the wrong
method and, worse, drawn the wrong conclusion about *which idea* matters.

This is Finding 10 escalating from a magnitude error to a **ranking** error, and
it is the strongest evidence yet for the §II.4 rule: a point measurement at one
size is not a result. It also vindicates the decision to move P01's scored tier
to N ≥ 10,000 rather than reporting the laptop numbers.

### Finding 16 — P09's headroom survives matched CPU; the reinstatement is confirmed

A matched-budget run swept both tools' search effort and compared log-likelihood at
**equal CPU seconds**, which was the blocking confound on Finding 11.

| tier | seed | matched CPU | raxml-ng lnL | IQ-TREE lnL | gap |
|---|---:|---:|---:|---:|---:|
| hard128 | 42 | 104.3 s | −76110.301 | **−76105.444** | 4.86 |
| hard128 | 43 | 100.2 s | **−78731.531** | −78765.869 | 34.34 |
| hard128 | 44 | 86.5 s | **−74929.239** | −74935.344 | 6.11 |
| hard256 | 42 | 503.7 s | **−154102.857** | −154350.066 | 247.21 |
| hard256 | 43 | 234.9 s | **−152332.909** | −152432.520 | 99.61 |
| hard256 | 44 | 314.0 s | **−148869.290** | −149031.446 | 162.16 |

| tier | unmatched median gap | matched median gap |
|---|---:|---:|
| hard128 | 34.93 | 6.11 |
| hard256 | 231.42 | 162.16 |

**The gap is not a budget artifact.** Equalising CPU removes most of it at 128
taxa (34.9 → 6.1) and little of it at 256 (231 → 162), but it never closes. And
the **winner alternates by seed** — IQ-TREE takes hard128/seed42, raxml-ng takes
the other five. Two SOTA searches landing in *different* local optima, with
neither dominating, is the signature of genuine search difficulty rather than one
tool simply being tuned harder.

**P09 is confirmed reinstated, scored regime n_taxa ≥ 128.** Below ~50 taxa it
remains saturated and must not be scored. The full arc is worth keeping as a
methodological record: measured headroom against our own baseline (11.74) →
refuted by real incumbents → reinstated by scaling to 128+ taxa → confirmed by
equalising budget. Three of those four steps overturned the previous one.

### Finding 13 — bits/cell is strongly scale-dependent, which reframes Result 2

The scored-tier run evaluated `sub_rlpbwt_streaming` at the scored tiers. The index does not
merely survive the move to scale — it gets **far better**:

| tier | N | M | bits/cell | runs/column | runs/col ÷ N |
|---|---:|---:|---:|---:|---:|
| small | 980 | 9,329 | 0.2510 | 5.44 | 0.00555 |
| n10k | 9,984 | 114,415 | **0.0293** | 6.88 | 0.00069 |
| n30k | 29,984 | 129,048 | **0.0112** | 8.20 | 0.00027 |

The mechanism is clean and is the coalescent doing the work: **runs per column
barely grow with N** (5.44 → 8.20 while N grows 30×), because the number of runs
in a PBWT column is governed by the number of distinct ancestral haplotypes at
that locus, not by how many samples you drew. So r is close to N-independent,
bits/cell ≈ 32r/(N·M) falls roughly as **1/N**, and a 30× larger panel yields a
~22× smaller per-cell index.

**Why this reframes Result 2.** The six search candidates were all optimised at
the small tier, where the incumbent is 0.2510 and the best candidate reached
0.02566. But at n30k the **incumbent alone is already 0.0112 — better than the
best small-tier candidate.** The 9.78× is therefore a *small-tier* result, and
whether it transfers is an open empirical question: the candidates' gains come
from encoding gap distributions, and those distributions change with N. A
scored-tier run measures the transfer rather than assuming it (Finding 15).

This is the scaling-curve warning of §II.4 landing on our own headline number.
A point improvement measured at one size is not a result until its curve is
measured — and here the *baseline's* curve is steep enough to swallow the gain.

### Finding 14 — our own baseline was committing the accounting cheat we audit for

`sub_streaming_pbwt.nbytes()` returned **0**, with the comment "stores nothing
beyond the panel it was handed." But the object holds `self.panel` alive and
answers every query from it. Those bytes are part of the index whatever we call
them — this is precisely the dishonest-`.nbytes()` cheat the verification agents
are instructed to hunt, sitting in the arena's own baseline.

It mattered: reporting 0 made the streaming submission look free and understated
every comparison against it by 8 bits/cell (the uint8 panel). Fixed — `nbytes()`
now returns `self.panel.nbytes`. The honest baseline table is the one in
Result 1 (8.00 bits/cell for streaming), not the 0.0 the harness was printing.

The general lesson, and it is uncomfortable: we wrote the threat model, we
instructed six auditors to find this exact class of bug in candidate code, and
the bug was in ours the whole time. **Audit the incumbent with the same tooling
you point at submissions.**

### Result 2 — the search loop ran, and it worked

The first real agentic search campaign in this arena. Six islands, each proposing
a compressed index representation, scored **only** by the sealed evaluator, each
survivor then independently audited by a skeptic agent instructed to find the
cheat. Target: beat 0.251 bits/cell at fixed exact correctness.

**6/6 islands passed the gate. 6/6 were CONFIRMED by independent verification.**

| module | bits/cell | vs incumbent |
|---|---:|---:|
| **`sub_freeform`** | **0.02566** | **9.78×** |
| `sub_column_diff` | 0.02591 | 9.68× |
| `sub_bitvector_rank` | 0.02735 | 9.17× |
| `sub_delta_varint` | 0.03528 | 7.11× |
| `sub_bitpack_fixed` | 0.03594 | 6.98× |
| `sub_eliasfano` | 0.03604 | 6.96× |
| *(incumbent `sub_rlpbwt_streaming`)* | 0.25092 | — |

Re-scored by us on further fresh seeds (4242, 9191, 55555): identical values,
gate passing, all returning the same 9,552-match set. Every candidate is a
*queryable index*, not a compressed blob.

**The plausibility floor, which is the check that makes this believable.**
`xz -9e` on the bit-packed panel reaches 0.03234 bits/cell — and it is not an
index at all, merely storage. The top three candidates are **below** that:
0.02566, 0.02591, 0.02735. A queryable exact index smaller than what a
general-purpose compressor achieves for pure storage is a meaningful result, and
it rules out the obvious "you're just compressing well" objection. (It also
caught a real overstatement: `sub_delta_varint` at 0.03528 claims in its own
docstring that no general-purpose compressor beats it. xz does. The verifier
flagged it; the claim is wrong and the code is fine.)

The verification was not a rubber stamp. Auditors ran `gc.get_referents` walks to
prove no numpy array was reachable from the index, checked `__slots__` to show a
`__dict__` was not even representable, pickled the index to bound its true size,
**destroyed the panel in place and re-queried** to prove it was not retained,
reconstructed the full panel bit-for-bit to prove the codec was lossless, and
re-derived correctness against a brute-force enumerator written from the spec
rather than trusting `oracle.py`. One ran 600 randomized differential tests plus
8 adversarial edge cases.

**What this does and does not show.** It is a space result on the *permutation*
half of the index, at the small tier. The divergence array is still recomputed
during the sweep, so a single online query remains O(N·M) and the Finding 6
target — `d` in o(N·M) space with O(1) access — is still untouched. And per
Finding 10 these numbers are not yet scored results until they are re-run at
N ≥ 10,000.

**The loop's own verdict on the method:** a 9.78× improvement on a target that a
literature audit had localized precisely, found in one campaign, with every
result surviving adversarial audit. That is the verifier-first strategy
working end to end — the evaluator did the discriminating, and the agents did
the searching.

### Result 3 — the verifiers found a hole in our own harness

Worth recording because the arena's credibility rests on the harness, not on the
candidates.

`check_lock()` was **trust-on-first-use**: if `immutable.lock` was missing it
silently *created* it and reported "ok". So deleting the lock, editing
`oracle.py` to weaken the gate, and re-running would have re-locked to the
tampered files with a cheerful pass. The lock was also not covered by any hash of
its own.

Fixed and verified: a missing lock is now a hard refusal to score; re-creating it
requires an explicit `--init-lock`, which is the only path allowed to write it
and now also handles the stale-lock case. Confirmed by test — stale lock →
`MODIFIED: score_submission.py`, missing lock → `refusing to score`.

**Residual weakness, stated rather than hidden:** the lock is an unsigned local
file, so the real protection is that it be committed to git where any edit is
visible in history. `agentic_discovery/` is currently **git-untracked**, which an
auditor correctly flagged: there is no baseline to diff the immutable set
against. That should be committed.

### Result 1 — a submission that meets P01's space target (`sub_rlpbwt_streaming`)

The first submission built against the measured target rather than a guess.

**The observation.** During a left-to-right sweep, *every* panel access the
algorithm makes reduces to `alleles[j]` — the allele of the haplotype at position
j of the current PBWT column. The streaming submission computes that as
`panel[cur_a, k]`, which is why it must carry the panel (8 bits/cell as uint8).
But the PBWT column in `a_k` order **is** the run-length-encoded object the
RLPBWT literature stores, so we can hold the runs and decode column k on the fly.
The panel is then never touched at query time at all.

**Measured, small tier (980 × 9,330 = 9.14M cells, seed 42):**

| submission | stored bytes | bits/cell | vs bit-packed panel |
|---|---:|---:|---:|
| `sub_vectorized_pbwt` (dense a+d + panel) | 82,335,760 | 72.04 | 72× larger |
| `sub_streaming_pbwt` (panel only, uint8) | 9,143,400 | 8.00 | 8× larger |
| *(same, if bit-packed)* | 1,142,925 | 1.00 | — |
| **`sub_rlpbwt_streaming` (runs only)** | **286,778** | **0.251** | **4× smaller** |

50,700 runs, 5.43 per column. That is **287× smaller than the dense index, 31.9×
smaller than the uint8 panel, and 4× below the bit-packed panel itself.**

**Exactness:** 40/40 randomized differential tests against the oracle, 12
synthetic + 2 coalescent gate instances, and byte-identical agreement with every
other submission (9,552 matches on the local instance; 3,393 on the 8-query A/B).

**Speed:** interleaved A/B, 7 repeats, bootstrap CI on the median ratio —
**1.052× faster than streaming, 95% CI [1.038, 1.075]**. Entire CI above 1.0 and
point estimate ≥ 1.05, so it is claimable under §II.3 — but only just, and the
honest summary is "not slower," not "faster." The space result is the finding.

**This meets the space half of the PRIMARY beat target** ("under 1.0 bit per panel
cell without being slower than the uncompressed PBWT"): 0.251 bits/cell, exact,
not slower.

**What it does NOT do — and this is the important half.** The divergence array is
still *recomputed* during the sweep in O(N) per site, not *stored* with O(1)
access. So a single online query still costs O(N·M), and the headline open
problem from Finding 6 — represent `d` in o(N·M) space with O(1) access — is
untouched. The measured RLE cost of `d` is 51 bits/cell against 0.355 for the
permutation; that asymmetry is still the whole game.

And per Finding 10, this result is at a panel size where the benchmark cannot yet
distinguish index quality. **It must be re-run at N ≥ 10,000 on the HPC node before it
counts as a scored result** (`cluster/slurm/p01_large_n.slurm`).

### Finding 10 — P01's scored tier is in the wrong regime, and one spec claim is retracted

Three measurements, run in-repo and reproducibly (`run_stats.py`, `scaling_N.py`).

**(a) The audit's run-structure numbers replicate exactly.** Independently
re-derived: r_d/cell = 0.797 (seed 42) and 0.813 (seed 7); r = 48,372 on seed 7 —
identical to the audit. The space split is now quantified:

| representation | bits/cell |
|---|---:|
| dense `a`+`d` (current baseline) | 64.0 |
| bit-packed panel (the yardstick) | 1.0 |
| **RLPBWT `a`-side, 1 word/run** | **0.355** |
| **RLE of `d`** | **51.0** |

The permutation compresses **180×**; the divergence array compresses **1.25×**.
The target named in Finding 6 is confirmed by our own measurement.

**(b) "N-independent query" is impossible for this problem — retracting a claim.**
Measured output size grows as **matches/query ~ N^1.31**. Since the answer itself
grows superlinearly in N, per-query time *must* grow with N; no index can prevent
it. Normalising by output, both submissions are already sub-output-linear
(time ~ output^0.72 stored, ^0.66 streaming), i.e. per-match cost *falls* as N
grows (786 µs/match at N=242 → 303 µs at N=3,992).

This retracts the "engineering headroom" claim in Finding 6, which compared our
0.247 s at N=980 against Dynamic μ-PBWT's 0.2325 s at N=100,000 and concluded
"both are N-independent per query, so the gap is overhead." That comparison is
invalid: μ-PBWT answers **SMEM** queries, whose output is O(1) per position,
while P01 reports **all long maximal matches**, whose output is Θ(N^1.31). This
is precisely the all-long-match / set-maximal / k-SMEM disambiguation the audit
itself flagged — and then fell into. **Any P01 timing compared against the
literature must first confirm the other tool solves the same problem.**

**(c) The index is currently dominated by having no index at all.** At
N=1,936 × M=9,263, stored-index (144 MB, 64 bits/cell) versus the zero-byte
streaming submission, wall-clock including build:

| queries | stored-index | streaming | winner |
|---:|---:|---:|---|
| 1 | 1.64 s | 1.57 s | streaming |
| 8 | 6.04 s | 5.21 s | streaming |
| 64 | 41.50 s | 31.91 s | streaming |

And at N=3,992 the 341 MB index loses by **1.56×** — it is not merely wasteful
but actively counterproductive, almost certainly memory traffic against a
structure that cannot be cached.

**Consequence for the benchmark, which is the real finding:** at laptop-scale N
the no-index baseline dominates, so **P01's small tier cannot distinguish index
quality at all** — the thing the problem exists to measure is invisible there. An
index earns its bytes only when the O(N·M) sweep becomes prohibitive, which is
the regime the literature actually operates in (μ-PBWT reports N = 100,000).

Therefore: **P01's scored tier moves to N ≥ 10,000, and laptop runs are demoted
to correctness-gate duty only.** This is the same failure mode as Finding 5 in a
new costume — a benchmark configuration that does not cover the regime where the
science lives certifies nothing. It is also exactly what the HPC node is for.

### Finding 9 — the size of an "engineering win" is itself machine-dependent

All gates re-run on an HPC compute node (52 s total, 1 thread pinned):

| submission | build (laptop) | build (HPC) | query/q (laptop) | query/q (HPC) | index B/cell |
|---|---:|---:|---:|---:|---:|
| `baseline_pbwt` | 56.2 s | **3.40 s** | 0.252 s | 0.104 s | 8.00 |
| `sub_vectorized_pbwt` | 0.662 s | **0.274 s** | 0.254 s | 0.100 s | 8.00 |
| `sub_streaming_pbwt` | ~0 | ~0 | 0.211 s | 0.0797 s | **0.00** |

The vectorization "win" — the submission with *zero* algorithmic content — is
**85× on the laptop and 12.4× on the HPC node.** Same code, same instance size, same
comparison; a sevenfold difference in the apparent magnitude of the result,
produced entirely by the machine. A Python interpreter loop is punished far more
severely on a slow core than on a fast one, so the engineering gap shrinks as the
hardware improves.

This is Finding 2 with a second edge. It is not merely that wall-clock cannot
separate an algorithm from an implementation — it is that **the size of the
implementation effect is not even a stable quantity.** Any arena that reports
speedups without pinning the node is reporting a property of its hardware.
Leaderboard A ranks on deterministic counters for exactly this reason.

*Discipline note, applied to ourselves:* the laptop column and the HPC column
have different `instance.sha` (msprime 1.3.1 vs 1.4.3 — 9,552 vs 9,603 matches),
so per Finding 8 this table is **indicative, not a scored claim**. It is shown
here to make a design point, and the scoring step would refuse to compare these
rows. Same for P02: 64.3 s / recall 0.332 / precision 0.791 on the laptop versus
5.03 s / 0.341 / 0.745 on the HPC node — a 12.8× speedup and a real metric shift, across
two different instances.

### Finding 7 — P09's headroom was refuted by the real incumbents

The scaffold-fidelity rule (§II.2) exists to catch wins scored against a weak
in-scaffold reference. On the HPC node it caught one of ours.

Earlier, comparing our NNI baseline against **random restarts of itself**, we
measured an 11.74 log-unit gap in the low-information regime and recorded it as
headroom. Benchmarking against **raxml-ng 2.0.2** and **IQ-TREE 3.1.3** — with
every tree, ours and theirs, re-scored by our own Felsenstein implementation
after parsing their Newick (round-trip verified exact to 1e-8) — refutes it:

| tier | taxa × sites | arena-NNI deficit | raxml-ng | IQ-TREE | our slowdown |
|---|---|---:|---:|---:|---:|
| small | 16 × 1000 | 0.108 | 0.000 | 0.000 | 9.5× |
| hard16 | 16 × 200 | 0.78 / 0.34 | 0.000 | 0.000 | ~4× |
| hard32 | 32 × 300 | 15.58 / 4.74 | 0.307 / 0.063 | **0.000** | ~10× |
| hard48 | 48 × 300 | 20.38 / 9.13 | 0.238 / 0.017 | **0.000** | ~13× |

Two independently-implemented SOTA searches land within **0.02–0.31 log-units**
of each other on every instance. That is what "no headroom" looks like. Our
baseline trails by up to 20 log-units and runs 10–13× slower, failing the 1.5×
scaffold-fidelity bar by a wide margin. **The 11.74 figure is retracted, and P09
is demoted pending the 128–256 taxa check** (`cluster/slurm/p09_disagreement.slurm`).

A secondary result worth keeping: at hard48 the incumbents reach the ML optimum
with RF = 12–16 to the *true* tree (max 90). The textbook warning made concrete —
at low information the ML tree is not the true tree, so topological accuracy can
never be the scored objective.

**The methodological point generalizes.** Every problem in the arena needs this
check before it is scored, not after. Measuring headroom against your own
baseline measures your own baseline. It cost one job on the HPC node to learn that P09 is
saturated; it would have cost a whole search campaign to learn it afterwards.

### Finding 8 — instance generation is not version-stable

The same seed produces **different instances** under different msprime versions.
Measured across the laptop (msprime 1.3.1 / tskit 0.5.7) and HPC (msprime 1.4.3 /
tskit 1.0.3):

| | laptop | HPC |
|---|---|---|
| P02 tiny/42 sites | 7,382 | 7,384 |
| P01 small/42 matches | 9,552 | 9,603 |
| P09 small/42 lnL | −9817.409561258373 | −9817.409561258373 (identical) |

P09 is exempt because it uses only numpy's RNG; anything touching msprime is not.
Comparing a laptop row to an HPC row would have silently measured the instance
difference rather than the algorithm — a 0.5% swing in match count, which is the
same order as a plausible algorithmic claim.

Fix: `harness/measure.py:instance_fingerprint` records a content SHA plus all
generator versions in **every** result row, and the scoring step must refuse to
compare rows whose `sha` differs. This is the `task.lock` idea from §II.3, made
mandatory rather than aspirational.

**Addendum (release check, 2026-10-23).** A fresh environment with msprime 1.4.3 /
tskit 1.0.3 / numpy 2.4.6 on macOS arm64 reproduces the *laptop* column (7,382
sites; 9,552 matches) and the P09 lnL to the last digit. The drift therefore tracks
the platform (macOS vs the Linux HPC node, plausibly floating-point differences in
the math library) at least as much as the msprime version. The fix is unchanged:
never compare rows whose `instance.sha` differs.

### The P01 → P02 transfer, measured

P02 was selected over higher-ranked problems on the claim that its dominant cost
*is* P01's problem, so one breakthrough wins two benchmarks. That claim is now
measured rather than asserted. Stage decomposition of the P02 incumbent
(600 haplotypes × 43,521 sites, min over 3 runs each):

| stage | time | share |
|---|---:|---:|
| **1 — seeding (all-pairs long matches = P01)** | 68.423 s | **99.98%** |
| 2–4 — filter, extend, report | 0.012 s | 0.02% |

The Amdahl consequence is close to ideal: a k-fold speedup of the P01 index
transfers almost undiminished to the full P02 pipeline — 2× → 2.00×, 5× → 5.00×,
10× → 9.98×. So an agent working only on P01's index, never touching a line of
IBD logic, moves both leaderboards. This is the strongest argument for pairing
them, and it is the kind of claim the arena should be forced to demonstrate
before it is allowed to make it.

### Tier-0 calibration result — and what it costs us

`harness/calibrate.py` runs the incumbent against **itself** under the full
interleaved A/B protocol. Identical code in both arms, so the true ratio is
exactly 1.0. Result on P01 (vectorized PBWT query, 7 repeats, this laptop):

```
candidate median 2.2408s   incumbent median 2.1189s
speedup ratio    1.0576    95% CI [0.9759, 1.1101]
spread (max/min) 1.720x                          PASSED
```

It passed — the CI straddles 1.0 — but read the numbers before celebrating:

- **Identical code posted an apparent 5.8% speedup.** A naive benchmark reporting
  point estimates would have announced a win here.
- **The point-estimate rule alone would NOT have caught it.** §II.3 sets the win
  threshold at "CI entirely above 1.0 **and** point estimate ≥ 1.05" — and 1.0576
  clears the 1.05 half. Only the CI clause rejects it. On this machine the
  conjunction is not belt-and-braces; the CI is doing all the work.
- **Spread was 1.72× max/min**, with a 3.53 s outlier against a 2.05 s minimum —
  ordinary laptop scheduling noise. §II.3's escalation rule (r → 11 when IQR
  exceeds 3% of the median) fires immediately here.

**Practical consequence for this arena, stated up front:** on an unpinned laptop
the noise floor is roughly ±10%, so **wall-clock speedups below ~1.2× are not
claimable at the small tier**, no matter how many times they reproduce. Laptop
runs are for the *search loop* and for the algorithmic counters, which are
deterministic and immune to this. Every wall-clock claim moves to a pinned
HPC node under the full §II.3 protocol. This is exactly why Leaderboard A ranks
on counters rather than seconds.

### The P09 headroom probe, in full

Six independent random starts per setting, 45 s budget each (`difficulty_probe.py`):

| taxa | sites | mean blen | NJ+NNI lnL | best random start | spread across starts | NJ deficit | RF to true |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1000 | 0.10 | −10508.31 | −10508.35 | 17.40 | 0.00 | 0 |
| 16 | 200 | 0.30 | −3529.52 | **−3517.78** | 143.99 | **11.74** | 10 |
| 32 | 300 | 0.30 | −10650.44 | −11824.04 | 714.47 | 0.00 | 0 |
| 48 | 300 | 0.40 | −17361.53 | −18936.75 | 383.82 | 0.00 | 48 |

- The surface **is** rugged: independent starts finish 17–714 log-units apart, so
  hill climbing genuinely gets stuck.
- **But the starting tree dominates.** NJ+NNI beat the best of six random starts
  in three of four settings. The tempting hypothesis "random restarts beat hill
  climbing" is **false** here — worth knowing before building a search loop on it.
- The incumbent is beatable only in the **low-information regime** (16 taxa × 200
  sites, long branches): an 11.74 log-unit deficit, with NJ+NNI's tree RF=10 from
  the truth. That regime is where a scored tier must live.

**Next steps, in order.** (1) Benchmark P09's baseline against real `raxml-ng`
and `iqtree`; publish the scaffold-fidelity ratio. (2) Run the Tier-0
incumbent-against-itself calibration on P01. (3) Only then point a search loop at
P01's index-size target (8.0 → below 1.0 bytes per panel cell at equal
correctness), and measure whether the improvement transfers into P02's stage 1.
