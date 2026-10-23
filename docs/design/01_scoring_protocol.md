# Scoring Protocol for a Bioinformatics Algorithm-Discovery Arena

*Design document. Verified claims carry URLs; a "What is verified vs. inferred" section at the end separates sourced facts from my judgment calls.*

---

## 1. The Pareto Problem

**Recommended default: a two-layer rule — an ε-constraint admissibility gate, then a single-objective ranking inside three named tracks. Strict Pareto dominance is used for the *archive*, never for the leaderboard.**

### Layer A — admissibility gate (hard, pre-declared)

A candidate is admissible on a task only if all four hold:

1. **Validity.** Output passes a semantic validator, not a format checker. For a mapper: CIGARs consistent with the query and reference, MAPQ in range, no duplicate primary records. For an imputer: every target genotype present, dosages in [0,2], site order matching the panel. Failure → score 0, no partial credit. AlgoTune's structure is the same — an `is_solution(problem, proposed_solution)` verifier, and invalid output scores a speedup of exactly 1× rather than being silently dropped ([arXiv:2507.15887v3](https://arxiv.org/html/2507.15887v3)).
2. **Accuracy non-inferiority.** accuracy ≥ incumbent − δ, with δ a per-task, pre-declared practical-equivalence margin, decided by the test in the next subsection.
3. **Memory cap.** peak RSS ≤ κ × incumbent peak RSS, default κ = 1.0.
4. **Index cap.** on-disk index bytes ≤ incumbent's, where an index exists.

### Layer B — one objective per track

Among admissible candidates, rank by a single objective. Run three tracks per task:

| Track | Constrained | Minimized / maximized |
|---|---|---|
| **T** (default) | accuracy ≥ inc − δ, RAM ≤ inc, index ≤ inc | minimize algorithmic cost (§2), wall clock co-reported |
| **M** | accuracy ≥ inc − δ, time ≤ inc | minimize peak RSS |
| **A** | time ≤ inc, RAM ≤ inc | maximize accuracy |

Each track is a clean ε-constraint problem: single-objective, statistically testable, and directly interpretable as an adoption claim ("drop-in replacement, same accuracy, same memory, faster"). Track A is where a genuinely new algorithm with better sensitivity lands, and it is the one most arenas forget to build.

**Why not the alternatives.** Strict Pareto dominance almost never fires, and when it does it is fragile to noise — a 0.3 % RAM regression would kill a 3× speedup. It also produces a *set*, not a ranking, so it cannot drive evolutionary selection. Scalarization with declared exchange rates gives you a scalar but the rates are arbitrary and immediately gameable (dump RAM to buy time). Use scalarization only if you must produce one number for an abstract, and then publish the rate before the round *and* the rank stability under ±50 % perturbation of it.

**Where Pareto belongs.** Maintain a 4-D non-dominated archive (accuracy, time, RAM, index size) and report hypervolume improvement over an incumbent-anchored reference point as a descriptive secondary statistic. Use the archive — not the scalar leaderboard — as the parent pool for the search loop, so diversity survives. AlphaEvolve-lineage harnesses do this with island models; Vesper runs 5 islands, migration interval 50 algorithms, migration rate 0.1, ring topology ([arXiv:2605.15221](https://arxiv.org/abs/2605.15221)).

### The exact "within noise" test

Two noise sources need different treatment, and conflating them is the most common error.

**(a) Accuracy differences are a *sampling* problem, not a seed problem.** For deterministic tools on a fixed dataset, run-to-run accuracy variance is zero. The uncertainty is over the finite sample of evaluation units. So:

- Define an evaluation unit per task: **read** (mapping), **(sample, marker) genotype** (imputation), **reference alignment column** (MSA), **replicate search** (phylogenetics), n/a (k-mer counting is exact).
- Compute Δ = acc(candidate) − acc(incumbent) on the *same* units.
- **Paired block bootstrap, 10,000 resamples.** Block, don't resample units independently: resample by contig or 1 Mb window for mapping, by LD block for imputation, by gene family for MSA. Report the 95 % percentile CI on Δ.
- **Non-inferiority rule:** pass the gate iff the *lower* bound of the 95 % CI on Δ > −δ. One-sided at α = 0.025. The burden of proof sits on the challenger, deliberately.
- **Superiority claim (Track A):** lower bound > 0 *and* point estimate ≥ a pre-declared minimum important difference.

Recommended starting δ (these are my judgment, not sourced):

- **minimap2 / long-read mapping:** δ = 0.001 in mapping F1 at MAPQ ≥ 10 against simulated ground truth.
- **Beagle / imputation:** δ = 0.002 in mean aggregate r² for MAF ≥ 0.5 %; **plus a separate, non-negotiable gate at δ = 0.005 for MAF 0.05–0.5 %.** Rare-variant accuracy is precisely what a cheap state-space approximation destroys, and a single pooled r² will hide it.
- **MAFFT:** δ = 0.005 in SP-score and in TC-score on BAliBASE/HomFam-style references (both, not either).
- **RAxML-NG:** log-likelihood is unbounded, so a δ on LnL is meaningless in isolation. Use: final LnL within 2 log-units of the incumbent's best-of-N, *and* RF distance to the reference tree no worse by more than 0.5 %.
- **KMC3:** no δ. The gate is bit-identical output (sorted k-mer/count multiset).

**Sizing rule for δ:** the sealed test set must be large enough that δ ≥ 3 × the bootstrap SE of Δ. Check this when the task is authored. If it fails, enlarge the test set — never loosen δ.

**(b) Runtime, RAM, and stochastic-algorithm noise.**

- **Deterministic tools:** no algorithmic seed exists. Replication is purely about measurement noise → **r = 5** timed repeats, see §3.
- **Stochastic algorithms** (RAxML-NG's search, Beagle's sampling, and any candidate that introduces sketching, subsampling, or randomized linear algebra): the seed is a real experimental factor. **s = 20 independent seeds** per arm. Seeds are not meaningfully pairable across different algorithms, so treat as two independent samples and bootstrap the difference of medians (10,000 resamples). **Also report the worst seed.** An algorithm that is fast on average and catastrophic 1-in-20 is not deployable, and the mean will hide that.
- If you want *s* derived rather than asserted, use the Kalibera–Jones cookbook, which computes the repetitions needed at each level (build / execution / iteration) for a target precision and reports effect-size confidence intervals ([dl.acm.org/doi/10.1145/2464157.2464160](https://dl.acm.org/doi/10.1145/2464157.2464160)).
- **Multiplicity:** do *not* apply a family-wise correction to non-inferiority gates — that makes the gate easier to pass, in the wrong direction. Apply Benjamini–Hochberg at q = 0.05 across *superiority* claims only, within a round.

---

## 2. The Implementation-vs-Algorithm Confound

**Recommended default: a mandated strong scaffold with a small mutable core, instrumented work counters as the primary metric, and two explicitly labeled leaderboards. Never a language handicap factor.**

### (a) Mandated scaffold, immutable everywhere except the algorithmic core

Ship, per task, a harness repository in **one language for all entrants — I recommend Rust** (hermetic builds via cargo, memory safety inside the sandbox, competitive with C). It contains:

- All I/O — FASTA/FASTQ/BAM/VCF parsing and writing — already fast, mmap-backed, multithreaded, and **immutable**.
- Output serialization and the validator: immutable.
- A trait defining the algorithmic contract. `build_index(refs) -> Index` and `map(&Index, read) -> Vec<Alignment>`; `impute(&Panel, &Target) -> Dosages`; `count(ReadStream, k) -> KmerTable`.
- A reference implementation of the **incumbent's algorithm** (not its codebase) inside that same scaffold.
- Mutable region delimited by `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` sentinels (the AlphaEvolve/OpenEvolve convention). The harness hashes every immutable file and refuses to score a diff that touched one.

**The non-negotiable part: scaffold fidelity.** If your in-scaffold minimap2 reimplementation is 5× slower than real minimap2, every "win" is an artifact of your own bad reference. Require the scaffold reference to land within **1.5×** of the real incumbent's wall clock on the medium tier before the task goes live. Publish that ratio as the task's **scaffold fidelity number** in every result table. This is the expensive part of building the arena and there is no way around it.

### (b) Implementation-independent counters — instrument these exactly

Counters are incremented **inside the immutable API layer**, not by agent code, so a candidate cannot under-report. Per task:

| Task | Primary algorithmic counters |
|---|---|
| **Read mapping** (minimap2) | minimizers emitted; index probes; anchors returned; **chaining DP predecessor comparisons**; **base-level DP cells entered** (counted per logical cell, SIMD-independent); reference bases touched |
| **k-mer counting** (KMC3) | k-mers ingested; hash/radix probes; **bytes spilled to and re-read from temp storage**; comparison count in sort; **bits per distinct k-mer** in the final table |
| **Imputation/phasing** (Beagle) | **Li–Stephens HMM state-transitions evaluated** = Σ_markers (#states at marker) — this is the exact quantity a composite-reference/PBWT state reduction attacks; PBWT block operations; emission evaluations |
| **MSA** (MAFFT) | pairwise distance evaluations; **DP cells filled** in pairwise and profile alignment; profile column comparisons; guide-tree merges; refinement rounds |
| **ML phylogenetics** (RAxML-NG) | **conditional-likelihood-vector updates**, in units of sites × states × rate categories; NNI/SPR moves evaluated; branch-length optimization iterations; full-tree likelihood evaluations |

Plus a **universal set**, identical for every task and every arm:

- **Retired instructions**, user-mode only: `perf stat -e instructions:u`. For determinism across machines, `valgrind --tool=cachegrind`. Instruction counting is the standard trick for noise-free benchmarking in noisy or virtualized environments — this is the entire premise of `iai` and `iai-callgrind` ([github.com/bheisler/iai](https://github.com/bheisler/iai), [crates.io/crates/iai-callgrind](https://crates.io/crates/iai-callgrind)). **Caveat, and it is real:** instruction counts systematically undercount SIMD wins and overcount branchy code. Instruction count is *complementary*, never the headline.
- **Cache/memory traffic:** `perf stat -e LLC-load-misses,LLC-store-misses`, or cachegrind's simulated D1/LL miss counts when determinism matters more than speed.
- **I/O volume:** `/proc/<pid>/io` `read_bytes` / `write_bytes` (actual block-layer traffic), plus cgroup-level totals. For KMC3-class and panel-heavy Beagle-class problems this often *is* the algorithm.
- **Bits per element** for any persisted index: index_bytes × 8 / #indexed elements. Comparable across dataset sizes; the honest way to state an index-compression win.
- **Peak RSS** from cgroup v2 `memory.peak`.

Designate exactly **one primary counter per task** as the headline for Leaderboard A (DP cells for mapping and MSA, HMM state-transitions for imputation, bytes spilled for k-mer counting, CLV updates for phylogenetics). Report the rest.

### (c) Dual leaderboards, both published, both required

- **Leaderboard A — Algorithmic.** Same scaffold, same language, same immutable I/O. Cost = the designated primary counter, at fixed admissibility. **This is the scientific claim.**
- **Leaderboard B — Engineering / wall-clock.** Any language, SIMD intrinsics, GPU, anything — provided the output validates and the gate passes. Cost = wall clock and peak RSS against the **real released incumbent binary on the same node**. **This is the usability claim.**

The headline arena result is the *pair*. A-win + B-loss = "promising algorithm, unengineered" — publish it as exactly that; it is a real and useful result. B-win + no A-movement = "engineering win, no algorithmic advance" — also publish, also labeled. This names the confound instead of burying it. MLPerf solves the same problem with Closed vs. Open divisions: Closed requires equivalence to a reference implementation for apples-to-apples comparison, Open permits model and pipeline changes and must be labeled "MLPerf Open" ([inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)).

### Two guards you will need

- **Counter-avoidance detection.** "Fewer DP cells" is trivially achieved by not aligning. The accuracy gate in §1 is what makes counters meaningful — counters are never scored alone. Additionally, flag for human review any candidate whose primary counter drops > 10× while accuracy is flat: that is either a breakthrough or a leak, and you want to know which. Vesper assigns this role to an independent agent session that reads the candidate's code and judges whether it solves the problem or exploits the evaluator, excluding flagged candidates from the parent pool — and found that **stronger models hack more**: 16.6 % hack rate for GPT-5.2-codex vs 0 % for the mini model ([arXiv:2605.15221](https://arxiv.org/abs/2605.15221)). Budget for this pass; it is not optional.
- **Engineering headroom constant.** Once per task per round, implement the incumbent's algorithm both naively and optimally in the scaffold. Publish the ratio. If a candidate's Leaderboard-B win is smaller than that constant, you cannot attribute it to the algorithm, and you must say so.

**Do not use language handicap factors** ("divide Python by 30"). They are indefensible and will be the first thing a reviewer attacks. Fix language by mandating the scaffold.

---

## 3. Hardware and Measurement Normalization

**Node.** One dedicated bare-metal node class per round. No VMs, no shared cloud instances for official numbers. Emit a `system_description.json` with every result: CPU model, microcode revision, DIMM configuration, NUMA topology, kernel version, glibc version, filesystem, storage class (local NVMe vs. network).

**Containers.** One OCI image for candidates and incumbent alike, base image pinned by sha256 digest, never by tag. Apptainer/Singularity on HPC. Containers are for *dependency reproducibility*, not performance isolation — isolation comes from cgroups and pinning.

**Isolation — exact settings.**

- Boot with `isolcpus=`, `nohz_full=`, `rcu_nocbs=` over the benchmark core set; leave core 0 and its sibling to the OS.
- **SMT off** in BIOS. Halves core count, roughly halves your variance.
- **Turbo off** (`intel_pstate/no_turbo=1`, or fixed frequency via `cpupower` on AMD). Turbo residency depends on what other cores are doing — exactly the confound you are trying to kill.
- Governor `performance`, min = max frequency. **Verify per run**: read `/proc/cpuinfo` MHz at start and end, record both, abort if drift > 2 %.
- `cset shield` / `taskset -c` to a fixed physical core list on **one NUMA node**; `numactl --membind` to the same node.
- **Leave ASLR on** (`randomize_va_space=2`) and run enough repeats instead. Disabling it buys precision at the cost of a systematic bias. Measurement bias from causes as innocuous as link order and UNIX environment size is large, ubiquitous, and was found in most SPEC CPU2006 C programs across three architectures and two compilers ([Mytkowicz et al., ASPLOS '09](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf)). Apply that paper's remedy: **setup randomization** — randomize environment-block size and run order across repeats, and record the randomization seed in the result row.
- Transparent hugepages set to a fixed declared value (`never` is my default; `always` creates variance that is hard to attribute) and recorded.
- `nmi_watchdog=0`; disable cron, unattended-upgrades, and irqbalance for the duration of a round.

**Threads.** Two mandatory configurations: **T = 1** (the honest algorithmic comparison) and **T = 8** pinned to 8 physical cores on one NUMA node (the practical one). Report both. The harness — never the candidate — sets thread count, via the tool flag *and* `OMP_NUM_THREADS` / `RAYON_NUM_THREADS`. Verify: read `Threads:` from `/proc/self/status` and compute cpu-time / wall-time from the cgroup. If that ratio exceeds T by more than 5 %, disqualify the run.

**Cache state.** Measure and report **both**, and declare the headline per task in advance:

- **Cold** = `sync; echo 3 > /proc/sys/vm/drop_caches` before each timed repeat. **Headline for index-heavy / I/O-bound tasks** (mapping against a large index, k-mer counting, panel-based imputation), because a warm page cache erases exactly the class of win those algorithms produce.
- **Warm** = one untimed warmup pass, then timed. Headline for compute-bound tasks (MAFFT, RAxML-NG).

**Same-node A/B — interleaved, mandatory.** For each dataset run `I C I C I C I C I C`, never `IIIII CCCCC`. Randomize which arm starts. Interleaving is the cheapest and highest-value protection against thermal drift and slow environmental change, and it is the rule most often skipped.

**Repeats and statistic.**

- **r = 5** timed repeats per (arm, dataset, thread count, cache state). Escalate to **r = 11** if the IQR exceeds 3 % of the median.
- **Report min, median, and the raw times. Decide on the median**, with a **95 % bootstrap CI on the ratio median(candidate)/median(incumbent)**, 10,000 resamples of the paired repeat sets.
- Rationale for reporting both: `min` is the right estimator when noise is purely additive, and it is what AlgoTune uses — untimed warmup, one timed run, repeated 10×, minimum kept, via `time.perf_counter_ns` ([arXiv:2507.15887v3](https://arxiv.org/html/2507.15887v3)). But for I/O-bound genomics work the noise is *not* purely additive; it includes real I/O scheduling a user will actually experience. So median is the more honest headline, min the more honest algorithmic signal. SPEC's convention is the conservative precedent: reportable runs use 2 or 3 iterations, taking the median of 3 or the *slower* of 2 ([SPEC CPU2017 run rules](https://www.spec.org/cpu2017/Docs/runrules.html)).
- **Win threshold on wall clock:** the entire 95 % CI on the speedup ratio must lie above 1.0, **and** the point estimate must be ≥ 1.05. Sub-5 % differences on a node like this are not claims.

**Peak RAM.** Do not rely on `/usr/bin/time -v` maxresident alone — it handles child processes and mmap in ways that will mislead you. Default: each arm in its own **cgroup v2**, read `memory.peak` (kernel ≥ 5.19), which captures the whole process tree. Also record `getrusage` max RSS and the peak of a 10 ms sampler over `/proc/<pid>/smaps_rollup` `Pss`. RAM is far less noisy than time: **r = 3**, take the **max** — worst case is the deployable number.

**Index and scratch size.** On-disk index bytes after `sync`; bits/element (§2); and **peak temporary-disk usage during construction**, measured by 100 ms polling of the scratch directory, or by putting scratch on a dedicated tmpfs/loopback with a readable quota. For external-memory algorithms this is a first-class objective, not a footnote.

**Result row schema.** One JSON line per run:

```
task, task_lock_sha, tier, size_n, arm, commit_sha, container_digest, node_id,
threads, cache_state, repeat_index, run_order_index, wall_s, cpu_s,
cgroup_memory_peak_bytes, max_rss_bytes, peak_scratch_bytes,
io_read_bytes, io_write_bytes, instructions_u, llc_misses,
primary_counter_name, primary_counter_value, accuracy_metric, validator_ok,
cpu_mhz_start, cpu_mhz_end, env_randomization_seed
```

Ship raw rows, not summaries. Every table in every writeup must be regenerable from the raw rows by a script in the repo.

---

## 4. Scaling-Curve Scoring

Three tiers is three points. A power-law exponent fit to three points is numerology. Fix it structurally.

**Use a dedicated size ladder, not the tiers.** The small/medium/large tiers drive the headline table. Scaling gets **≥ 7 log-spaced sizes over ≥ 2.5 decades**: e.g. 10⁴ → 10⁷ reads for mapping; 500 → 500,000 reference haplotypes for imputation; 100 → 30,000 sequences for MSA. If the top end is unaffordable, **extend the bottom, not the top** — you need the lever arm, and small points are nearly free. AlgoTune's sizing convention is a useful anchor for choosing the *center* of a ladder: pick n such that the reference solver takes ~100 ms on one core ([arXiv:2507.15887v3](https://arxiv.org/html/2507.15887v3)).

**What to fit.** Fit the **implementation-independent primary counter first** — it is deterministic and it is what "asymptotics" actually means — and wall clock second. Model:

```
log(cost) = a + b·log(n) + ε
```

OLS on log–log; **b** is the empirical exponent. CI on b from a **two-level bootstrap**: resample sizes with replacement, then resample repeats within each size. When the counter is deterministic the inner level collapses, which is precisely why you need ≥ 7 sizes.

**Three mandatory diagnostics, published with every exponent.**

1. **Curvature test.** Also fit `a + b·log n + c·(log n)²` and report c with its CI. **If the CI on c excludes 0, a single exponent is a lie** — report piecewise exponents (low half, high half) instead. This is the most important guard, because most real bioinformatics algorithms have a regime change (in-cache → out-of-cache, in-RAM → external-memory) that masquerades as a better exponent when you sample only one side of it.
2. **Leave-one-size-out stability.** Refit with each size dropped; report max |b − b₍₋ᵢ₎|. If that exceeds the half-width of b's CI, the fit is point-driven — downgrade the claim explicitly.
3. **Alternative-model comparison.** Also fit n·log n and n·(log n)² forms; report ΔAIC. If a power law is not clearly preferred (**ΔAIC < 4**), report **"no reliable exponent"** rather than a number.

**How it enters the score.** Do not put b in a scalar. A great exponent with a terrible constant helps nobody, and it is easy to manufacture. Instead:

- **Primary scaling score = extrapolated cost ratio at XL.** Fit both arms, extrapolate one decade past the largest measured size, report the predicted candidate/incumbent ratio **with propagated CI**, explicitly labeled a prediction. Then, **once per round, actually run the XL point for the top candidate and publish predicted-vs-actual.** That table is how the arena earns the right to talk about asymptotics at all.
- **Secondary = Δb = b_cand − b_inc**, with CI, reported only when both fits pass all three diagnostics. Award the **"asymptotic improvement"** badge only when: the 95 % CI on Δb excludes 0, the curvature test passes for both arms, **and the fitted crossover n lies within or below the measured range.** An algorithm that only wins beyond every size you tested is a conjecture, not a result.
- **Always report the crossover n in plain language** — "faster above ~2 M reads" is the number practitioners actually want.

Fit **memory vs. n** and **index bits/element vs. n** the same way. A sublinear-memory result is often more valuable than a time result and is considerably easier to establish cleanly, because memory measurements are far less noisy.

---

## 5. Governance

**Splits.** Three disjoint layers per task, from different *generative sources* wherever possible — a different random subset mostly tests nothing.

- **DEV (open):** full data and labels, unlimited runs, evaluator source readable. ~30 % of compute budget.
- **VAL (open, rate-limited):** labels withheld; evaluator is a service returning a score. Where the agent gets signal without truth.
- **TEST (sealed):** labels never released; ideally data never released either (run-my-container service). **Different source** from DEV/VAL — different sequencing run, different species, different simulator parameterization, different reference panel. Precedent: the precisionFDA Truth Challenge used a completed-but-unpublished GIAB benchmark as the blind truth set ([FDA](https://www.fda.gov/science-research/fda-science-forum/precisionfda-truth-challenge-v2-using-crowdsourcing-benchmark-variant-calling-innovation)). AlgoTune's split is the same shape: 5 dev tasks for free iteration, 154 held-out test tasks for the score.

**How often the sealed test may be run — use both mechanisms.**

- **Budget:** **N = 5 sealed evaluations per lineage per task per round** (rounds ~quarterly). No rollover. A run consumes budget even if the candidate fails to compile.
- **Ladder-style limited feedback:** the sealed evaluator returns the new score **only if it improves on that lineage's previous best by more than a step η**; otherwise it returns the previous best. This is the Ladder mechanism, which bounds leaderboard error at roughly log(k)^{2/3}/n^{1/3} under adaptive submission ([Blum & Hardt](https://arxiv.org/pdf/1502.04585)); the general theory of holdout reuse is [Dwork et al.](https://arxiv.org/pdf/1506.02629). **Set η = δ**, the practical-equivalence margin from §1 — so the leaderboard structurally cannot resolve differences you already declared meaningless. Cheap, and I would make it mandatory.
- **Return only the primary scalar plus pass/fail per gate.** No per-read error breakdowns, no confusion matrices, no phase timings. Rich diagnostics live on DEV/VAL only. This is the main defense against reverse-engineering the sealed set.

**Pre-registration.** Before every sealed run, submit an immutable record: candidate git SHA, container digest, declared track (T/M/A), declared δ and parameters, **expected VAL result**, and a one-paragraph mechanism claim ("what algorithmic change makes this better, and why"). Hash it, timestamp it, publish it whether or not the run succeeds. Implementation: a hash-chained append-only JSONL ledger (each line carries the SHA-256 of the previous), mirrored; or a git repo with signed tags, which is adequate for a single lab. **The point is that the number of attempts becomes publicly countable** — that is what makes the sealed-test statistics honest, and it is what almost every ML leaderboard fails to do.

**Provenance logging.** Every candidate carries: agent transcript hash, model name and version, token count and cost, search wall-clock, parent ID and lineage, the diff, DEV/VAL score history, and every external resource fetched. Vesper's design is a good template — evolution as git branches, a dedicated **git worktree per agent** for filesystem isolation without full clones, and an SQLite program database recording branch lineage, evaluation results, diffs, and improvement ideas; 4 parallel agents gave 3.2–3.9× wall-clock speedup ([arXiv:2605.15221](https://arxiv.org/abs/2605.15221)).

**Preventing evaluator editing — layered, because any single layer fails.**

1. **The evaluator does not live in the agent's repo.** The agent's worktree holds the scaffold plus a client stub. Evaluation runs in a separate process, in a separate container, launched by the orchestrator, with the candidate mounted **read-only** and the evaluator mounted from an image the agent's container cannot see.
2. **No network egress from the evaluation container**, denied at the network-namespace level. The *search* container may have network for docs; the *evaluation* container never does.
3. Non-root, seccomp profile, read-only rootfs, dropped capabilities, no `ptrace`, no `/proc/sys` writes, quota'd tmpfs scratch.
4. **Hash verification at eval time.** The orchestrator recomputes SHA-256 over every immutable scaffold file and over the evaluator, and refuses to score on any mismatch. Expected hashes are stored outside the container.
5. **Ground truth never enters the candidate's container.** The candidate writes output; a separate scoring pass in a different container compares to truth. This is the structural fix — it makes reading the answers *impossible*, not merely forbidden.
6. **Behavioral checks.** Run twice in fresh containers with different working directories, hostnames, and file mtimes; require identical output (or identical-given-seed for stochastic candidates). Run once with shuffled input order and once on a held-back shard to catch input-specific hard-coding. MLPerf explicitly bans **benchmark detection, input-based optimization, caching queries or responses, and hard-coding the total number of queries**, and restricts non-determinism to FP operation order, fixed-seed random traversal, and rounding ([inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)). Import those clauses close to verbatim.
7. **LLM hack review** on every gate-clearing candidate before it enters the leaderboard *or* the parent pool. Vesper's hack rate for the strongest model tested was 16.6 %.
8. **Human review for any headline claim.** Non-negotiable. A 10× counter reduction gets a person reading the diff.

**Versioning and hash-pinning.** Semantic-version each task (`minimap2-map-ont/v2.1.0`). **Major** bump when the sealed set, δ, gates, or metric definition change — results across major versions are not comparable and the leaderboard must refuse to display them together. **Minor** for added datasets or an extended ladder. **Patch** for score-neutral fixes. Publish a `task.lock` with SHA-256 for: every dataset and truth file, the evaluator source tarball, the container digest, the incumbent's source tarball with exact version/commit and build flags, and the scaffold's immutable file set. Every result row references the `task.lock` hash. **If you fix an evaluator bug, re-run every affected historical result or mark it withdrawn** — never silently leave stale numbers up. AlgoTune's own report of a vector-quantization verifier that failed to check optimality is the canonical case for treating evaluator bugs as retractions.

**Pin the incumbent, and tune it fairly.** Record the incumbent's version, whether it was built from source with `-O3 -march=native` or pulled from bioconda, and its exact CLI parameters. The quiet, common form of cheating in this literature is running the incumbent with defaults that are wrong for the data. Weber et al. are explicit: benchmark authors should have approximately equal familiarity with all methods included, and no method should be favored through more extensive parameter tuning; use one consistent strategy across all methods (defaults, published values, or systematic grid search) ([Genome Biology 2019](https://link.springer.com/article/10.1186/s13059-019-1738-8)). Concrete rule: the incumbent's invocation is set by a **named external expert** from that tool's user or developer community, published in the task spec, frozen for the round, and given the *same* DEV tuning budget the agent had. Offer the incumbent's authors right of reply before publication.

**Honest publication, including negatives.**

- Publish **every pre-registered sealed attempt** in time order with its outcome. The denominator is the result.
- Publish the **Pareto archive and a failure taxonomy**: counts for failed-to-compile, failed-validator, failed-accuracy-gate, failed-RAM-cap, flagged-as-hack, withdrawn-after-review.
- Publish a standing **"no improvement found"** statement per task per round when that is the truth. A task where nothing beat the incumbent across 200 candidates and $X of tokens is a *result about the maturity of that incumbent*, and it is the most scientifically load-bearing output this arena can produce. Give it a DOI.
- Publish the **scaffold fidelity number** and the **engineering headroom constant** every round. Without them, no reader can calibrate any win.
- Publish raw run rows and analysis scripts; summaries regenerable by `make results`.
- **Review period plus audit**, MLPerf-style: a fixed window where participants can inspect submissions and file objections, plus a random audit. MLPerf audits up to two submissions per round (one random, one committee-selected), auditor report within 30 days, audit complete within 90, and submissions failing at a material level are moved to Open or removed. For a small arena the realistic version is: **one winning candidate per round, independently reproduced by a second person on a second node.**
- **Conflict of interest:** whoever authored a task's scaffold and its in-scaffold incumbent reimplementation may not submit candidates to that task in that round.

---

## Minimum Viable Version — one researcher, one week

**Scope to one task, and make it k-mer counting vs. KMC3.** Not read mapping. KMC3 gives you an exact-output correctness gate (bit-identical multiset), so the entire accuracy-metric problem vanishes; the interesting axes are time, RAM, and temp-I/O, which are exactly what you need to learn to measure; the algorithm is small enough to reimplement in a scaffold in a day; and the counters (k-mers ingested, bytes spilled, bits per distinct k-mer) are unambiguous. Read mapping vs. minimap2 is the more exciting task and the wrong first one — defining and validating the accuracy metric alone will consume three days.

**Day 1 — freeze the task.** Seven log-spaced dataset sizes deterministically subsampled from one public read set. Pin KMC3's version and exact invocation (`-k31 -ci1 -cs255`, declared threads, declared max-RAM). Write `task.lock` with SHA-256 of every input and of the KMC3 source tarball. Build one Docker image on a digest-pinned base. That image is the world.

**Day 2 — scaffold.** A small Rust binary: immutable FASTQ reader, immutable sorted-output writer, a `trait Counter` with one `EVOLVE-BLOCK`, and a reference implementation of the KMC3 *approach* (minimizer binning → per-bin radix sort). Instrument the immutable layer with three counters: k-mers ingested, bytes spilled, bytes re-read. Accept 1.5–3× slower than real KMC3 for the MVP — and **write that number down** as your scaffold fidelity number.

**Day 3 — `bench.py`, ~250 lines.** Pins cores with `taskset`; sets governor and `no_turbo` and *asserts* them; drops caches; runs interleaved `I C I C I C I C I C` at r = 5; captures wall time with `perf_counter_ns`, peak memory from cgroup v2 `memory.peak`, I/O from `/proc/<pid>/io`, instructions and LLC misses from `perf stat`. Appends one JSON line per run to `runs.jsonl`. **No aggregation yet** — raw rows only.

**Day 4 — `score.py`.** Validator (bit-identical vs. KMC3 output); admissibility gate (exact output + RAM ≤ incumbent); median-ratio bootstrap CI at 10,000 resamples; log–log fit with curvature test and leave-one-size-out. One markdown table out. **Then score KMC3 against itself.** The pipeline must report speedup 1.00 with a CI straddling 1. If it doesn't, your measurement setup is broken and you just learned that for free. **This is the highest-value hour of the week — do not skip it.**

**Day 5 — governance minimum.** Split the ladder: DEV = 3 sizes from one read set, TEST = 4 sizes from a *different* read set (different instrument or organism). Put truth outputs and the scorer in a directory the agent container never mounts. Add the immutable-file hash check at eval time. Create `preregistration.jsonl` as a hash-chained append-only file and commit to one line before every TEST run. Cap yourself at 5 TEST runs.

**Day 6 — run the loop.** One agent, git-worktree-per-attempt, DEV-only feedback, 20–50 candidates. Log everything. **Read the top 5 diffs yourself** — that is your hack detection for the MVP.

**Day 7 — publish.** `runs.jsonl`, the scoring script, the pre-registration ledger, the scaffold fidelity number, the failure taxonomy counts, and the result — including the likely outcome that nothing beat KMC3. Write down explicitly which pieces you deferred: dual leaderboards, LLM hack review, Ladder-thresholded feedback, XL extrapolation validation, external expert for incumbent tuning.

**The three things the MVP must not skip**, because skipping any of them invalidates everything downstream: **interleaved same-node A/B**, **the incumbent-vs-itself calibration run**, and **the pre-registration counter on sealed runs**.

---

## What is verified vs. inferred

**Verified in this session, with sources:**

- Vesper harness: git-worktree-per-agent isolation, 5 islands / migration interval 50 / rate 0.1 ring topology, independent-agent hack detection excluding flagged candidates from the parent pool, hack rates 16.6 % (GPT-5.2-codex) vs 0 % (GPT-5.1-codex-mini), 4 parallel agents → 3.2–3.9× speedup, ~89.6K tokens/algorithm vs OpenEvolve's ~23.9K, SQLite program database with lineage and diffs — arXiv:2605.15221.
- AlphaEvolve: evolutionary coding agent with evaluator feedback; 4×4 complex matmul in 48 scalar multiplications — arXiv:2506.13131.
- AlgoTune: untimed warmup + one timed run, repeated 10×, **minimum kept**; `time.perf_counter_ns`; `is_solution` verifier; invalid → speedup 1×; **harmonic mean** across tasks; sizes chosen so the reference takes ~100 ms on one core; 5-task dev set vs 154-task test set; a real verifier bug (vector-quantization optimality unchecked) — arXiv:2507.15887v3.
- SPEC CPU2017: reportable runs use 2–3 iterations; median of 3, or the *slower* of 2.
- MLPerf: Closed/Open/Network divisions; Available/Preview/RDI categories; audit of up to two submissions per round (one random, one committee-selected), 30-day report, 90-day completion, material failures moved to Open or removed; bans on benchmark detection, input-based optimization, caching queries/responses, hard-coded query counts; non-determinism restricted to FP order, fixed-seed traversal, rounding.
- Ladder mechanism and its log(k)^{2/3}/n^{1/3} leaderboard-error bound; Dwork et al. on holdout reuse.
- precisionFDA Truth Challenge used a completed-but-unpublished GIAB benchmark as the blind truth set.
- Mytkowicz et al.: measurement bias from link order and environment size is large, ubiquitous, and unaddressed in 133 surveyed papers; remedies are causal analysis and setup randomization.
- Kalibera & Jones: cookbook for repetitions at each experimental level; effect-size confidence intervals.
- Weber et al. (Genome Biology 2019): equal familiarity with all compared methods; no preferential parameter tuning; separate train/test; prefer *multiple* metrics identifying a set of consistently strong methods over a single composite ranking; systematic scalability investigation across sizes; replicates for stochastic methods; transparent discussion of limitations; extensible design.
- Cachegrind/Callgrind instruction counts as stable, environment-noise-immune proxies (iai, iai-callgrind).

**Inferred — my judgment, not from any source:**

- All specific δ values; κ = 1.0; r = 5 / r = 3 / s = 20; 10,000 bootstrap resamples; the ≥7-sizes-over-2.5-decades ladder rule; ΔAIC < 4 as the "no reliable exponent" threshold; the δ ≥ 3 × bootstrap-SE sizing floor; the 5 % minimum reportable speedup; N = 5 sealed runs per round.
- The three-track (T/M/A) structure and the A/B dual-leaderboard structure as applied here.
- **The per-tool counter lists** (chaining and base-level DP cells, Li–Stephens state-transitions, CLV updates, bits/element). These are the natural cost measures for those algorithm families and I am confident they are the right *quantities*, but I did not verify against each tool's source that they are cleanly instrumentable at the API boundary I describe. Budget a day of code reading per tool before committing a task spec.
- "Scaffold fidelity number" and "engineering headroom constant" are my coinages.
- That Beagle's speed advantage comes specifically from composite-reference/PBWT state reduction is background knowledge, not verified here.
- The claim that instruction counts undercount SIMD wins and overcount branchy code follows from how the counter works, but I did not verify a source quantifying it for genomics workloads.

**Sources:** [arXiv:2605.15221 (Vesper / harness engineering)](https://arxiv.org/abs/2605.15221) · [arXiv:2506.13131 (AlphaEvolve)](https://arxiv.org/abs/2506.13131) · [arXiv:2507.15887v3 (AlgoTune)](https://arxiv.org/html/2507.15887v3) · [MLPerf inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) · [SPEC CPU2017 run rules](https://www.spec.org/cpu2017/Docs/runrules.html) · [Blum & Hardt, The Ladder](https://arxiv.org/pdf/1502.04585) · [Dwork et al., holdout reuse](https://arxiv.org/pdf/1506.02629) · [Mytkowicz et al., ASPLOS '09](https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf) · [Kalibera & Jones 2013](https://dl.acm.org/doi/10.1145/2464157.2464160) · [Weber et al., Genome Biology 2019](https://link.springer.com/article/10.1186/s13059-019-1738-8) · [precisionFDA Truth Challenge V2](https://www.fda.gov/science-research/fda-science-forum/precisionfda-truth-challenge-v2-using-crowdsourcing-benchmark-variant-calling-innovation) · [CAMI, Nature Methods](https://www.nature.com/articles/s41592-022-01431-4) · [iai](https://github.com/bheisler/iai) · [iai-callgrind](https://crates.io/crates/iai-callgrind) · [pyperf system tuning](https://pyperf.readthedocs.io/en/latest/system.html)