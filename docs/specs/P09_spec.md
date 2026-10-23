# P09 — Maximum-likelihood phylogenetic tree search
### Bioinformatics Algorithm Arena · benchmark specification v1 · 2026-09-01
### Companion to `docs/SPEC.md` (scoring protocol) and the existing scaffold at `problems/P09_ml_tree_search/`

---

## Formal statement

Fix a finite alphabet $\Sigma$ (DNA, $|\Sigma|=4$), an alignment $D \in (\Sigma\cup\{-\})^{n\times m}$ over taxon label set $L$, $|L|=n$, and a reversible continuous-time Markov substitution model $M$ with **parameters $\theta$ that are given and held fixed** (rate matrix, base frequencies, discrete-Γ rate categories and shape).

Let $\mathcal{T}_n$ be the set of unrooted binary topologies on $L$, $|\mathcal{T}_n| = (2n-5)!!$. For $T\in\mathcal{T}_n$ let $E(T)$ be its $2n-3$ edges and $b\in[b_{\min},b_{\max}]^{|E(T)|}$ its branch lengths. Felsenstein's pruning algorithm computes

$$\ell(T,b;\theta,D)=\sum_{j=1}^{m} w_j \log \sum_{c=1}^{k} p_c \sum_{s\in\Sigma}\pi_s \, \mathrm{CLV}^{(c)}_{\mathrm{root},s}(j)$$

exactly (to floating-point precision) in $O(n\,m\,|\Sigma|^2 k)$ operations, where $w_j$ are site-pattern weights and $k$ the number of Γ categories.

**Objective.** Return $\hat T$ maximizing the *profile* likelihood

$$F(T) \;=\; \max_{b\in[b_{\min},b_{\max}]^{|E(T)|}} \ell(T,b;\theta,D).$$

Find $\arg\max_{T\in\mathcal{T}_n} F(T)$. This is NP-hard (Roch 2006, *IEEE/ACM TCBB* 3:92–94, [doi:10.1109/TCBB.2006.4](https://doi.org/10.1109/TCBB.2006.4)); the *evaluation* of any candidate is exact and cheap. That asymmetry — expensive search, exact free verification — is precisely the FunSearch shape.

**Two deliberate reductions from what real tools do**, both of which make the evaluator ungameable:

1. **$\theta$ is fixed and supplied by the harness** (set to the simulation truth). Real tools co-optimize $\theta$ with $T$; that co-optimization is a numerical-analysis problem, not a combinatorial-search problem, and it introduces a second axis on which two runs can differ for reasons unrelated to search quality. Fixing $\theta$ makes $F$ a well-defined function of the topology alone, so "did the search find a better tree?" has one answer.
2. **$F(T)$ is computed by the harness, never by the submission.** A submission returns a topology (branch lengths optional and used only as an optimizer warm start). This removes every incentive to report a number rather than find a tree.

Constants: $b_{\min}=10^{-6}$, $b_{\max}=100$ (RAxML-NG's `RAXML_BRLEN_MIN`/`RAXML_BRLEN_MAX`, verified in [`src/constants.hpp`](https://raw.githubusercontent.com/amkozlov/raxml-ng/master/src/constants.hpp)).

---

## Input / output / submission API

### Leaderboard A (algorithmic — the scientific claim)

Python 3.11, mandated scaffold. **The alignment is never handed to the submission as an array.** It is owned by an immutable `LikelihoodEngine` that meters every unit of algorithmic work. This is the enforcement mechanism that makes the counter (below) non-circumventable: a submission physically cannot run a private, uncounted likelihood engine because it never sees the characters unless it pays for them.

```python
# harness/p09_api.py  — IMMUTABLE. Submissions import, never modify.

class LikelihoodEngine:
    """Owns the alignment. Every method increments a counter the submission cannot reach."""

    n_taxa: int
    n_sites: int                 # ungapped columns
    n_patterns: int              # distinct site patterns after compression
    taxon_labels: tuple[str, ...]
    model: FixedModel            # GTR rates, base freqs, +G4 shape — READ ONLY

    # --- likelihood primitives (metered in CLV-site-ops) ---
    def loglh(self, topo: Topology, blens=None) -> float: ...
    def loglh_partial(self, topo, changed_edges) -> float:
        """Incremental re-evaluation reusing valid CLVs. Charges only recomputed CLVs."""
    def opt_branches(self, topo, blens=None, *, rounds=1, eps=None) -> tuple[float, np.ndarray]:
        """Newton-Raphson branch-length optimisation. Charges per NR iteration."""
    def opt_branch_triplet(self, topo, edge, blens) -> tuple[float, np.ndarray]:
        """The 3-branch reoptimisation RAxML-NG uses in slow SPR rounds."""

    # --- cheap non-likelihood primitives (metered on separate, cheap axes) ---
    def parsimony(self, topo) -> int: ...
    def ml_distances(self) -> np.ndarray:            # n x n, charged once
    def pattern_weights(self) -> np.ndarray:         # charged once
    def site_patterns(self) -> np.ndarray:           # RAW DATA. Charged once, at the
                                                     # price of one full-tree loglh().
                                                     # Escape hatch for learned heuristics.

class Topology:
    """Unrooted binary tree. Immutable value type with structural hashing."""
    def nni_neighbours(self, edge) -> tuple["Topology", "Topology"]: ...
    def spr_neighbours(self, subtree, radius) -> Iterator[tuple["Topology", int]]: ...
    def to_newick(self, blens=None) -> str: ...
    def rf_splits(self) -> frozenset[frozenset[int]]: ...
```

Submission contract:

```python
# EVOLVE-BLOCK-START
def search(engine: LikelihoodEngine, budget: Budget, rng: np.random.Generator) -> Topology:
    """Return the best topology found. `budget` exposes .ops_remaining and .seconds_remaining
    and raises BudgetExhausted; returning early is allowed and rewarded by the primal integral."""
# EVOLVE-BLOCK-END
```

`rng` is seeded by the harness. Wall-clock is measured but is *not* the Leaderboard-A cost; ops are.

### Leaderboard B (engineering — the usability claim)

Any language, any SIMD/threading. Contract is a CLI:

```
<tool> --msa aln.fasta --model <MODEL_STRING> --seed <S> --threads <T> \
       --budget-seconds <B> --out tree.nwk
```

Scored against the real `raxml-ng` binary on the same pinned node.

### Instance and submission payloads

| Artifact | Format | Notes |
|---|---|---|
| alignment | FASTA, no gaps in scored tiers | taxa named `t0000…t{n-1}` — no phylogenetic information in labels |
| model string | RAxML-NG/IQ-TREE syntax, e.g. `GTR{1.21/3.44/0.87/0.79/4.10}+FU{0.28/0.22/0.23/0.27}+G4m{0.55}` | fully parameterised; nothing free |
| manifest | JSON | seed, generator version, true tree Newick, certified optimum (gate tiers) or BKS, Pythia difficulty, scorer-noise ε |
| submission output | Newick, one tree, unrooted, binary | branch lengths optional |

---

## Ground truth (EXACT vs CURATED vs PROXY, and justify)

Three distinct truths are in play and the spec keeps them **rigorously separate**. Conflating the ML optimum with the generating tree is the single most common error in this literature and it is the error this section exists to prevent.

| Quantity | Class | Provenance | Role |
|---|---|---|---|
| $F(T)$ for a *given* $T$ | **EXACT** | Felsenstein pruning, deterministic, double precision, single thread. No answer key exists — it is a function of the submission's own output. | The scorer. Contamination exposure: **zero**. |
| $\max_T F(T)$, $n\le 9$ | **EXACT (certified)** | Exhaustive enumeration of all $(2n-5)!!$ topologies with heavy branch-length optimisation, computed once per instance and cached in the manifest. | The **optimality gate**. Verifies not merely correctness but *optimality*. |
| $\max_T F(T)$, $n>9$ | **PROXY (best-known score, BKS)** | Portfolio: RAxML-NG 2.0.2 ×20 seeds default + ×20 seeds `--opt-topology classic --spr-radius 30`, IQ-TREE 3.1.3 ×20 seeds + ×20 with `-nstop 500 -allnni`, FastTree 2.2.0, plus every admissible arena submission ever scored on that instance. Monotone: BKS only improves, and the improver is credited. | Reference point for the gap metric. Explicitly a *lower bound* on the optimum, exactly as TSPLIB/DIMACS treat best-known tours. |
| the generating ("true") tree | **EXACT by construction** | Our seeded generator wrote it. | **Secondary, reported, never the objective.** |

**Why the true tree cannot be the objective.** The ML tree is a *statistically consistent but finite-sample* estimator of the generating tree. On short alignments or long branches it is systematically wrong, and Höhler et al.'s 20,000-MSA simulation study found RF distance to the true tree reaching ≥0.7 on high-difficulty simulated data *for all three of RAxML-NG, IQ-TREE 2 and FastTree 2*, with **no substantial differences between them** ([Bioinformatics Advances 5:vbaf300, 2025](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf300/8340590)). Scoring on RF would therefore reward statistical estimators, not search algorithms, and would be nearly flat across tools that differ enormously in search quality. RF is a *constraint and a report*, never the score. See "Accuracy metric".

**Why simulated over curated.** Per SPEC.md Finding 1, curated truth is the only class with unfixable contamination exposure. Every published alignment with a published ML tree (the Rokas/Zhou 19 phylogenomic datasets, TreeBASE, OrthoMaM) has its best-known trees in GitHub, in supplementary tables, and near-certainly in pretraining. Fresh-seed simulation makes memorisation structurally impossible: **the instance did not exist when the model was trained.** RAxMLGrove is used *only* to parameterise the simulator — empirical distributions of $\theta$, tree shape and branch-length scale — which is its stated design purpose, not as an answer key.

---

## Instance generation and tiers

### Environment (verified against the Anaconda API, 2026-09-01)

```bash
conda create -n arena-p09 -c conda-forge -c bioconda \
  raxml-ng=2.0.2 iqtree=3.1.3 fasttree=2.2.0 veryfasttree=4.0.5 \
  seq-gen=1.3.5 dendropy=5.0.8 python=3.11 numpy
conda activate arena-p09
conda install -c conda-forge pythiaphylopredictor=2.0.0   # PyPythia 2.0.0, noarch
```

All of `raxml-ng`, `iqtree`, `fasttree`, `veryfasttree`, `seq-gen` have **native `osx-arm64` builds on bioconda** (details under "Incumbents"). `dendropy` and `pythiaphylopredictor` are `noarch`.

### Step 1 — true tree (seeded, ours, never downloaded)

```python
# generate_tree.py  — pinned, hash-recorded
import random, dendropy
from dendropy.simulate import treesim

def true_tree(n, seed, diameter=0.6, clock_sd=0.6):
    rng = random.Random(seed)
    t = treesim.birth_death_tree(birth_rate=1.0, death_rate=0.35,
                                 num_extant_tips=n, rng=rng)          # ultrametric
    # relaxed clock: multiply each edge by an independent lognormal rate.
    # clock_sd is THE difficulty knob — it creates long/short branch heterogeneity,
    # which is what makes tree space rugged (Money & Whelan 2012).
    for e in t.preorder_edge_iter():
        if e.length is not None:
            e.length *= rng.lognormvariate(0.0, clock_sd)
    scale = diameter / max(l.distance_from_root() for l in t.leaf_node_iter())
    for e in t.preorder_edge_iter():
        if e.length is not None:
            e.length = min(max(e.length * scale, 1e-6), 100.0)
    t.is_rooted = False
    return t.as_string(schema="newick", suppress_rooting=True)
```

### Step 2 — alignment, primary simulator (AliSim, inside IQ-TREE 3.1.3)

$\theta$ is drawn per instance from RAxMLGrove empirical distributions ([github.com/angtft/RAxMLGrove](https://github.com/angtft/RAxMLGrove); Höhler et al., *Bioinformatics* 38:1741–1742, [doi:10.1093/bioinformatics/btac007](https://doi.org/10.1093/bioinformatics/btac007)), then **written verbatim into the model string so the searcher and the scorer use identical, fixed parameters**:

```bash
iqtree3 --alisim P09_n50_s${SEED} \
        -t true_tree_s${SEED}.nwk \
        -m "GTR{1.21/3.44/0.87/0.79/4.10}+F{0.28/0.22/0.23/0.27}+G4{0.55}" \
        --length 1000 --seed ${SEED} --out-format fasta -redo
```

AliSim: Ly-Trong, Naser-Khdour, Lanfear & Minh, *MBE* 39:msac092 (2022), [doi:10.1093/molbev/msac092](https://doi.org/10.1093/molbev/msac092).

### Step 3 — independent simulator for the transfer tier (Seq-Gen 1.3.5)

Guards against the simulation-derived failure mode named in SPEC.md Finding 1: *fitting the simulator*. Same true tree, same $\theta$, different codebase and different RNG.

```bash
seq-gen -mGTR -r1.21,3.44,0.87,0.79,4.10 -f0.28,0.22,0.23,0.27 \
        -a0.55 -g4 -l1000 -n1 -z${SEED} -of \
        < true_tree_s${SEED}.nwk > P09_n50_s${SEED}_seqgen.fasta
```

(Seq-Gen options verified against the [official manual](https://github.com/rambaut/Seq-Gen/blob/master/documentation/Seq-Gen.Manual.html): `-m` model, `-r` GTR rates, `-f` frequencies, `-a` Γ shape, `-g` discrete categories, `-l` length, `-n` replicates, `-z` seed, `-of` FASTA, tree on stdin. Rambaut & Grassly, *CABIOS* 13:235–238, 1997.)

### Step 4 — difficulty screening (the step that decides whether the problem is saturated)

```bash
pythia -m P09_n50_s${SEED}.fasta -r $(which raxml-ng) -o difficulty.txt
```

**Instances are admitted only if Pythia difficulty ∈ [0.4, 0.8]; otherwise regenerate with a new seed and a higher `clock_sd`.** This is not cosmetic. Liu et al. 2024 showed directly that "gene alignments with lower difficulty scores had a higher chance of finding the best-of-100 gene tree topology" ([*Syst. Biol.* 73:807–822](https://doi.org/10.1093/sysbio/syae031)) — easy instances are *solved*, and a benchmark built on them measures nothing. The existing P09 scaffold already discovered this empirically: its 16-taxon × 1000-site setting reached RF = 0 and came within 0.018 log-units of the true tree, i.e. **saturated**, while its 16 × 200, mean-branch-0.30 setting left an 11.74-log-unit deficit (`problem.yaml`, `headroom_evidence`). Pythia turns that ad-hoc observation into a screening rule.

Difficulty ≥ 0.8 is excluded too: there the tree is not identifiable from the data and lnL differences stop corresponding to anything, which is Höhler et al.'s "all tools perform poorly" regime.

### Tiers

| Tier | $n$ | $m$ | Truth | Budget (Leaderboard A) | Where |
|---|---|---|---|---|---|
| **G8 — certified gate** | 8 | 1,000 | **EXACT**, all 10,395 topologies enumerated | 2×10⁸ CLV-site-ops | laptop, ~3 s/instance |
| **G9 — hard certified** | 9 | 1,500 | **EXACT**, all 135,135 topologies | 1×10⁹ ops | laptop; certification offline |
| **T1 — small (dev loop)** | 50 | 1,000 | BKS | 1×10¹⁰ ops ≈ 60 s | **MacBook. Mandatory.** |
| **T2 — medium** | 200 | 4,000 | BKS | 5×10¹¹ ops ≈ 20 min × 4 cores | laptop (slow) or HPC |
| **T3 — large** | 1,000 | 10,000 | BKS | 2×10¹³ ops ≈ 4 h × 32 cores | HPC (2× EPYC 7763, 128 cores, 256 GB/node) |
| **X1 — transfer** | 50 | 1,000 | BKS | as T1 | Seq-Gen instances, sealed |
| **X2 — misspecified** | 50 | 1,000 | BKS | as T1 | simulate under a 2-class heterotachy/GHOST mixture, *infer under GTR+Γ4* |

**Scaling ladder** (SPEC.md §II.4 requires ≥7 log-spaced points over ≥2.5 decades). Two ladders, fitted independently:

- taxa: $n \in \{8, 16, 32, 64, 128, 256, 512, 1024, 2048\}$ at $m = 2{,}000$ — 9 points, 2.41 decades.
- sites: $m \in \{100, 200, 400, 800, 1600, 3200, 6400, 12800, 32000\}$ at $n = 100$ — 9 points, 2.51 decades.

Per SPEC.md, extend downward if the top is unaffordable: $n=2048, m=2000$ costs ~30 min/replicate on one HPC node and is the only ladder point that needs the cluster.

**Instance count.** Dev set: 12 instances × 1 seed (fresh each generation, seeds derived from a generation counter so they are never reused). Sealed milestone set: 100 instances × 20 seeds, generated from a seed stream never exposed to the search loop, run once per milestone.

**No downloads anywhere in the scored path.** Nothing to memorise, nothing to go stale, nothing that needs a data-access agreement.

---

## Correctness gate (binary, applied before any scoring)

All eight must pass. Failure scores zero — no partial credit for a fast wrong answer.

1. **Parses.** Output is a single valid Newick tree.
2. **Bijection.** Leaf label multiset equals the alignment's taxon set exactly. No missing, extra or duplicated labels.
3. **Binary and unrooted.** Every internal node has degree 3; exactly $n-2$ internal nodes and $2n-3$ edges. No polytomies (a polytomy is a way to abstain from choosing, and would let a candidate dodge the hard part of the problem).
4. **Branch lengths admissible.** If supplied: all finite, non-NaN, in $[10^{-6}, 100]$. Values outside are clipped, and the clipping is logged — a submission that relies on clipping is flagged.
5. **Self-report consistency.** If the submission reports a log-likelihood, `|reported − harness-recomputed| < 10⁻⁶`. A mismatch is treated as fraud, not as a bug.
6. **Optimality on the certified gate.** On all 25 fresh G8 instances and 10 fresh G9 instances the returned topology must satisfy $F(\hat T) \ge F^\star - \varepsilon_{\text{gate}}$ with $\varepsilon_{\text{gate}} = 0.01$ log-units, where $F^\star$ is the exhaustively certified optimum. **This is the strongest gate in the arena: elsewhere we verify correctness, here we verify optimality.** A heuristic that silently returns local optima cannot pass.
7. **Determinism.** Two runs with the same seed on the same instance return byte-identical Newick (after canonical ordering). Non-deterministic submissions must declare it and are then run at 20 seeds with the worst seed reported (SPEC.md §II.1).
8. **No side channels.** The submission wrote no files, opened no sockets, spawned no subprocesses, and did not import the manifest, the true tree, or the BKS registry. Enforced by a restricted import list plus a filesystem/network sandbox. Import-time wall-clock is measured separately and must be < 0.5 s (this closes the "precompute during import" hole).

---

## Accuracy metric

### Primary — log-likelihood gap, in native log-units

$$\mathrm{gap}(\hat T) = F^{\mathrm{ref}} - F(\hat T), \qquad F^{\mathrm{ref}} = \begin{cases} F^\star & \text{certified tiers}\\ \mathrm{BKS} & \text{otherwise}\end{cases}$$

Do **not** normalise by $|F|$. The absolute log-unit is the field's own unit: statistical topology tests (KH, SH, AU) operate on absolute log-likelihood differences, so a gap of 2.0 has a fixed, interpretable meaning at every $n$ and $m$, whereas a relative gap does not.

**How $F(\hat T)$ is computed — the canonical scorer.** This is the most load-bearing paragraph in the spec.

```bash
raxml-ng --evaluate --msa aln.fasta --tree candidate.nwk \
         --model "GTR{...}+FU{...}+G4m{...}" \
         --opt-model off --opt-branches on \
         --threads 1 --precision 9 --seed 1 --prefix score_cand
```

`--opt-model off` freezes $\theta$ at the simulation truth, so only branch lengths are optimised — the profile likelihood of the definition (option semantics verified on the [RAxML-NG tuning wiki](https://github.com/amkozlov/raxml-ng/wiki/Tuning-model-optimization-and-tree-search)). Three hardening rules:

- **Multi-start branch-length optimisation.** Run from 8 initialisations — the submitted branch lengths (if any), all-0.1 (`RAXML_BRLEN_DEFAULT`), the parsimony-derived lengths, and 5 log-uniform random draws — and take the maximum. This makes the score *monotone in effort by the scorer, not by the submission*, so a candidate cannot gain by handing the scorer adversarially favourable starting branch lengths, and cannot lose by handing it none.
- **Scorer self-consistency $\varepsilon_{\text{scorer}}$ must be measured and published per tier**: the spread of $F$ across those 8 restarts, over 200 random topologies. **The problem does not go live until $\varepsilon_{\text{scorer}} < 0.05$ log-units at every tier**; if it is not, increase `--brlen-smoothings` / tighten `--lh-epsilon` until it is. Every win threshold below is defined as a multiple of this measured number.
- **Single-threaded, `--precision 9`, fixed binary hash.** SIMD width and thread count change floating-point summation order and therefore the last digits of $\ell$. Every candidate is scored by the identical pinned binary on the identical pinned machine.

**Second opinion.** 10% of scored trees are re-scored with IQ-TREE 3.1.3 (`iqtree3 -s aln.fasta -te candidate.nwk -m "<MODEL>" -blmin 1e-6 -blmax 100 -T 1 -seed 1 -pre score_iq`). Disagreement > 0.1 log-units between the two independent engines halts scoring and triggers an audit. Two independently written likelihood implementations agreeing is the closest thing this problem has to a proof.

### Statistical materiality — is the difference real?

A raw gap is not automatically a scientific claim. Run the AU test over {incumbent tree, candidate tree, BKS tree}:

```bash
cat incumbent.nwk candidate.nwk bks.nwk > trees.nwk
iqtree3 -s aln.fasta -m "<MODEL>" -z trees.nwk -n 0 -zb 10000 -au -T 1 -pre au_test
```

(AU test: Shimodaira, *Syst. Biol.* 51:492–508, 2002; RELL: Kishino, Miyata & Hasegawa 1990. Command syntax verified in the [IQ-TREE Command Reference](https://iqtree.github.io/doc/Command-Reference).) Report the fraction of instances where the candidate tree is *not rejected* and the incumbent tree *is* — this is the honest phylogenetic reading of a win.

### Secondary — topological accuracy (reported and constrained, never the objective)

$$\mathrm{nRF}(\hat T, T_{\text{true}}) = \frac{\mathrm{RF}(\hat T, T_{\text{true}})}{2n-6} \in [0,1]$$

Computed with the pinned tool, not a hand-rolled function:

```bash
cat candidate.nwk true_tree.nwk > pair.nwk
raxml-ng --rfdist --tree pair.nwk --prefix rf   # reports absolute and relative RF
```

Also report $\mathrm{nRF}(\hat T, T_{\text{incumbent}})$ and, on Linux tiers only, the quartet distance via `tqdist` (bioconda `tqdist=1.0.0` is **linux-64 only** — no macOS build).

**The RF constraint, and why it is a constraint.** RF is *not* an admissibility gate, because a genuinely better ML tree that is further from truth is a real and publishable finding, not a failure. But it must not be silently buried. Rule:

> A candidate whose median $\mathrm{gap}$ improves by ≥ 2.0 log-units **and** whose median nRF-to-truth degrades by more than 0.02 is **admitted but flagged `LNL_TRUTH_DIVERGENCE`**, and the pair (Δgap, ΔnRF) is published on the leaderboard as a two-column result. Such a candidate may not be described as "more accurate" — only as "a better maximiser".

This is the honest resolution of question (d). The alternative — blending lnL and RF into one number — would silently trade off two quantities that measure different things and would make the score gameable by whichever component is noisier.

---

## Resource metrics

There is no persistent index in this problem, so the space axis is **peak conditional-likelihood-vector memory** — which is a real, published optimisation target, not a bookkeeping artifact (Izquierdo-Carrasco, Smith & Stamatakis, *BMC Bioinformatics* 12:470, 2011, [doi:10.1186/1471-2105-12-470](https://doi.org/10.1186/1471-2105-12-470), on memory/recomputation trade-offs for CLVs).

| Axis | Definition | How measured |
|---|---|---|
| **Search cost (A)** | CLV-site-ops (below) | counter inside the immutable engine |
| **Search cost (B)** | wall-clock to termination | interleaved `I C I C …`, randomized start arm, r = 5 (11 if IQR > 3% of median), decide on the median with a bootstrap CI on the ratio |
| **Peak RAM** | cgroup v2 `memory.peak` on Linux; `task_info`/`rusage_info` `ri_lifetime_max_phys_footprint` on macOS. r = 3, take the **max** | never `/usr/bin/time -v` (mishandles children and mmap) |
| **CLV memory** | peak simultaneously-live CLV bytes, exactly $\;\text{live\_clvs}\times n_{\text{patterns}}\times|\Sigma|\times k\times 8$ | metered by the engine's allocator; **theoretical floor for a full-tree evaluation is $(n-2)$ live CLVs; anything below that implies recomputation, which the ops counter will show** |
| **Output size** | Newick bytes | trivially small; reported for completeness, never a differentiator here |
| **Model/state size** | bytes of any learned artifact (weights, tables) a submission carries | **first-class.** A learned move-ordering policy that needs 400 MB of weights is a different scientific object from a 40-line heuristic and must be disclosed |

There is no build/query split. The analogue is **preprocessing** (site-pattern compression, ML distance matrix, parsimony scoring, starting-tree construction) versus **search** (topology moves). They are metered separately, because a method that spends 90% of its budget on a smarter starting tree — which is exactly what Treeline does, see below — is a different claim from one that spends it on smarter moves.

Machine fingerprint is recorded with every number (SPEC.md §II.3): CPU model, core count, whether SMT and turbo are off, governor, thread pinning verified from the cgroup cpu/wall ratio, and — a real hazard on this laptop — **whether Python is running natively or under Rosetta.** The existing `results.jsonl` in the P09 scaffold records `"machine": "x86_64"` on an `arm64` host; that run was translated, and its timings are not comparable to a native run. A number without its fingerprint is treated as unreported.

---

## Implementation-independent counters

**Primary: CLV-site-ops.**

$$\text{ops} \;=\; \sum_{\text{CLV recomputations}} n_{\text{patterns}} \times |\Sigma|^2 \times k$$

One "op" is one site-pattern × one rate category × one $4\times4$ transition-matrix application producing one child's contribution to one partial vector. This is the quantity that dominates every ML phylogenetics implementation regardless of language, vectorisation or hardware, and it is exactly what libpll/coraxlib's inner kernel executes. It is incremented **inside** `LikelihoodEngine`, which the submission cannot modify and cannot bypass because it never holds the alignment.

The existing scaffold's `phylo.CLV_UPDATES` counter is the right idea at the wrong granularity — it counts *node* updates, which conflates a 200-site instance with a 20,000-site one. Weighting by $n_{\text{patterns}}\times|\Sigma|^2\times k$ fixes that and makes the counter comparable across the whole ladder.

**Secondary counters, all reported:**

| Counter | What it isolates |
|---|---|
| distinct topologies evaluated (structural hash) | breadth of the search |
| full-tree vs incremental evaluations | whether the candidate exploits locality — the single largest engineering lever in RAxML-NG (`--extra fastclv-off` exists precisely to turn this off) |
| Newton–Raphson branch-length iterations | the branch-length optimisation schedule stage |
| transition-matrix ($P(t) = e^{Qt}$) computations | cost hidden by CLV counting alone; matters when many short branches are re-optimised |
| parsimony-score evaluations | cost of parsimony-guided move ordering, the cheap surrogate everyone uses |
| peak live CLVs | the memory/recomputation trade-off |

**Counter-avoidance guard (SPEC.md §II.2).** "Fewer CLV ops" is trivially achieved by not searching. Counters are meaningful only behind the gap gate. Any candidate whose primary counter drops > 10× while the median gap stays flat is **automatically routed to human review** — that is either a genuine breakthrough or a leak, and the point of the flag is to find out which. Budget for this: published harness work finds stronger models hack more.

**Engineering headroom constant.** Before the problem goes live, implement RAxML-NG's own default algorithm inside the scaffold both naively and carefully, and publish the ratio. Any Leaderboard-B win smaller than that constant is not attributable to the algorithm and must be labelled so. Likewise **scaffold fidelity**: the in-scaffold reference must reach within 1.5× of real `raxml-ng` on time-to-target before any score is published. The existing `problem.yaml` correctly marks the whole problem `MEASURED_BUT_BLOCKED_ON_SCAFFOLD_FIDELITY` for exactly this reason, and that blocker stands.

---

## Incumbent(s) to beat

All four verified present as **native `osx-arm64` conda packages** (queried against the Anaconda API on 2026-09-01):

| Tool | Version | osx-arm64? | Build uploaded |
|---|---|---|---|
| **RAxML-NG** | **2.0.2** | ✅ `raxml-ng-2.0.2-h0f3a4f4_0.conda` | 2026-06-01 |
| **IQ-TREE** | **3.1.3** | ✅ `iqtree-3.1.3-h6cc7423_0.conda` | 2026-07-26 |
| **FastTree** | **2.2.0** | ✅ `fasttree-2.2.0-hd34ca47_1.conda` | 2025-11-25 |
| **VeryFastTree** | **4.0.5** | ✅ | — |
| seq-gen | 1.3.5 | ✅ | — |
| tqDist | 1.0.0 | ❌ **linux-64 only** | — |

Upstream release dates (GitHub API): RAxML-NG 2.0.0 on 2026-03-11, 2.0.1 on 2026-04-24, 2.0.2 on 2026-05-29; IQ-TREE 3.0.0 on 2025-04-03, 3.1.3 on 2026-06-19. Both v2/v3 lines are current, not stale.

### The mutable components, named precisely (question (a))

**RAxML-NG 2.0.x** — SPR-only hill climbing, staged. Verified from the [tuning wiki](https://github.com/amkozlov/raxml-ng/wiki/Tuning-model-optimization-and-tree-search), [`src/Options.cpp`](https://raw.githubusercontent.com/amkozlov/raxml-ng/master/src/Options.cpp), [`src/constants.hpp`](https://raw.githubusercontent.com/amkozlov/raxml-ng/master/src/constants.hpp), and Togkousidis, Kozlov, Stamatakis, *MBE* 40:msad227 (2023), [doi:10.1093/molbev/msad227](https://doi.org/10.1093/molbev/msad227):

| Stage | RAxML-NG's choice |
|---|---|
| starting trees | 10 random + 10 maximum-parsimony (randomised stepwise addition), parsimony default since v1.2.0; optional parsimony-SPR polish |
| move set | **SPR only** (no NNI in the classic path; NNI rounds added in the adaptive path) |
| move ordering / pruning | descend into a subtree only if its likelihood clears `--spr-cutoff` (default **1.0**, `off` disables) |
| radius schedule | `--spr-radius` auto-detected; adaptive path sets it by a triangular function of Pythia difficulty, from 5 at difficulty 0 or 1 up to **30 at difficulty 0.5** |
| lazy vs full evaluation | three phases: **fast SPR rounds** (reuse existing branch lengths), **mixed**, **slow SPR rounds** (re-optimise the 3 branches adjacent to each insertion point, keep the top-20 topologies, then full branch-length optimisation on those 20) |
| branch-length optimisation | Newton–Raphson (`--blopt nr_fast`), `RAXML_BRLEN_SMOOTHINGS = 32`, `RAXML_BRLEN_DEFAULT = 0.1`, bounds $[10^{-6}, 100]$ |
| epsilons | `DEF_LH_EPSILON = 10`, `DEF_LH_EPSILON_BRLEN_TRIPLET = 1000`, `RAXML_PARAM_EPSILON = 0.001` (source-verified in both the `1.2.2` tag and `master`) |
| termination | v1.x: log-likelihood improvement only. v2.x: `--stop-rule {kh, kh-mult, sn-rell, sn-normal, off}` — a **statistical** stopping test, from Togkousidis, Stamatakis & Gascuel, *Syst. Biol.* syaf043 (2025), [doi:10.1093/sysbio/syaf043](https://doi.org/10.1093/sysbio/syaf043) |
| top-level strategy | `--opt-topology {adaptive (default in master), classic, simple (fixed radius 10), rbs, nni, off}`; `--fast` = simplified heuristic with early stopping, **~50× faster than v1.2** per the [v2.0.0 release notes](https://github.com/amkozlov/raxml-ng/releases/tag/2.0.0) |

**IQ-TREE 3.1.3** — NNI hill climbing inside a stochastic-perturbation metaheuristic. Defaults verified in the [Command Reference](https://iqtree.github.io/doc/Command-Reference) and Nguyen, Schmidt, von Haeseler & Minh, *MBE* 32:268–274 (2015), [doi:10.1093/molbev/msu300](https://doi.org/10.1093/molbev/msu300):

| Stage | IQ-TREE's choice |
|---|---|
| starting trees | `-ninit 100` parsimony trees + BIONJ; parsimony SPR radius `-sprrad 6` |
| candidate initialisation | `-ntop 20` best are NNI-optimised |
| population | `-nbest 5` maintained in the candidate set ℂ |
| move set | NNI, with **multiple non-conflicting NNIs applied simultaneously** per round |
| perturbation | random NNI kick of strength `-pers 0.5`, i.e. $0.5(n-3)$ random NNIs on a randomly chosen candidate |
| move ordering | by default only NNIs in the vicinity of previously applied NNIs; `-allnni` considers all |
| termination | `-nstop 100` unsuccessful perturbations |
| fast path | `-fast`: two starting trees (MP + BIONJ), NNI only |
| v3 addition | `--pathogen` dynamically switches to CMAPLE for low-divergence pandemic-scale data (Wong et al., *MBE* 43:msag117, 2026, [doi:10.1093/molbev/msag117](https://doi.org/10.1093/molbev/msag117)) |

**FastTree 2.2.0** — minimum-evolution first, likelihood second: $4\log_2 N$ ME-NNI rounds + 2 SPR rounds, then $2\log_2 N$ ML-NNI rounds ([morgannprice.github.io/fasttree](https://morgannprice.github.io/fasttree/)).

**Treeline** (Wright, *Mol. Ecol. Resour.* 26, 2026, [doi:10.1111/1755-0998.70163](https://doi.org/10.1111/1755-0998.70163), in DECIPHER/Bioconductor) — a **2026** entrant whose contribution is a new *starting-tree* strategy: perturbing the patristic distance matrix used to seed candidate trees. R/Bioconductor, not conda; treated as a reference point rather than a scored incumbent.

### Published numbers, with sources

| Finding | Number | Source |
|---|---|---|
| RAxML-NG found the best-scoring tree on 19/21 empirical datasets, 1.3–4.5× faster than competitors; "IQ-TREE results show much lower variance" | 19/21 | Kozlov et al., *Bioinformatics* 35:4453–4455 (2019), [doi:10.1093/bioinformatics/btz305](https://doi.org/10.1093/bioinformatics/btz305) |
| Best-likelihood frequency across thousands of single-gene alignments: IQ-TREE-10 **80.17%**, RAxML-10 **75.99%**, IQ-TREE 54.03%, PhyML-10 52.35%, RAxML 35.98%, PhyML 24.17%, FastTree **1.67%** | as listed | Zhou, Shen, Hittinger & Rokas, *MBE* 35:486–503 (2018), [doi:10.1093/molbev/msx302](https://doi.org/10.1093/molbev/msx302) |
| On 17 supermatrices IQ-TREE found the highest lnL on **all 17**, beating RAxML on 7 and published RAxML/ExaML results on 8 | 17/17 | same |
| Adaptive RAxML-NG: 3.11× (empirical) / 4.27× (simulated) accumulated speedup, ~94% of trees statistically indistinguishable, 99% within 0.1% relative lnL | as listed | Togkousidis et al., *MBE* 40:msad227 (2023) |
| KH-based early stopping: **5×** (DNA) / **3.9×** (protein) with parsimony starts, **10.2×** at RAxML-NG defaults; 98% (DNA) / 92–96% (AA) statistically equivalent | as listed | Togkousidis, Stamatakis & Gascuel, *Syst. Biol.* syaf043 (2025) |
| PhyloSmew, 20,000 simulated + ~5,000 TreeBASE MSAs: RAxML-NG best mean lnL at all difficulty levels; IQ-TREE 2 within **4 lnL units**; FastTree2 up to **6.3 units** worse; RF differences among the three "insubstantial" on simulated data | as listed | Höhler, Haag, Kozlov & Stamatakis, *Bioinformatics Advances* 5:vbaf300 (2025), [doi:10.1093/bioadv/vbaf300](https://doi.org/10.1093/bioadv/vbaf300) |
| Raising RAxML-NG's $\varepsilon_{\mathrm{LnL}}$ to 10 and $\varepsilon_{\mathrm{brlen}}$ to 10³ gives **1.9 ± 0.6×** speedup with no quality loss | as listed | Höhler et al., *Bioinformatics Advances* 3:vbad124 (2023), [doi:10.1093/bioadv/vbad124](https://doi.org/10.1093/bioadv/vbad124) |

---

## Where the real headroom is

**Verdict: the incumbents are demonstrably NOT near-optimal, the evidence is quantitative and recent, and the gap is largest exactly in the regime this spec targets.** This is question (b), and it is the section that decides whether P09 is worth running.

**1. A single default RAxML-NG or IQ-TREE search finds the best tree only about half the time.** Liu, Zhou, Li, Hittinger, Pan, Huang, Chen, Rokas, Chen & Shen ran **100 independent tree searches on each of 19,414 gene alignments** from 15 phylogenomic datasets:

> single search recovers the best-of-100 topology for **~51%** of alignments with IQ-TREE and **~42%** with RAxML-NG; **10 searches** raises this only to **69%** and **64%**.
> — *Syst. Biol.* 73:807–822 (2024), [doi:10.1093/sysbio/syae031](https://doi.org/10.1093/sysbio/syae031)

Read that again with the arena's eyes: **at the recommended operating point of 10 searches, roughly one third of real alignments end below the best likelihood that the same tool finds with more restarts.** That is not a subtle inefficiency — it is a search failure rate of ~30–35% on the field's default configuration. Their explicit recommendation (≥20 searches for IQ-TREE, ≥10 for RAxML-NG) is a restart-count patch, not an algorithmic fix. Data are public at [Dryad doi:10.5061/dryad.rv15dv4b7](https://datadryad.org/dataset/doi:10.5061/dryad.rv15dv4b7).

**2. Two runs of the *same tool* on the *same alignment* disagree, materially.** Shen, Li, Hittinger, Chen & Rokas ran identical replicate analyses (20 tree searches each) on the same 19,414 alignments:

> topologically different trees between Run1 and Run2 for **3,515/19,414 genes (18.1%)** with IQ-TREE and **1,813/19,414 (9.3%)** with RAxML-NG; mean lnL difference among irreproducible genes **1.098** (IQ-TREE) and **0.151** (RAxML-NG); mean nRFD between runs **0.28** and **0.34**.
> — *Nature Communications* 11:6096 (2020), [doi:10.1038/s41467-020-20005-6](https://doi.org/10.1038/s41467-020-20005-6)

Different seeds, same data, materially different final answers. This is the direct answer to question (b): the documented cases exist, at scale, in a top-tier journal, on real data.

**3. The tools disagree with each other by amounts that flip published conclusions.** Zhou et al. found IQ-TREE reaching higher likelihoods than RAxML/ExaML on **all 17** supermatrices, including beating *previously published* RAxML/ExaML trees on 8 of them — and, damningly, that the better trees "can be recovered by simply re-optimizing the RAxML-generated (or RAxML/ExaML-published) results using a function built in RAxML itself for four (or six) data sets" ([*MBE* 35:486–503](https://doi.org/10.1093/molbev/msx302)). **Published phylogenies were sitting on top of trivially reachable improvements that the search had left on the table.**

**4. Tree space is genuinely rugged, and the field has a validated measure of when.** Pythia's ground-truth difficulty label is built from exactly the disagreement structure above — average pairwise relative RF among 100 ML trees, count of unique topologies among them, and the fraction surviving statistical tests — and its authors state plainly that on hard MSAs "the independent tree inferences converge to multiple topologically distinct, yet statistically indistinguishable, locally optimal trees" (Haag & Stamatakis, *MBE* 39:msac254, 2022, [doi:10.1093/molbev/msac254](https://doi.org/10.1093/molbev/msac254); preprint [biorxiv 2022.06.20.496790](https://www.biorxiv.org/content/10.1101/2022.06.20.496790v2)). About **10% of empirical MSAs carry difficulty ≥ 0.7.** The theoretical grounding is older: Money & Whelan showed that attraction-basin structure — the count and size of local optima reachable by greedy hill climbing — tracks alignment information content ([*Syst. Biol.* 61:228–239, 2012](https://academic.oup.com/sysbio/article/61/2/228/1646300)).

**5. The optimal *effort schedule* is unsolved, and that is a second, orthogonal headroom axis.** Togkousidis et al. get 5–10.2× speedups purely by stopping when improvements stop being statistically significant, with 92–98% of results statistically equivalent (*Syst. Biol.* syaf043, 2025). Adaptive RAxML-NG gets 3–4× (12.6–16× on easy data) purely by conditioning the search schedule on predicted difficulty (*MBE* 40:msad227, 2023). The Free Lunch paper got 1.9× by *changing two constants* (*Bioinformatics Advances* 3:vbad124, 2023). **Three independent groups extracted multiplicative speedups from schedule and threshold choices alone, within the last three years, without changing the move set.** That is the signature of a design space that is nowhere near exhausted.

**6. The design space is still yielding brand-new ideas, this year.** Treeline (Wright, *Mol. Ecol. Resour.*, June 2026) introduces starting-tree generation by *perturbing the patristic distance matrix* — a component nobody had varied — and reports being "more accurate, memory efficient or faster than existing programmes designed for a single optimisation objective."

**7. Our own probe already reproduces the phenomenon at laptop scale.** The existing `difficulty_probe.py` found random starts finishing **17 to 714 log-units apart** on the same instance, and a **11.74-log-unit** NJ+NNI deficit at 16 taxa × 200 sites with mean branch length 0.30 — while the 16 × 1000 setting was saturated (RF 0, within 0.018 of the true tree). Ruggedness is reachable on a MacBook; you just have to aim at it. The Pythia screening band [0.4, 0.8] is the operationalisation.

### Where the headroom is *not*

Be honest about the negative space, because it is where a naive agent will waste its budget:

- **Easy instances are solved.** At Pythia ≤ 0.3 (≈60% of empirical MSAs) all tools converge to the same tree and every difference is noise. Excluded by construction.
- **"Random restarts beat hill climbing" is false here.** Our own probe found NJ+NNI beating the best of six random starts in 3 of 4 settings, and Liu et al. independently found "a random starting tree was not as effective as the BioNJ and parsimony starting trees." The starting tree dominates. An agent that proposes random multistart has proposed something already known to be worse.
- **Raw likelihood-kernel speed is a solved engineering problem.** libpll/coraxlib, IQ-TREE's kernels and VeryFastTree's vectorisation are heavily optimised. Leaderboard B against them is a hardware contest; Leaderboard A exists precisely so that is not what gets rewarded.
- **The single largest measured gap in this literature is not a new move set** — it is restart count and stopping schedule. An agent should be told this explicitly, because "invent TBR" is the obvious idea and the evidence says it is not where the money is.

**Standing blocker (carried forward from `problem.yaml`, unresolved).** All of the above is external evidence plus a probe against *our own* NNI hill-climber. Per SPEC.md §II.2, a win over an in-scaffold reference weaker than the real tool is an artifact. **P09 does not go live until the scaffold reference is benchmarked head-to-head against `raxml-ng 2.0.2` and `iqtree 3.1.3` on identical instances and the fidelity ratio is published.** If the scaffold reference is not within 1.5×, the headroom numbers must be recomputed against the real binaries before any score is claimed.

---

## Beat target

Question (e): **both, and neither alone.** "Final lnL at fixed budget" degenerates to noise once the incumbent has converged (which it has, on easy instances); "CPU to fixed threshold" is undefined when a method never reaches the threshold. The standard resolution in combinatorial-optimisation benchmarking is to use an *anytime* scalar plus a *censored* time metric, and that is what this spec adopts.

Let $g(t) = F^{\mathrm{ref}} - F(\hat T_t)$ be the gap of the incumbent-best tree at cost $t \in [0, B]$.

**Headline metric — primal integral** (Berthold's measure, standard in MIP benchmarking):

$$\mathrm{PI} = \frac{1}{B}\int_0^B \min\bigl(g(t),\, g_{\max}\bigr)\, dt, \qquad g_{\max}=100 \text{ log-units}$$

It rewards reaching a good tree *early* and *staying* there, it is finite whether or not the target is ever reached, and it cannot be gamed by a lucky final-second jump.

**A win requires ALL of the following, pre-declared, on the sealed 100-instance × 20-seed set:**

| # | Condition |
|---|---|
| **W0** | Passes the correctness gate, including 100% certified optimality on G8 and G9 |
| **W1** | Median PI improvement over `raxml-ng 2.0.2` (default `--search`, matched budget) with the **lower bound of a 95% paired block-bootstrap CI (10,000 resamples, blocked by instance) strictly above 0**; point estimate ≥ 5% |
| **W2** | **Time-to-target**, target = $F^{\mathrm{ref}} - 0.1$ log-units: **PAR-10 geometric-mean speedup ≥ 1.5×** with the entire 95% CI above 1.0 (PAR-10: unsolved instances charged 10 × budget, as in the SAT competitions). Reported alongside the raw solved-count |
| **W3** | **Hard-instance quality**: on the Pythia ∈ [0.6, 0.8] subset, median final gap improves by **≥ 2.0 log-units**, with the 95% CI lower bound above **+0.5** |
| **W4** | **Worst seed**: over 20 seeds, the worst-seed final gap is not worse than the incumbent's worst-seed gap by more than 2.0 log-units. A method that is excellent on average and catastrophic 1-in-20 is not a phylogenetics tool |
| **W5** | **Non-inferior memory**: peak RSS ≤ incumbent's (κ = 1.0), and peak CLV bytes reported |
| **W6** | **Transfer**: W1 and W2 still hold on X1 (Seq-Gen) and X2 (misspecified model), which the search loop never sees. A win that evaporates on a different simulator was a win over AliSim, not over the problem |
| **W7** | **Novelty**: a literature check confirming the mechanism is not RAxML-NG's, IQ-TREE's, FastTree's, Treeline's, or an already-published variant. Per SPEC.md, this is a scored gate, not an afterthought — all five tools' source is on GitHub and near-certainly in pretraining |

**Why 2.0 log-units for W3.** It must clear three floors simultaneously: (i) the measured scorer noise $\varepsilon_{\text{scorer}} < 0.05$, by ≥ 40×; (ii) the observed run-to-run irreproducibility of the tools themselves — mean 1.098 log-units for IQ-TREE (Shen et al. 2020) — so the claim is not merely "we got a luckier seed"; (iii) the range within which AU/KH tests typically cannot separate topologies. A 2.0-unit median improvement across 100 instances is a real, defensible difference. **Anything below 0.5 log-units is declared not a claim, regardless of p-value.**

**Additional, separately-published win types** (each is a distinct scientific object and is labelled as such):

- **Scaling win (W-S)**: over the ≥9-point ladder, the fitted exponent of CLV-site-ops vs $n$ at fixed target quality is lower, with a bootstrap CI excluding the incumbent's. Fit the deterministic counter first, wall clock second. This is the most valuable outcome — it changes the curve, not a point.
- **Memory win (W-M, Track M)**: equal quality and time at strictly lower peak CLV bytes.
- **Certified-optimality win (W-G)**: 100% of G8/G9 instances solved to certified optimality at ≥ 2× lower ops than the incumbent. This is the only win in the arena that is a statement about *provable* optimality.

**Anti-win.** If after a full milestone cycle no candidate clears W1–W6, that is published as "P09 saturated at tier T1 against RAxML-NG 2.0.2 under budget B", with the distribution of near-misses. A clean negative on a well-built evaluator is a result.

---

## Mutable surface

The point of decomposition is that an agent replaces **one** stage and keeps the rest of the pipeline fixed, so a win is attributable. Each stage is a separate `EVOLVE-BLOCK` with its own reference implementation and its own ablation slot.

| Stage | Interface | What the incumbents do | What is open |
|---|---|---|---|
| **S1 · distance estimation** | `dists(engine) -> ndarray` | ML pairwise distances; BIONJ | corrected/shrunk distances; **patristic-matrix perturbation à la Treeline (2026)**; learned distance correction |
| **S2 · starting-tree construction** | `start_trees(engine, k) -> list[Topology]` | RAxML-NG: 10 random + 10 randomised-stepwise-addition parsimony. IQ-TREE: 100 parsimony + BIONJ. | **the highest-leverage stage.** How many starts, how diverse, how to *choose* $k$ from cheap data features, how to seed a diverse *portfolio* rather than i.i.d. draws |
| **S3 · move neighbourhood** | `neighbours(topo, state) -> Iterator[Move]` | RAxML-NG: SPR only. IQ-TREE: NNI (+ parsimony SPR radius 6 for starts). FastTree: NNI + SPR. | TBR; ratchet-style reweighting; sectorial/subtree-constrained search; hybrid schedules; moves defined on splits rather than edges |
| **S4 · move ordering & pruning** | `rank(moves, cheap_features) -> ordering` | `--spr-cutoff 1.0` subtree descent cutoff; IQ-TREE's "vicinity of previous NNIs" | parsimony- or distance-based surrogates; learned rankers (Azouri et al., *Nat. Commun.* 12:1983, 2021, [doi:10.1038/s41467-021-22073-8](https://doi.org/10.1038/s41467-021-22073-8), predicted good SPR moves from 19 cheap features **without computing the likelihood**); bandit allocation over subtrees |
| **S5 · radius / effort schedule** | `radius(iter, difficulty, history) -> int` | auto-detect; adaptive triangular 5→30→5 in difficulty | any function of search history; per-subtree adaptive radii |
| **S6 · evaluation fidelity** | `score(move) -> float` at chosen fidelity | fast SPR (reuse blens) → mixed → slow SPR (3-branch reopt, top-20, then full BLO) | **multi-fidelity search**: cheap surrogate → medium → exact, with a learned or bandit-driven promotion rule. The stage where CLV-op savings are largest |
| **S7 · branch-length optimisation schedule** | `opt_blens(topo, blens, level)` | Newton–Raphson, 32 smoothings, $\varepsilon_{\mathrm{LnL}}=10$, $\varepsilon_{\text{triplet}}=10^3$ | which branches, how often, to what tolerance; warm-start policies; a *schedule* of tolerances over the run |
| **S8 · acceptance / metaheuristic** | `accept(cand, cur, T, hist) -> bool` | strict hill climbing; IQ-TREE perturbs a candidate-set member with $0.5(n-3)$ random NNIs | simulated annealing, tabu, GRASP, path-relinking between candidates, population crossover on splits |
| **S9 · termination** | `stop(hist, budget) -> bool` | v1: lnL improvement. v2: `--stop-rule {kh, kh-mult, sn-rell, sn-normal}` | any anytime-optimal stopping rule; the primal-integral objective *rewards* stopping well, unlike a fixed-budget objective |
| **S10 · restart / portfolio policy** | `allocate(budget, starts) -> schedule` | fixed count of independent starts | **the stage the published evidence most implicates.** Adaptive restart counts, early abandonment of unpromising starts, racing/successive-halving over starts |

**Immutable, and enforced:** `LikelihoodEngine` and every counter inside it; the model parameters; the scorer; the instance generator; the manifest; the gate.

---

## Gameability risks

| Risk | Mechanism | Mitigation |
|---|---|---|
| Report a good number without a good tree | submission returns its own lnL | Harness **always** recomputes $F$ from the returned Newick with the pinned scorer. Self-reported values are checked to $10^{-6}$; a mismatch is fraud, not a bug |
| Score under a different (easier) model | e.g. drop +Γ4 | $\theta$ is fixed by the harness and passed to the scorer; the submission cannot alter the model object |
| Degenerate branch lengths inflating lnL numerically | zero-length or huge branches | Clipped to $[10^{-6},100]$, then re-optimised by the scorer's 8-restart protocol. Since the scorer re-optimises anyway, adversarial branch lengths cannot help |
| Adversarial branch-length warm start steering the scorer's optimiser | hand the scorer a start that lands in a favourable local BL optimum | Score = max over 8 pinned restarts *including* the submitted one — monotone, identical treatment for all, never worse than a neutral start |
| Precompute during import | heavy work in module scope | Import time measured separately, capped at 0.5 s; the instance is delivered only after the clock starts and its seed is fresh |
| Uncounted private likelihood engine | reimplement pruning on the raw alignment | **The alignment is not exposed.** `site_patterns()` is the only raw accessor and is charged the price of a full-tree likelihood evaluation. This is the structural fix, not a policy |
| Counter avoidance | return fast with a bad tree | Counters are meaningful only behind the gap gate; > 10× counter drop with flat gap → mandatory human review |
| Overfitting the instance generator | tune to AliSim's RNG or to the birth–death shape | Sealed X1 (Seq-Gen, independent codebase) and X2 (misspecified generating model) tiers; W6 requires the win to survive both |
| Overfitting the *difficulty band* | exploit that all instances sit in [0.4, 0.8] | 10% of the sealed set is drawn from outside the band and must not regress |
| Memorising public trees | recall the ML tree for a known alignment | Structurally impossible: fresh seeds, no public alignment in the scored path |
| Reimplementing a published schedule and calling it novel | RAxML-NG's SPR staging, IQ-TREE's perturbation, Treeline's matrix perturbation are all on GitHub and in the weights | **W7 novelty gate.** Per SPEC.md Finding 1, this is the arena-wide threat and is scored, not assumed |
| Terrace artifacts | on partitioned/missing data many topologies share identical lnL (Sanderson et al., *Syst. Biol.* 60:41–51, 2011) | Scored tiers are single-partition with **no missing data**. If partitions are ever introduced, terrace size must be reported via `raxml-ng --terrace` |
| Unidentifiable topology from duplicate sequences | identical rows make some splits free | Generator rejects instances containing duplicate sequences (RAxML-NG errors on them by default) |
| Floating-point non-determinism masquerading as a win | SIMD width / thread count change summation order | Scorer is single-threaded, `--precision 9`, one pinned binary hash, one pinned machine; 10% cross-checked against IQ-TREE's independent engine |
| Rosetta / architecture drift in timings | translated Python on Apple Silicon | Machine fingerprint recorded per run, including native-vs-translated. The existing `results.jsonl` shows `x86_64` on an `arm64` host — those timings are not comparable and are marked unreported |
| Thermal drift on a laptop | long A-then-B runs | Interleaved `I C I C I C …` with randomized starting arm, mandatory |

---

## Evaluation cost per candidate at the small tier

Measured on this machine (MacBook, Apple Silicon, Python 3.8.8 running **translated**, numpy JC69 implementation in `phylo.py`) on 2026-09-01:

| $n$ | $m$ | one full `loglik` | 2-pass golden-section branch-length optimisation |
|---|---|---|---|
| 6 | 400 | 1.62 ms | 0.199 s |
| 8 | 1,000 | 1.33 ms | 0.331 s |
| 10 | 1,000 | 0.94 ms | 0.464 s |
| 16 | 1,000 | 1.72 ms | 1.260 s |
| 24 | 1,000 | 2.42 ms | 2.922 s |

Topology counts: $n=8 \to 10{,}395$; $n=9 \to 135{,}135$; $n=10 \to 2{,}027{,}025$.

**Certification (one-time, amortised, cached in the manifest).** G8 with the current Python optimiser: $10{,}395 \times 0.33\,\text{s} \approx 57$ min per instance. G9: $135{,}135 \times 0.4\,\text{s} \approx 15$ h — an HPC batch job, or minutes with a C engine. **These are paid once per instance, ever.** Certified manifests are committed and reused, so the marginal per-candidate cost of the strongest gate in the arena is the cost of 35 short searches, not 35 enumerations.

**Per-candidate cost, T1 dev loop (12 instances × 1 seed × 60 s budget):**

| Item | Cost |
|---|---|
| 12 searches at the T1 budget | 12 min single-core; **~2 min on 8 cores** |
| G8 gate, 25 cached instances × 3 s | 75 s single-core; ~10 s on 8 cores |
| G9 gate, 10 cached instances | ~30 s on 8 cores |
| scoring (`raxml-ng --evaluate`, 8 restarts × 47 trees) | ~30 s |
| RF + AU test | ~20 s |
| **Total per candidate** | **≈ 3–4 minutes on an 8-core MacBook** |

That supports roughly **15–20 candidates per hour** locally — a workable FunSearch inner loop, though 5–10× slower than P01's. If that proves too slow to sustain an evolutionary population, shrink the dev set to 8 instances at a 30 s budget (≈ 90 s/candidate) and rely on the sealed set for the real signal.

**Milestone cost, sealed set (100 instances × 20 seeds × 60 s = 33 CPU-hours per tier):** one HPC node for ~20 minutes, or ~4 h wall on the laptop. Runs **once per milestone**, never per generation, per the Seal Protocol.

**T3 (1,000 taxa × 10,000 sites):** ~4 h × 32 cores per replicate. HPC only.

---

## Readiness scores

| Axis | Score | Rationale |
|---|---|---|
| **Evaluator fidelity** | **5 / 5** | The score is the exact objective, computed by Felsenstein pruning from the submission's own output — there is no answer key to leak, and at $n \le 9$ we certify the *provable global optimum* by exhaustive enumeration, so we verify optimality and not merely correctness. Uniquely strong in the arena; contingent on publishing $\varepsilon_{\text{scorer}} < 0.05$ before going live. |
| **Evaluator cost** | **3 / 5** | ~3–4 min per candidate at T1 on an 8-core laptop — usable but 5–10× costlier than P01, and the certified-gate enumerations cost 1–15 h each (one-time, cached). Meets the SPEC.md ≥ 3 bar; does not clear it comfortably. |
| **Headroom** | **5 / 5** | Quantified and recent: a single default search finds the best-of-100 tree only ~51%/~42% of the time and 10 searches only reach 69%/64% (Liu et al. 2024); the same tool with different seeds returns different topologies on 18.1%/9.3% of 19,414 real alignments (Shen et al. 2020); published phylogenies were improvable by RAxML's own reoptimisation (Zhou et al. 2018); three groups extracted 1.9×–10.2× from schedules and thresholds alone in three years; a genuinely new starting-tree idea published in 2026. |
| **Infra fit** | **4 / 5** | All four incumbents plus both simulators have native `osx-arm64` conda builds (verified). T1 runs on the laptop; T3 fits one HPC node; no controlled-access data anywhere. Docked one point: `tqdist` is linux-64 only, PyPythia adds a conda-forge dependency, and the current scaffold runs translated rather than native on this host. |
| **Learning payoff** | **5 / 5** | Building the evaluator forces mastery of Felsenstein pruning, CLV scaling and numerical underflow, Newton–Raphson branch-length optimisation, tree-space topology and RF/quartet metrics, the RELL/KH/SH/AU testing framework, and anytime-optimisation benchmarking (primal integral, PAR-10). |
| **Science payoff** | **4 / 5** | A real improvement lands in a tool used in tens of thousands of papers per year, and the memory/schedule axes transfer directly to our own phylogenetics-adjacent pipelines. Not directly a statistical-genetics result, hence 4. |

**Overall: RUN IT — with the standing blocker.** Fidelity 5 and headroom 5 clear the SPEC.md bar by a wide margin. **Do not publish any score until the scaffold-fidelity ratio against real `raxml-ng 2.0.2` and `iqtree 3.1.3` is measured and published**, and until $\varepsilon_{\text{scorer}}$ is measured at every tier.

---

## Confidence notes

**Verified by direct query or fetch (2026-09-01):**

- Conda availability and exact build filenames on `osx-arm64` via the Anaconda API: `raxml-ng` 2.0.2 (`raxml-ng-2.0.2-h0f3a4f4_0.conda`, uploaded 2026-06-01), `iqtree` 3.1.3 (uploaded 2026-07-26), `fasttree` 2.2.0, `veryfasttree` 4.0.5, `seq-gen` 1.3.5; `dendropy` 5.0.8 noarch (bioconda), `pyvolve` 1.1.0 noarch, `pythiaphylopredictor` 2.0.0 noarch (conda-forge); **`tqdist` 1.0.0 is linux-64 only**; `ete3` is not on bioconda `osx-arm64` (conda-forge `ete3` 3.1.2 is noarch).
- Upstream release tags and dates via the GitHub API: RAxML-NG 2.0.0 / 2.0.1 / 2.0.2 on 2026-03-11 / 04-24 / 05-29; 2.0-beta2 2025-05-12, 2.0-beta3 2025-10-31. IQ-TREE 3.0.0 2025-04-03 … 3.1.3 2026-06-19.
- RAxML-NG source constants fetched from both the `1.2.2` tag and `master`: `DEF_LH_EPSILON = 10`, `DEF_LH_EPSILON_BRLEN_TRIPLET = 1000`, `RAXML_BRLEN_MIN = 1.0e-6`, `RAXML_BRLEN_MAX = 100.`, `RAXML_BRLEN_SMOOTHINGS = 32`, `RAXML_BRLEN_DEFAULT = 0.1`, `RAXML_PARAM_EPSILON = 0.001`; `Options.cpp` shows `spr_cutoff(1.0)`, `spr_radius(-1)`, `brlen_opt_method(CORAX_OPT_BLO_NEWTON_FAST)`, `num_searches(1)`, and `topology_opt_method(TopologyOptMethod::adaptive)` as the master default.
- IQ-TREE defaults quoted from the official Command Reference: `-ninit 100`, `-ntop 20`, `-nbest 5`, `-nstop 100`, `-pers 0.5`, `-sprrad 6`, `-allnni` off, default start = 100 parsimony + BIONJ.
- Seq-Gen option syntax from the official manual in `rambaut/Seq-Gen`.
- AliSim syntax (`--alisim`, `-t`, `--length`, `-seed`, `--out-format fasta`, `--num-alignments`, `-rlen`, `--branch-scale`, `RANDOM{yh/N}`) from the IQ-TREE AliSim docs.
- Crossref metadata for Liu et al. 2024 (*Syst. Biol.* 73(5):807–822, [doi:10.1093/sysbio/syae031](https://doi.org/10.1093/sysbio/syae031), abstract retrieved verbatim) and Wright 2026 Treeline (*Mol. Ecol. Resour.* 26, [doi:10.1111/1755-0998.70163](https://doi.org/10.1111/1755-0998.70163), abstract retrieved verbatim).
- Published numbers in the incumbents and headroom tables, each fetched from the cited article page and quoted in the source's own words.
- HPC node specs (the cluster used for the large tiers): 2× AMD EPYC 7763 Milan, 128 cores, 256 GB, 288 GB local SSD.
- Timing table and topology counts: **measured by me on this machine** using the repository's `phylo.py`, this session.

**Inferred, not verified:**

- CLV-op-to-wall-clock conversions in the tier table (e.g. "1×10¹⁰ ops ≈ 60 s") are extrapolations from my Python measurements to a C engine and **must be recalibrated against the real binaries** before budgets are frozen.
- The exhaustive-certification times for G8/G9 assume the current Python optimiser; a C scorer will be far faster, but I did not benchmark one.
- The claim that raising $\varepsilon$ defaults explains `DEF_LH_EPSILON = 10` is my reading of the Free Lunch paper against the source constant; the wiki's older text says `--lh-epsilon` default 0.1, which appears stale.

**Could NOT confirm — treat as open:**

1. **No tool was installed or executed.** I did not download or run `raxml-ng`, `iqtree3`, `fasttree` or `seq-gen`; nothing on this host has them. Every CLI invocation in this spec is transcribed from official documentation or source, not from a successful run. **Before the problem goes live, run `raxml-ng -h` and `iqtree3 -h` on the pinned builds and reconcile every option, spelling and default against this document.**
2. **The bioconda `iqtree` 3.x executable name** — the v3 docs use `iqtree3`, but bioconda's package is named `iqtree` and may install `iqtree2`, `iqtree3` or `iqtree`. Check `ls $CONDA_PREFIX/bin | grep -i iqtree`.
3. **`--opt-model off` semantics in RAxML-NG 2.0.2.** The tuning wiki states it disables model-parameter optimisation in `--evaluate` mode. I could not confirm that a fully parameterised model string (`GTR{...}+FU{...}+G4m{...}`) is honoured *exactly* rather than partially re-estimated. **The whole "fixed θ" design depends on this**; verify empirically by evaluating the same tree under two model strings differing only in a fixed parameter and confirming the log-likelihoods differ.
4. **`--precision 9` and single-threaded determinism** were not tested. Confirm bitwise reproducibility of `--evaluate` across repeated invocations before publishing $\varepsilon_{\text{scorer}}$.
5. **Whether AliSim writes the random tree** when `-t RANDOM{...}` is used. The AliSim docs I fetched say it does not; a search result asserted a `.treefile` is written. The conflict is moot here because the spec generates the true tree itself and passes it with `-t`, which is the safer design regardless.
6. **IQ-TREE `-te` semantics.** The Command Reference text I retrieved describes `-te` in language that reads like `-t` (starting tree). In IQ-TREE `-te` is the *fixed* user tree. Verify with `iqtree3 -h` before wiring the second-opinion scorer.
7. **Exact log-likelihood-unit differences between tools on the same supermatrix** in Zhou et al. 2018. Their Figure 6A plots differences on a log scale and the main text does not give raw log-unit values; my headroom claims from that paper are stated as *frequencies of finding the best score* and as the *qualitative* statement about NNI-recoverable improvements, which are what the text supports. Do not attribute log-unit magnitudes to that paper.
8. **RAxML-NG's default of "10 random + 10 parsimony" starting trees** is taken from Togkousidis et al. 2023's description of standard RAxML-NG, not from a v2.0.2 help string. Confirm on the pinned build — v2's adaptive default may vary the count with predicted difficulty.
9. **Whether `pythiaphylopredictor` 2.0.0's CLI entry point is `pythia`** and whether it accepts a RAxML-NG 2.0.2 binary (its predictor was trained against earlier RAxML-NG versions). Verify, and pin the RAxML-NG version used for difficulty features separately from the one used for scoring if they must differ.