# P09 — Maximum-likelihood phylogenetic tree search

The cleanest evaluator in the arena: the score is the log-likelihood of the
submission's own output, computed exactly. Nothing to download, memorize, or bias.

| file | role |
|---|---|
| `phylo.py` | exact JC69 Felsenstein pruning, branch-length optimisation, topology enumeration, RF distance |
| `generate.py` | alignments simulated along a known random tree, fresh seed |
| `baseline_search.py` | incumbent: neighbour-joining start + NNI hill climbing |
| `evaluate.py` | **exhaustive-optimum gate** + score at fixed CPU budget |
| `difficulty_probe.py` | the headroom test: is this problem saturated? |
| `problem.yaml` | tiers, mutable surface, measured headroom, and the scaffold-fidelity blocker |

```bash
python3 evaluate.py --tier small --seed 42 --budget 25
```

**The gate is the unusual part.** At 6 taxa there are exactly 105 unrooted
topologies, so the harness enumerates all of them, optimises branch lengths on
each, and takes the maximum — the *provable global optimum*. A submission must
reach within 0.5 log-units of it. Elsewhere in the arena we can verify
correctness; here we verify **optimality**, so a heuristic that quietly returns
local optima cannot hide.

> **Update (superseded below):** P09 was later reinstated with its scored regime
> bounded to **n_taxa ≥ 128**, where RAxML-NG and IQ-TREE diverge even at matched
> CPU. See `docs/SPEC.md` Findings 11 and 16. The note below is kept as the record.

**Do not score this problem yet.** `difficulty_probe.py` shows the incumbent is
beatable only at 16 taxa × 200 sites with long branches (11.74 log-units), and
that gap is measured against *our own* NNI hill-climber, not against `raxml-ng`
or `iqtree`. Per SPEC.md §II.2 that makes it an artifact of a weak reference
until the scaffold-fidelity ratio is published.
