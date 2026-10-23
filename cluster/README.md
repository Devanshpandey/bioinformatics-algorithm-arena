# Running the scored tiers on a SLURM cluster

The small tiers run on a laptop. The scored tiers do not: P01 is scored at
N ≥ 10,000 haplotypes (docs/SPEC.md Finding 10) and P09 at n_taxa ≥ 128
(Findings 11 and 16). The results in `docs/` came from 128-core AMD EPYC 7763
nodes with 256 GB RAM.

```bash
export ARENA_ROOT=/path/to/bioinformatics-algorithm-arena
cd "$ARENA_ROOT/cluster" && mkdir -p logs
# edit "#SBATCH -A YOUR_ALLOCATION" (and the partition names) in each script, then:
sbatch slurm/gates.slurm                  # every correctness gate; run first, and after any env change
sbatch slurm/p01_candidates_scored.slurm  # space score, incumbent + six candidates, all tiers
sbatch slurm/p01_large_n.slurm            # N-scaling curve into the scored regime
sbatch slurm/p09_disagreement.slurm       # RAxML-NG vs IQ-TREE at 32–256 taxa
```

`activate.sh` activates the `arena` environment (conda or micromamba; override the
name with `ARENA_ENV`) and pins every BLAS/OpenMP thread count to 1.

## Two things that will bite you

**1. Threads must be pinned.** Unpinned, numpy can die on a login node with
`OpenBLAS error: Memory allocation still failed after 10 retries`. Beyond the
crash, docs/SPEC.md II.3 requires the *harness* to set thread count, never the
candidate, so leave it pinned and set it explicitly per job.

**2. Instances are not reproducible across machines.** The same seed gave different
panels on a macOS laptop and on the Linux cluster: P02's tiny tier yields 7,382
sites on one and 7,384 on the other, and P01's small tier gives 9,552 matches on one
and 9,603 on the other. A later check with matched msprime/tskit versions on macOS
still gave the laptop values, so the platform matters, not only the version
(docs/SPEC.md Finding 8). Every result row therefore
carries an `instance` fingerprint (content SHA + generator versions) from
`harness/measure.py:instance_fingerprint`. **Never compare two rows whose
`instance.sha` differs.** That measures the instance, not the algorithm. P09 is
exempt: it uses only numpy's RNG and reproduces bit-identically.

## Why `p09_disagreement` mattered

Measured with our own scorer evaluating every tree, so the comparison is
apples-to-apples (deficit = log-units below the best tree found by any method):

| tier | taxa × sites | arena-NNI deficit | raxml-ng deficit | iqtree deficit | our slowdown |
|---|---|---:|---:|---:|---:|
| small | 16 × 1000 | 0.108 | 0.000 | 0.000 | 9.5× |
| hard16 | 16 × 200 | 0.78 / 0.34 | 0.000 | 0.000 | ~4× |
| hard32 | 32 × 300 | 15.58 / 4.74 | 0.307 / 0.063 | **0.000** | ~10× |
| hard48 | 48 × 300 | 20.38 / 9.13 | 0.238 / 0.017 | **0.000** | ~13× |

The 11.74 log-unit "headroom" first measured against our own NNI baseline did not
survive contact with the real incumbents: below ~50 taxa, raxml-ng and iqtree land
within 0.02–0.31 log-units of each other. `p09_disagreement.slurm` extended the
comparison to 128 and 256 taxa, where the two tools do diverge (docs/SPEC.md
Finding 11), and a matched-CPU sweep confirmed the gap is not a budget artifact
(Finding 16).
