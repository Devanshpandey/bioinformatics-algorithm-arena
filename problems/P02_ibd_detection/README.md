# P02 — IBD segment detection

| file | role |
|---|---|
| `generate.py` | msprime + tskit instance generator; **exact genealogical IBD truth**, fresh seed |
| `panel_matches.py` | all-pairs maximal long matches: brute-force oracle + Durbin Algorithm 3 (validated 40/40) |
| `baseline_hapibd.py` | incumbent: seed-and-extend (the hap-IBD design), built on P01's PBWT |
| `evaluate.py` | cM-level recall/precision (fragmentation-robust) + segment power + resources |
| `problem.yaml` | tiers, metrics, mutable surface, and the genealogical-vs-detectable caveat |

```bash
python3 evaluate.py --tier small --seed 42
```

Measured (600 haplotypes × 43,521 sites, 251 true segments, seed 42):
cM-recall **0.332**, cM-precision **0.791**, 100 calls, 65.1 s.

**Two design points worth reading before using this.** First, the truth is
*genealogical* — tskit ignores allelic state entirely — while callers detect
identity by state. Absolute recall is therefore low even for a correct caller,
and the arena scores only relative to the incumbent. Second, `panel_matches.py`
imports P01's PBWT column update **by explicit path**: putting P01's directory on
`sys.path` made `import generate` silently load the wrong problem's generator,
which is exactly the kind of self-contamination a benchmark suite must not have.
