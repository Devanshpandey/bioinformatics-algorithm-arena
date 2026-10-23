#!/bin/bash
#SBATCH -J p01_scaling
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o logs/p01_scaling_%j.out
#SBATCH -e logs/p01_scaling_%j.err

# P01 across the size ladder, for the scaling-curve component (SPEC.md II.4).
# Note the rule: three tiers is three points and a power law fit to three points
# is numerology. This is a first pass; the real ladder needs >=7 log-spaced
# sizes over >=2.5 decades before any exponent is reported.

set -euo pipefail
source "${ARENA_ROOT:?export ARENA_ROOT=/path/to/bioinformatics-algorithm-arena}/cluster/activate.sh"
cd "$ARENA_ROOT/problems/P01_haplotype_long_match"

for tier in tiny small medium; do
  for seed in 42 43 44; do
    for sub in sub_vectorized_pbwt sub_streaming_pbwt; do
      echo "=== $tier seed=$seed $sub ==="
      python evaluate.py --tier "$tier" --seed "$seed" --L 300 \
          --submission "$sub" --gate-instances 8 --repeats 3 \
          --log scaling_hpc.jsonl || echo "FAILED: $tier $seed $sub"
    done
  done
done
echo "P01 SCALING DONE"
