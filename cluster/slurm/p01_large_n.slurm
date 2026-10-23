#!/bin/bash
#SBATCH -J p01_large_n
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o logs/p01_large_n_%j.out
#SBATCH -e logs/p01_large_n_%j.err

# P01 AT THE SCORED SCALE.
#
# NOTE: the cluster this was written for allocates whole nodes (256 GB) and
# REJECTS an explicit --mem. Add one if your scheduler needs it.
#
# SPEC.md Finding 10: below N~10,000 the zero-byte streaming submission beats the
# stored index at every query count, so the laptop tiers cannot distinguish index
# quality -- the very thing this problem exists to measure. The regime where an
# index earns its bytes (and where the literature operates: mu-PBWT reports
# N=100,000) starts here.
#
# Memory warning: the dense int32 a+d index costs 8 bytes/panel-cell. At
# N=100,000 x M~10,000 that is ~8 TB and WILL NOT FIT -- which is itself the
# result. Expect sub_vectorized_pbwt to fail at n30k/large while
# sub_streaming_pbwt (0 bytes) survives. Record the failure; do not hide it.
# (Post-run correction: this prediction was WRONG. M grows with N in a coalescent
# panel, so n30k dense was ~29 GB and ran fine. See docs/SPEC.md Finding 12.)

set -euo pipefail
source "${ARENA_ROOT:?export ARENA_ROOT=/path/to/bioinformatics-algorithm-arena}/cluster/activate.sh"
cd "$ARENA_ROOT/problems/P01_haplotype_long_match"

echo "=== node $(hostname), $(free -g | awk '/Mem:/{print $2}') GB RAM ==="

# 1) the N-scaling curve, extended into the scored regime
python scaling_N.py --sizes 1000,2000,4000,8000,16000,32000 \
    --seq-len 2e6 --L 200 --seed 42 --out scaling_N_hpc.jsonl || true

# 2) the scored tiers, per submission, with the correctness gate on
for tier in n10k n30k; do
  for sub in sub_streaming_pbwt sub_vectorized_pbwt; do
    echo "=== $tier / $sub ==="
    timeout 6h python evaluate.py --tier "$tier" --seed 42 --L 300 \
        --submission "$sub" --gate-instances 8 --repeats 2 \
        --log scored_hpc.jsonl || echo "FAILED (recorded): $tier $sub"
  done
done

echo "P01 LARGE-N DONE"
