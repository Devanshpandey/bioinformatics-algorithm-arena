#!/bin/bash
#SBATCH -J p01_scored
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o logs/p01_scored_%j.out
#SBATCH -e logs/p01_scored_%j.err

# Space score (bits per panel cell) for the incumbent and every agent candidate
# at all three tiers, with the correctness gate on. This is the run behind the
# transfer table in docs/RLPBWT.md §7.4: the small tier alone mis-ranks the
# candidates, so never report a ranking without n10k and n30k.

set -euo pipefail
source "${ARENA_ROOT:?export ARENA_ROOT=/path/to/bioinformatics-algorithm-arena}/cluster/activate.sh"
cd "$ARENA_ROOT/problems/P01_haplotype_long_match"

MODULES="sub_rlpbwt_streaming candidates.sub_column_diff candidates.sub_bitvector_rank
candidates.sub_freeform candidates.sub_delta_varint candidates.sub_bitpack_fixed
candidates.sub_eliasfano"

for tier in small n10k n30k; do
  for mod in $MODULES; do
    echo "=== $tier / $mod ==="
    python score_submission.py --module "$mod" --tier "$tier" --seed 42 --json \
        >> scored_candidates_hpc.jsonl || echo "FAILED (recorded): $tier $mod"
  done
done
echo "P01 CANDIDATES SCORED DONE"
