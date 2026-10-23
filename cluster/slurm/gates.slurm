#!/bin/bash
#SBATCH -J arena_gates
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o logs/gates_%j.out
#SBATCH -e logs/gates_%j.err

# Verify every problem's correctness gate on a compute node.
# Run this FIRST after any environment change: a gate that passes on the login
# node but fails here means the environment is not what you think it is.

set -euo pipefail
source "${ARENA_ROOT:?export ARENA_ROOT=/path/to/bioinformatics-algorithm-arena}/cluster/activate.sh"
cd "$ARENA_ROOT/problems"

echo "=== node: $(hostname)  threads: $OMP_NUM_THREADS ==="
python -c "import msprime,tskit,numpy; print('msprime',msprime.__version__,'tskit',tskit.__version__,'numpy',numpy.__version__)"

echo; echo "########## P01 ##########"
cd P01_haplotype_long_match
for sub in baseline_pbwt sub_vectorized_pbwt sub_streaming_pbwt; do
  echo "--- $sub ---"
  python evaluate.py --tier small --seed 42 --L 300 --submission "$sub" \
      --gate-instances 16 --repeats 3 --log results_hpc.jsonl
done

echo; echo "########## P02 ##########"
cd ../P02_ibd_detection
python evaluate.py --tier small --seed 42 --log results_hpc.jsonl

echo; echo "########## P09 ##########"
cd ../P09_ml_tree_search
python evaluate.py --tier small --seed 42 --budget 60 --gate-instances 3 --log results_hpc.jsonl

echo; echo "ALL GATES DONE"
