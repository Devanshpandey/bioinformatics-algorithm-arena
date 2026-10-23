#!/bin/bash
#SBATCH -J p09_disagree
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o logs/p09_disagree_%j.out
#SBATCH -e logs/p09_disagree_%j.err

# THE DECIDING EXPERIMENT FOR P09.
#
# Small-scale runs already showed raxml-ng and iqtree agreeing to within
# 0.02-0.31 log-units at 16-48 taxa, which means the headroom measured against
# our own NNI baseline was our baseline being weak, not the problem being open.
#
# If the two SOTA tools also agree at 128 and 256 taxa, P09 is saturated
# everywhere we can afford to evaluate and must be DEMOTED rather than scored.
# If they diverge, the gap between them is genuine tool-independent headroom and
# P09 survives at that scale. Either answer is publishable; the wrong move is to
# run a search campaign without knowing which.

set -euo pipefail
source "${ARENA_ROOT:?export ARENA_ROOT=/path/to/bioinformatics-algorithm-arena}/cluster/activate.sh"
cd "$ARENA_ROOT/problems/P09_ml_tree_search"

python incumbent_disagreement.py \
    --tiers hard32,hard48,hard128,hard256 \
    --seeds 42,43,44,45,46 \
    --raxml-mode="--search" \
    --out "disagreement_hpc.jsonl"

echo "=== summary ==="
python - << 'PY'
import json, collections
rows=[json.loads(l) for l in open("disagreement_hpc.jsonl")]
by=collections.defaultdict(list)
for r in rows: by[(r["tier"],r["n_taxa"])].append(r)
print(f"{'tier':>9} {'taxa':>5} {'n':>3} {'median gap':>11} {'max gap':>9} {'median RF(rax,iq)':>18}")
for (t,n),rs in sorted(by.items(), key=lambda kv: kv[0][1]):
    g=sorted(x["gap"] for x in rs); rf=sorted(x["rf_raxml_iqtree"] for x in rs)
    print(f"{t:>9} {n:>5} {len(rs):>3} {g[len(g)//2]:>11.3f} {max(g):>9.3f} {rf[len(rf)//2]:>18}")
PY
