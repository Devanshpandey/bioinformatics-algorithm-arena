# Source this from a SLURM script or an interactive shell on the cluster.
#   export ARENA_ROOT=/path/to/bioinformatics-algorithm-arena
#   source "$ARENA_ROOT/cluster/activate.sh"
#
# Activate the environment from environment.yml however your site does it
# (conda, micromamba, a module), then pin threads. Unpinned BLAS can crash on
# login nodes, and docs/SPEC.md II.3 requires the harness, not the candidate,
# to set the thread count.

: "${ARENA_ROOT:?export ARENA_ROOT=/path/to/bioinformatics-algorithm-arena}"
: "${ARENA_ENV:=arena}"

if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"; micromamba activate "$ARENA_ENV"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"; conda activate "$ARENA_ENV"
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
