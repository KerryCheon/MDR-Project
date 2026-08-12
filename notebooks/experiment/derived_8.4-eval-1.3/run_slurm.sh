#!/bin/bash
#SBATCH --job-name=eval13-loso
#SBATCH --output=artifacts/slurm/slurm-%j.out
#SBATCH --error=artifacts/slurm/slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_debug
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000
#SBATCH --nodes=1
#
# derived_8.4-eval-1.3 full GPU pipeline (submit from this directory):
#   mkdir -p artifacts/slurm && sbatch run_slurm.sh
#
# Scope: --new-strategy-only — compute the 9 NEW Clustering_Backbone54_k2
# configs (63 LOSO folds + 9 full-baseline configs) and merge the 47 eval-1.1
# configs' results as eval-1.2 references (deterministic same-protocol results;
# see run_loso.py). The full 56-config run needs ~2 h of GPU time and cannot
# fit the 1 h wall: at 8 workers each XGBoost fold took 128-310 s (GPU
# serializes concurrent processes; no aggregate speedup), i.e. ~35 s/fold
# effective. With 72 jobs at n_parallel=6 (1 CPU per worker = the 6-CPU
# allocation) the run is ~30-40 min.
#
# Run order (full baseline FIRST so run_loso.py can backfill the NEW
# Clustering_Backbone54_k2 configs' temporal test R2 from full_config_summary.csv):
#   1a run_full_baseline.py   --new-strategy-only (9 configs), ~3-5 min
#   1b run_loso.py            --new-strategy-only (9 configs x 7 stations), ~25-35 min
# If the 1 h wall is hit, re-submit: completed folds are resumed via
# artifacts/jobs/<config_id>__<station>/meta.json (data_version match).
set -euo pipefail

# SLURM copies the script to its spool dir, so BASH_SOURCE[0] is NOT the
# submitted path — use SLURM_SUBMIT_DIR (set by slurm to the submission cwd).
EXP_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$EXP_DIR"
if [ ! -f "$EXP_DIR/config.yaml" ]; then
    echo "ERROR: config.yaml not found in $EXP_DIR — wrong dir resolution" >&2
    exit 1
fi
mkdir -p artifacts/slurm
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"

echo "[slurm] job ${SLURM_JOB_ID:-?} start $(date) host $(hostname) exp_dir=$EXP_DIR"
nvidia-smi -L 2>/dev/null | head -2 || true

step() { echo; echo "===== $(date +%H:%M:%S)  $* ====="; "$@"; }

step uv run --no-sync python run_full_baseline.py --new-strategy-only --n-parallel 6
step uv run --no-sync python run_loso.py --new-strategy-only --n-parallel 6

echo
echo "[slurm] ALL DONE $(date) — job ${SLURM_JOB_ID:-?} exit 0"
