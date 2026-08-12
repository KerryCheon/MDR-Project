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
# Run order (full baseline FIRST so run_loso.py can backfill the NEW
# Clustering_Backbone54_k2 configs' temporal test R2 from full_config_summary.csv):
#   1a run_full_baseline.py   56 configs on full trainval (no LOSO), ~3-5 min
#   1b run_loso.py            56 configs x 7 stations = 392 folds, ~35-45 min
# n_parallel comes from config.yaml (8; eval-2.0 parity). XGBoost workers use
# 1 CPU each (n_jobs=1): 8 workers mildly oversubscribe the 6-CPU allocation —
# if the node throttles, add `--n-parallel 6` to the steps (still < 1 h).
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

step uv run --no-sync python run_full_baseline.py --n-parallel 8
step uv run --no-sync python run_loso.py --n-parallel 8

echo
echo "[slurm] ALL DONE $(date) — job ${SLURM_JOB_ID:-?} exit 0"
