#!/bin/bash
#SBATCH --job-name=eval14-loso
#SBATCH --output=artifacts/slurm/slurm-%j.out
#SBATCH --error=artifacts/slurm/slurm-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_debug
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000
#SBATCH --nodes=1
#
# derived_8.4-eval-1.4 full GPU pipeline (submit from this directory):
#   mkdir -p artifacts/slurm && sbatch run_slurm.sh
#
# Scope: --new-strategy-only — compute the 12 NEW gating K-sweep configurations
# (Clustering_Backbone54_k3/k4, Clustering_Static_k2/k3/k4,
# Clustering_Weather_k2/k3/k4, Clustering_Dynamic_k3/k4, Clustering_V0_Full_k3/k4;
# 84 LOSO folds + 12 full-baseline configs = 96 jobs) and merge the 56 eval-1.3
# configs' results as references (deterministic same-protocol results; see
# run_loso.py). XGBoost GPU folds serialize on a single H100 (~35 s/fold
# effective at 8 workers — workers buy resilience/resume, not throughput), so
# the run is ~45-70 min; the 2 h wall fits with margin.
#
# Run order (full baseline FIRST so run_loso.py can backfill the NEW configs'
# temporal test R2 from full_config_summary.csv):
#   1a run_full_baseline.py   --new-strategy-only (12 configs), ~5-8 min
#   1b run_loso.py            --new-strategy-only (12 configs x 7 stations), ~40-60 min
# If the 2 h wall is hit, re-submit: completed folds are resumed via
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

step uv run --no-sync python run_full_baseline.py --new-strategy-only --n-parallel 8
step uv run --no-sync python run_loso.py --new-strategy-only --n-parallel 8

echo
echo "[slurm] ALL DONE $(date) — job ${SLURM_JOB_ID:-?} exit 0"
