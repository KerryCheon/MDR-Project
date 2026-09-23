#!/bin/bash
#SBATCH --job-name=routing-opt-1.0
#SBATCH --output=artifacts/slurm/slurm-%j.out
#SBATCH --error=artifacts/slurm/slurm-%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000
#SBATCH --nodes=1
#
# derived_8.4-routing-optimize-1.0 full GPU pipeline (submit from this directory):
#   mkdir -p artifacts/slurm && sbatch run_slurm.sh
#
# Scope: 3 pinned no-delta configurations (Guarded_Backbone54_k2 primary
# candidate, StationMean_Backbone54_k2 challenger, Clustering_Backbone54_k2
# replication anchor) x 30 temporal seeds (frozen split) + 3 x 5 LOSO seeds x 7
# held-out stations = 90 + 105 jobs. Estimated GPU wall ~2-4 h (XGBoost GPU folds
# serialize on one H100: n_parallel=8 workers buy resilience/resume, not
# aggregate throughput — observed in eval-1.3/formal-eval-1.0).
#
# If the wall is hit, re-submit: completed jobs resume via
# artifacts/jobs/<config_id>__s<seed>__<station>/meta.json (data_version + file
# presence match).
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

step uv run --no-sync python run_temporal.py --n-parallel 8
step uv run --no-sync python run_loso.py --n-parallel 8

echo
echo "[slurm] ALL DONE $(date) — job ${SLURM_JOB_ID:-?} exit 0"
