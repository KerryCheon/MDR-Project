#!/bin/bash
#SBATCH --job-name=formal-eval-1.0
#SBATCH --output=artifacts/slurm/slurm-%j.out
#SBATCH --error=artifacts/slurm/slurm-%j.err
#SBATCH --time=14:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000
#SBATCH --nodes=1
#
# derived_8.4-formal-eval-1.0 full GPU pipeline (submit from this directory):
#   mkdir -p artifacts/slurm && sbatch run_slurm.sh
#
# Scope: 20 pinned configurations x 30 temporal seeds (full-baseline protocol on the
# frozen split) + 20 x 5 LOSO seeds x 7 held-out stations + the val-selected delta
# selection grid (6 strategies x 9 val-grid points, train-only fits). Estimated GPU
# wall ~11-12 h (XGBoost GPU folds serialize on one H100: n_parallel=8 workers buy
# resilience/resume, not aggregate throughput — observed in eval-1.3).
#
# Run order:
#   1. select_deltas_val.py  — val-selected delta protocol (~30 min GPU); writes
#      val_selected_deltas.json consumed by the drivers' config pinning.
#   2. run_temporal.py       — 20 configs x 30 seeds (~5 h GPU).
#   3. run_loso.py           — 20 configs x 5 seeds x 7 stations (~5.8 h GPU).
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

step uv run --no-sync python select_deltas_val.py
step uv run --no-sync python run_temporal.py --n-parallel 8
step uv run --no-sync python run_loso.py --n-parallel 8

echo
echo "[slurm] ALL DONE $(date) — job ${SLURM_JOB_ID:-?} exit 0"
