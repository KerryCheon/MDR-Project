#!/bin/bash
#SBATCH --job-name=formal-eval-valrefresh
#SBATCH --output=artifacts/slurm/valrefresh-%j.out
#SBATCH --error=artifacts/slurm/valrefresh-%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000
#SBATCH --nodes=1
#
# Follow-up to run_slurm.sh (job 2043362): re-run the val-selected delta protocol with
# the SAME exact hyperparameters as the final evaluation (2500 trees — the first run's
# selection phase used the 500-tree proxy params for the ranking/grid fits, a protocol
# deviation fixed in select_deltas_val.py). If the val winners changed, purge the stale
# per-seed artifacts of the 6 val-winner configs and recompute them; the 14
# test-selected/none configs resume as completed (data_version unchanged). Then
# re-aggregate all 20 configs.
set -euo pipefail

EXP_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$EXP_DIR"
mkdir -p artifacts/slurm
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"

echo "[slurm] job ${SLURM_JOB_ID:-?} start $(date) host $(hostname) exp_dir=$EXP_DIR"
nvidia-smi -L 2>/dev/null | head -2 || true

step() { echo; echo "===== $(date +%H:%M:%S)  $* ====="; "$@"; }

step uv run --no-sync python select_deltas_val.py
if [ "$(uv run --no-sync python refresh_val_winners.py --check)" = "changed" ]; then
    echo "[refresh] val winners changed — purging stale artifacts and recomputing"
    step uv run --no-sync python refresh_val_winners.py --purge
else
    echo "[refresh] val winners unchanged — nothing to recompute"
fi
step uv run --no-sync python run_temporal.py --n-parallel 8
step uv run --no-sync python run_loso.py --n-parallel 8

echo
echo "[slurm] ALL DONE $(date) — job ${SLURM_JOB_ID:-?} exit 0"
