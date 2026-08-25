#!/bin/bash
#SBATCH --job-name=formal-eval-2.0
#SBATCH --output=artifacts/slurm/slurm-%j.out
#SBATCH --error=artifacts/slurm/slurm-%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000
#SBATCH --nodes=1
#
# derived_8.4-formal-eval-2.0 full GPU pipeline (submit from this directory):
#   mkdir -p artifacts/slurm && sbatch run_slurm.sh
#
# Scope: 20 pinned configurations x 30 seeds on both:
#   1. Temporal Washington test split (reused from formal-eval-1.0)
#   2. Out-of-State spatial evaluation on derived_8.4-oos (10 unseen stations, 25,176 rows)
#
set -euo pipefail

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
step uv run --no-sync python run_spatial.py --n-parallel 8
step uv run --no-sync python -m eval_formal.stats
step nb execute derived_8.4-formal-eval-2.0.ipynb --uv --timeout 1800
step uv run --no-sync python update_readme_from_notebook.py

echo
echo "[slurm] ALL DONE $(date) — job ${SLURM_JOB_ID:-?} exit 0"
