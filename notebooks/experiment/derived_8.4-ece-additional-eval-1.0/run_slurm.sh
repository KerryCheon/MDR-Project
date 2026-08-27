#!/bin/bash
#SBATCH --job-name=d84_ece_eval10
#SBATCH --output=artifacts/slurm/slurm-%j.out
#SBATCH --error=artifacts/slurm/slurm-%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=gpu_debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1

set -euo pipefail

EXP_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$EXP_DIR"
mkdir -p artifacts/slurm models predictions figures

command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"

echo "[slurm] job ${SLURM_JOB_ID:-?} start $(date) host $(hostname) exp_dir=$EXP_DIR"
nvidia-smi -L 2>/dev/null | head -2 || true

step() { echo; echo "===== $(date +%H:%M:%S)  $* ====="; "$@"; }

step uv run --no-sync python run_pipeline.py
step uv run --no-sync python plot_generator.py
step uv run --no-sync python build_notebook.py

echo "=== Executing report notebook ==="
cd ../..
nb execute experiment/derived_8.4-ece-additional-eval-1.0/derived_8.4-ece-additional-eval-1.0.ipynb --uv
cd experiment/derived_8.4-ece-additional-eval-1.0

step uv run --no-sync python update_readme.py

echo
echo "[slurm] ALL DONE $(date) — job ${SLURM_JOB_ID:-?} exit 0"
