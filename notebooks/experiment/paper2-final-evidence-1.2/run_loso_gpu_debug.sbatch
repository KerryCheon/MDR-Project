#!/bin/bash
#SBATCH --job-name=p2e12-global-loso
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:55:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

set -euo pipefail
EXP_DIR="${SLURM_SUBMIT_DIR:?Submit this script from the evidence directory}"
PROJECT_ROOT="$(cd "$EXP_DIR/../../.." && pwd)"
cd "$EXP_DIR"
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
echo "[slurm] job=${SLURM_JOB_ID:-?} stage=loso-global start=$(date) host=$(hostname)"
nvidia-smi -L 2>/dev/null | head -2 || true
uv run --project "$PROJECT_ROOT/notebooks" --no-sync python "$EXP_DIR/run_loso.py" --config-id Global_Single_54 --n-parallel 8
echo "[slurm] loso-global complete $(date)"
