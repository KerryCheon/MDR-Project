#!/bin/bash
#SBATCH --job-name=d84_ece_salvage11_fs
#SBATCH --partition=gpu_debug
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=feature_selection_artifacts/slurm/%j.out
#SBATCH --error=feature_selection_artifacts/slurm/%j.err

set -euo pipefail

# Slurm runs a staged copy of this script, so dirname "$0" is not the
# experiment directory.  SLURM_SUBMIT_DIR is the reproducible submission root.
EXP_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$EXP_DIR"
mkdir -p feature_selection_artifacts/slurm

echo "=== feature-selection job ${SLURM_JOB_ID:-?} start $(date) host $(hostname) ==="
nvidia-smi -L 2>/dev/null | head -2 || true
uv run --no-sync python run_feature_selection.py --stage all --resume
echo "=== feature-selection complete $(date) ==="
