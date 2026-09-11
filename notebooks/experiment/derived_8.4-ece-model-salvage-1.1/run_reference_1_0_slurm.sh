#!/bin/bash
# Refreshes the missing 1.0 reference seeds required by the 1.1 paired
# comparison and seven-line ECE version charts.  The 1.0 source code is not
# modified; this job only completes its resumable artifacts.
#SBATCH --job-name=d84_ece_salvage10_ref
#SBATCH --partition=gpu_debug
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=reference_artifacts/slurm/%j.out
#SBATCH --error=reference_artifacts/slurm/%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REFERENCE_DIR="${ROOT_DIR}/../derived_8.4-ece-model-salvage-1.0"
cd "$REFERENCE_DIR"

echo "=== 1.0 reference refresh ${SLURM_JOB_ID:-?} start $(date) host $(hostname) ==="
nvidia-smi -L 2>/dev/null | head -2 || true
# The 1.0 runner resumes by default; its CLI predates an explicit --resume flag.
uv run --no-sync python run_model_salvage.py
echo "=== 1.0 reference refresh complete $(date) ==="
