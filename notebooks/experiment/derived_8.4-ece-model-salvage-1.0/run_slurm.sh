#!/bin/bash
#SBATCH --job-name=d84_ece_model_salvage
#SBATCH --partition=gpu_debug
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=artifacts/slurm/%j.out
#SBATCH --error=artifacts/slurm/%j.err

set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXP_DIR"
mkdir -p artifacts/slurm artifacts/checkpoints artifacts/predictions figures
export PATH="$HOME/.local/bin:$HOME/.nb-cli/bin:$PATH"

echo "=== job ${SLURM_JOB_ID:-?} start $(date) host $(hostname) ==="
nvidia-smi -L 2>/dev/null | head -2 || true

step() {
    echo
    echo "===== $(date +%H:%M:%S) $* ====="
    "$@"
}

step uv run --no-sync python run_model_salvage.py --resume
step uv run --no-sync python build_notebook.py

cd ../..
step nb execute experiment/derived_8.4-ece-model-salvage-1.0/derived_8.4-ece-model-salvage-1.0.ipynb --uv --timeout 1800
cd experiment/derived_8.4-ece-model-salvage-1.0

step uv run --no-sync python update_readme.py
echo "=== complete $(date) ==="
