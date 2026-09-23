#!/bin/bash
#SBATCH --job-name=p2-ece-1p1
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail
EXP_DIR="${SLURM_SUBMIT_DIR:?Submit this script from ece_guarded}"
PROJECT_ROOT="$(cd "$EXP_DIR/../../../.." && pwd)"
cd "$EXP_DIR"
command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
echo "[slurm] job=${SLURM_JOB_ID:-?} stage=guarded-ece start=$(date) host=$(hostname)"
uv run --project "$PROJECT_ROOT/notebooks" --no-sync python "$EXP_DIR/run_guarded.py"
echo "[slurm] Paper 2 ECE v1.1 complete $(date)"
