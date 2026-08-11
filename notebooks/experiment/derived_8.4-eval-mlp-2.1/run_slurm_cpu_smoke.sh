#!/bin/bash
#SBATCH --job-name=mlp21-cpu-smoke
#SBATCH --output=artifacts/slurm/smoke-%j.out
#SBATCH --error=artifacts/slurm/smoke-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000
#SBATCH --nodes=1
#
# CPU-only validation of the derived_8.4-eval-mlp-2.1 pipeline under sbatch
# (no GPU; run BEFORE the GPU job to prove the compute-node env works). Runs:
#   (0a) the RNG-guard proof            -> verify_rng_guard.py
#   (0b) a 3-epoch smoke sweep          -> run_mlp_sweep.py --smoke
#        (data_version -1, never reused by the real v8 run)
#   (0c) the eval path                  -> run_mlp_eval.py
# Submit from this directory:
#   mkdir -p artifacts/slurm && sbatch run_slurm_cpu_smoke.sh
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

echo "[smoke] job ${SLURM_JOB_ID:-?} start $(date) host $(hostname) exp_dir=$EXP_DIR"
step() { echo; echo "===== $(date +%H:%M:%S)  $* ====="; "$@"; }

step uv run --no-sync python verify_rng_guard.py --epochs 6
step uv run --no-sync python run_mlp_sweep.py --smoke --n-parallel 2 \
      --families 2regime_54 --only w384x384_d0.3_gelu,w448x448_d0.3_gelu_swa085
step uv run --no-sync python run_mlp_eval.py --top-n 2 --per-regime-n 2

echo
echo "[smoke] ALL DONE $(date) — job ${SLURM_JOB_ID:-?} exit 0"
