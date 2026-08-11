#!/bin/bash
#SBATCH --job-name=mlp21-2regime-sweep
#SBATCH --output=artifacts/slurm/slurm-%j.out
#SBATCH --error=artifacts/slurm/slurm-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16000
#SBATCH --nodes=1
#
# derived_8.4-eval-mlp-2.1 full GPU pipeline (submit from this directory):
#   mkdir -p artifacts/slurm && sbatch run_slurm.sh
#
# 1a verify_rng_guard.py  H100 bit-identity proof (anchor vs 2.0; RNG guard)
# 1b run_mlp_sweep.py     phases 1-3, seeds 42/7/123, ~340 jobs, --resume
# 1c run_mlp_champion.py  5-seed ensembles of the top-1/family winners
# 1d run_mlp_eval.py      leaderboard (incl. 2.0 refs), per-regime, figures
# 1e analyze_*.py         bias / swa / selection / overfitting / ood / stopping
# (the report notebook runs offline afterwards on the login node)
#
# n_parallel comes from config.yaml (8; 2.0 parity). If the node OOMs or
# throttles at 6 CPUs, add `--n-parallel 6` to the sweep step (still < 2 h).
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

step uv run --no-sync python verify_rng_guard.py --epochs 6
step uv run --no-sync python run_mlp_sweep.py --resume
step uv run --no-sync python run_mlp_champion.py --top-n 1
step uv run --no-sync python run_mlp_eval.py
step uv run --no-sync python analyze_bias.py
step uv run --no-sync python analyze_swa.py
step uv run --no-sync python analyze_selection.py
step uv run --no-sync python analyze_overfitting.py
step uv run --no-sync python analyze_extrapolation.py
step uv run --no-sync python analyze_stopping.py --tag 20

echo
echo "[slurm] ALL DONE $(date) — job ${SLURM_JOB_ID:-?} exit 0"
