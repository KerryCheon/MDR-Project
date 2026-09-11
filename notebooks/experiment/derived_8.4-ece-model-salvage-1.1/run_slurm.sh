#!/bin/bash
#SBATCH --job-name=d84_ece_salvage11_models
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

# Slurm runs a staged copy of this script, so dirname "$0" is not the
# experiment directory.  SLURM_SUBMIT_DIR is the reproducible submission root.
EXP_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$EXP_DIR"
mkdir -p artifacts/slurm artifacts/checkpoints artifacts/predictions figures
echo "=== job ${SLURM_JOB_ID:-?} start $(date) host $(hostname) ==="
nvidia-smi -L 2>/dev/null | head -2 || true

# Compute nodes may not expose the login-node Jupyter runtime directory.
# Keep nb's kernel/runtime files in a job-local writable directory.
NB_RUNTIME_ROOT="${SLURM_TMPDIR:-/tmp}"
export JUPYTER_RUNTIME_DIR="${NB_RUNTIME_ROOT}/d84_ece_salvage11_jupyter_${SLURM_JOB_ID:-manual}"
mkdir -p "$JUPYTER_RUNTIME_DIR"

test -f feature_selection_artifacts/selected_features.json || {
    echo "Feature-selection manifest is missing; refusing to start model stage." >&2
    exit 2
}

# The 1.0 reference refresh is a prerequisite for the paired report and the
# seven-line global version charts.  It completed before this dependent stage
# was submitted; validate its five common-seed artifacts at runtime as well.
REFERENCE_1_0_DIR="../derived_8.4-ece-model-salvage-1.0/artifacts/predictions"
for seed in 42 7 13; do
    test -f "$REFERENCE_1_0_DIR/Global_Single_54_no_smap__s${seed}.csv" || {
        echo "Missing 1.0 reference artifact for seed ${seed}; refusing to start." >&2
        exit 3
    }
done

FORMAL_DIR="../derived_8.4-formal-eval-2.1-ece-v3/predictions_spatial"
for seed in 42 7 13; do
    test -f "$FORMAL_DIR/Global_Single_54__s${seed}__ece_preds.npy" || {
        echo "Missing original formal reference artifact for seed ${seed}; refusing to start." >&2
        exit 4
    }
done

step() {
    echo
    echo "===== $(date +%H:%M:%S) $* ====="
    "$@"
}

 # The runner resumes by default; its CLI uses --no-resume only to force refits.
step uv run --no-sync python run_model_salvage.py
step uv run --no-sync python build_notebook.py

cd ../..
step nb execute experiment/derived_8.4-ece-model-salvage-1.1/derived_8.4-ece-model-salvage-1.1.ipynb --uv --timeout 1800
step nb execute experiment/derived_8.4-ece-model-salvage-1.1/ece-input-weather-overlay-1.0.ipynb --uv --timeout 600
cd experiment/derived_8.4-ece-model-salvage-1.1

step uv run --no-sync python update_readme.py
echo "=== complete $(date) ==="
