#!/bin/bash
#SBATCH --job-name=formal_21_ece_v3
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=artifacts/slurm/%j.out
#SBATCH --error=artifacts/slurm/%j.err

set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXP_DIR"
mkdir -p artifacts/slurm

export PATH="$HOME/.local/bin:$HOME/.nb-cli/bin:$PATH"

echo "=== Running temporal evaluation (30 seeds) ==="
uv run python run_temporal.py

echo "=== Running spatial evaluation (30 seeds on ECE v3) ==="
uv run python run_spatial.py --n-parallel 16

echo "=== Running distance diagnostics ==="
uv run python analyze_cluster_distances.py

echo "=== Building report notebook ==="
uv run python build_notebook.py

echo "=== Executing report notebook ==="
cd ../..
nb execute experiment/derived_8.4-formal-eval-2.1-ece-v3/derived_8.4-formal-eval-2.1-ece-v3.ipynb --uv
cd experiment/derived_8.4-formal-eval-2.1-ece-v3

echo "=== Updating README.md ==="
uv run python update_readme_from_notebook.py

echo "=== Complete ==="
