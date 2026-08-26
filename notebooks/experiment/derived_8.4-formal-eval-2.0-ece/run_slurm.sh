#!/bin/bash
#SBATCH --job-name=formal_20_ece
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=artifacts/slurm/%j.out
#SBATCH --error=artifacts/slurm/%j.err

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p artifacts/slurm

echo "=== Running temporal evaluation (30 seeds) ==="
uv run python run_temporal.py

echo "=== Running spatial evaluation (30 seeds on ECE) ==="
uv run python run_spatial.py

echo "=== Running distance diagnostics ==="
uv run python analyze_cluster_distances.py

echo "=== Building report notebook ==="
uv run python build_notebook.py

echo "=== Executing report notebook ==="
cd ../..
nb execute experiment/derived_8.4-formal-eval-2.0-ece/derived_8.4-formal-eval-2.0-ece.ipynb --uv
cd experiment/derived_8.4-formal-eval-2.0-ece

echo "=== Updating README.md ==="
uv run python update_readme_from_notebook.py

echo "=== Complete ==="
