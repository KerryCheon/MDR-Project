import os
import sys
from pathlib import Path

# Set up project root path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Change current working directory to project root
# This ensures relative paths in config.yaml resolve correctly
os.chdir(project_root)

from Modeling.Src.soilmoist_fl.cli import run_feature_selection

def main():
    script_dir = Path("notebooks/experiment/derived_8.1-feature-selection")
    config_path = script_dir / "config.yaml"
    runs_dir = script_dir / "runs"
    
    print(f"Running feature selection with config: {config_path.resolve()}")
    print(f"Saving outputs to: {runs_dir.resolve()}")
    
    # Run the feature selection pipeline
    res = run_feature_selection(
        config_path=config_path,
        base_runs_dir=runs_dir,
    )
    print("Run completed successfully!")
    print(f"Results directory: {res['run_dir']}")
    print(f"Run ID: {res['run_id']}")

if __name__ == "__main__":
    main()
