import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from Modeling.Src.soilmoist_fl.cli import run_feature_selection

def main():
    config_path = "notebooks/experiment/derived_8.0-optimization-1.0/config.yaml"
    print(f"Running feature selection with config: {config_path}")
    
    res = run_feature_selection(config_path=config_path)
    selected_feats = list(res["selected_features"])
    run_dir = res["run_dir"]
    
    print(f"\nPipeline finished. Selected {len(selected_feats)} features.")
    print("Selected features list:")
    print(selected_feats)
    
    out_dir = Path("notebooks/experiment/derived_8.0-optimization-1.0")
    out_path = out_dir / "selected_features.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected_feats, f, indent=2)
        
    print(f"Saved selected features to {out_path}")

if __name__ == "__main__":
    main()
