import os
import sys
from pathlib import Path

# Set up project root path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from Modeling.Src.soilmoist_fl.cli import run_feature_selection

def main():
    config_path = "notebooks/experiment/derived_8.2-feature-selection/config_v4.yaml"
    print(f"Running feature selection with config: {config_path}")
    
    # Run the feature selection pipeline
    res = run_feature_selection(config_path=config_path)
    selected_feats = res["selected_features"]
    run_dir = res["run_dir"]
    
    print(f"\nPipeline finished. Selected {len(selected_feats)} features.")
    print(f"Run directory: {run_dir}")
    
    dest_path = Path("data/splits/derived_8.2/dataset_metadata.py")
    
    # Read the existing content
    if dest_path.exists():
        with open(dest_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = ""
        
    v4_str = f"OVERALL_SELECTED_FEATURES_V4 = {repr(list(selected_feats))}"
    
    # Process file lines to update or append OVERALL_SELECTED_FEATURES_V4
    lines = content.splitlines()
    new_lines = []
    has_v4 = False
    for line in lines:
        if line.startswith("OVERALL_SELECTED_FEATURES_V4"):
            new_lines.append(v4_str)
            has_v4 = True
        else:
            new_lines.append(line)
            
    if not has_v4:
        new_lines.append("")
        new_lines.append("# Selected features for overall global model - V4 (no MI, expanded ElasticNet)")
        new_lines.append(v4_str)
        
    updated_content = "\n".join(new_lines) + "\n"
    
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(updated_content)
        
    print(f"Successfully updated {dest_path} with OVERALL_SELECTED_FEATURES_V4!")

if __name__ == "__main__":
    main()
