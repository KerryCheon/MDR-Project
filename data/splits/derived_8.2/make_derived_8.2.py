# Compiles derived_8.2 from derived_8.1_pos by removing MFNooksack_WA_1011 station.

import os
import json
import pandas as pd

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, "..", "derived_8.1_pos"))
    
    print(f"Source split directory: {src_dir}")
    print(f"Target split directory: {script_dir}")
    
    # Create target directories
    os.makedirs(os.path.join(script_dir, "LIA"), exist_ok=True)
    
    # 1. Filter CSV files
    csv_files = [
        ("train.csv", "station_id"),
        ("val.csv", "station_id"),
        ("test.csv", "station_id"),
        ("station_static_features.csv", "station_id"),
        ("LIA/stations.csv", "station_id"),
        ("LIA/stations_lia.csv", "station_id"),
    ]
    
    for rel_path, col_name in csv_files:
        src_path = os.path.join(src_dir, rel_path)
        dest_path = os.path.join(script_dir, rel_path)
        
        print(f"Filtering {rel_path}...")
        df = pd.read_csv(src_path)
        initial_len = len(df)
        df_filtered = df[df[col_name] != "MFNooksack_WA_1011"].copy()
        filtered_len = len(df_filtered)
        
        print(f"  {rel_path}: {initial_len} -> {filtered_len} rows (dropped {initial_len - filtered_len} rows)")
        df_filtered.to_csv(dest_path, index=False)
        
    # 2. Filter config.yaml
    src_config_path = os.path.join(src_dir, "config.yaml")
    dest_config_path = os.path.join(script_dir, "config.yaml")
    
    print("Filtering config.yaml...")
    with open(src_config_path, "r", encoding="utf-8") as f:
        config_lines = f.readlines()
        
    start_idx = None
    for i, line in enumerate(config_lines):
        if line.strip() == "mf_nooksack_wa:":
            start_idx = i
            break
            
    if start_idx is not None:
        comment_indices = []
        curr = start_idx - 1
        while curr >= 0 and config_lines[curr].strip().startswith("#"):
            comment_indices.append(curr)
            curr -= 1
        if comment_indices:
            start_idx = min(comment_indices)
            
        end_idx = None
        for j in range(start_idx + 5, len(config_lines)):
            if config_lines[j].strip() == "# -----------------------------------------------------------":
                end_idx = j
                break
        if end_idx is None:
            end_idx = len(config_lines)
            
        print(f"  Removing config block from line {start_idx+1} to {end_idx}")
        new_config_lines = config_lines[:start_idx] + config_lines[end_idx:]
    else:
        print("  WARNING: mf_nooksack_wa not found in config.yaml")
        new_config_lines = config_lines
        
    with open(dest_config_path, "w", encoding="utf-8") as f:
        f.writelines(new_config_lines)
        
    # 3. Filter split_meta.json
    src_meta_path = os.path.join(src_dir, "split_meta.json")
    dest_meta_path = os.path.join(script_dir, "split_meta.json")
    
    print("Updating split_meta.json...")
    with open(src_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    # Remove station from list
    if "MFNooksack_WA_1011" in meta.get("stations", []):
        meta["stations"].remove("MFNooksack_WA_1011")
        print("  Removed MFNooksack_WA_1011 from station list")
    
    # Update row counts
    train_filtered = pd.read_csv(os.path.join(script_dir, "train.csv"))
    val_filtered = pd.read_csv(os.path.join(script_dir, "val.csv"))
    test_filtered = pd.read_csv(os.path.join(script_dir, "test.csv"))
    
    meta["source"] = "derived_8.1_pos splits with MFNooksack_WA_1011 filtered out"
    meta["rows"] = {
        "train": len(train_filtered),
        "val": len(val_filtered),
        "test": len(test_filtered)
    }
    meta["lia_csv"] = "data/splits/derived_8.2/LIA/stations_lia.csv"
    
    with open(dest_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    # 4. Create blank dataset_metadata.py
    dest_meta_py_path = os.path.join(script_dir, "dataset_metadata.py")
    print("Creating blank dataset_metadata.py...")
    with open(dest_meta_py_path, "w", encoding="utf-8") as f:
        f.write("# Blank metadata file for derived_8.2\n")
        
    print("Dataset compilation for derived_8.2 completed successfully.")

if __name__ == "__main__":
    main()
