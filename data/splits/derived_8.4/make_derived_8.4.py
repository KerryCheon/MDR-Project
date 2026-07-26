# Compiles derived_8.4 from derived_8.3 by removing MartenRidge_WA_999 and RainyPass_WA_711.

import os
import json
import pandas as pd

STATIONS_TO_REMOVE = ["MartenRidge_WA_999", "RainyPass_WA_711"]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, "..", "derived_8.3"))

    print(f"Source split directory: {src_dir}")
    print(f"Target split directory: {script_dir}")

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
        df_filtered = df[~df[col_name].isin(STATIONS_TO_REMOVE)].copy()
        filtered_len = len(df_filtered)

        print(f"  {rel_path}: {initial_len} -> {filtered_len} rows (dropped {initial_len - filtered_len} rows)")
        df_filtered.to_csv(dest_path, index=False)

    # 2. Copy config.yaml as-is
    src_config_path = os.path.join(src_dir, "config.yaml")
    dest_config_path = os.path.join(script_dir, "config.yaml")

    print("Copying config.yaml...")
    with open(src_config_path, "r", encoding="utf-8") as f:
        content = f.read()
    with open(dest_config_path, "w", encoding="utf-8") as f:
        f.write(content)

    # 3. Update split_meta.json
    src_meta_path = os.path.join(src_dir, "split_meta.json")
    dest_meta_path = os.path.join(script_dir, "split_meta.json")

    print("Updating split_meta.json...")
    with open(src_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for station in STATIONS_TO_REMOVE:
        if station in meta.get("stations", []):
            meta["stations"].remove(station)
            print(f"  Removed {station} from station list")

    train_filtered = pd.read_csv(os.path.join(script_dir, "train.csv"))
    val_filtered = pd.read_csv(os.path.join(script_dir, "val.csv"))
    test_filtered = pd.read_csv(os.path.join(script_dir, "test.csv"))

    meta["source"] = "derived_8.3 splits with MartenRidge_WA_999 and RainyPass_WA_711 filtered out"
    meta["rows"] = {
        "train": len(train_filtered),
        "val": len(val_filtered),
        "test": len(test_filtered)
    }
    meta["lia_csv"] = "data/splits/derived_8.4/LIA/stations_lia.csv"

    with open(dest_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 4. Copy dataset_metadata.py
    src_meta_py = os.path.join(src_dir, "dataset_metadata.py")
    dest_meta_py = os.path.join(script_dir, "dataset_metadata.py")

    print("Copying dataset_metadata.py...")
    with open(src_meta_py, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace("derived_8.3", "derived_8.4")

    with open(dest_meta_py, "w", encoding="utf-8") as f:
        f.write(content)

    print("Dataset compilation for derived_8.4 completed successfully.")

if __name__ == "__main__":
    main()
