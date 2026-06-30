# Compiles derived_8.1_pos by filtering out soil moisture values <= 0.0 from derived_8.1.

import os
import json
import pandas as pd

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    d81_dir = os.path.abspath(os.path.join(script_dir, "..", "derived_8.1"))
    
    print(f"Loading derived_8.1 splits from: {d81_dir}")
    train = pd.read_csv(os.path.join(d81_dir, "train.csv"))
    val = pd.read_csv(os.path.join(d81_dir, "val.csv"))
    test = pd.read_csv(os.path.join(d81_dir, "test.csv"))
    
    def filter_zeros(df, split_name):
        initial_len = len(df)
        # Drop rows where target is <= 0.0 or NaN
        filtered_df = df[df["soil_moisture_5cm"] > 0.0].copy()
        filtered_len = len(filtered_df)
        print(f"  {split_name}: {initial_len} -> {filtered_len} rows (dropped {initial_len - filtered_len} rows with target <= 0.0)")
        return filtered_df
        
    print("Filtering target = 0.0 values...")
    train_pos = filter_zeros(train, "train")
    val_pos = filter_zeros(val, "val")
    test_pos = filter_zeros(test, "test")
    
    # Save outputs
    print(f"Saving new splits to: {script_dir}")
    train_pos.to_csv(os.path.join(script_dir, "train.csv"), index=False)
    val_pos.to_csv(os.path.join(script_dir, "val.csv"), index=False)
    test_pos.to_csv(os.path.join(script_dir, "test.csv"), index=False)
    
    # Load original split_meta.json
    with open(os.path.join(d81_dir, "split_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    # Update split_meta.json
    meta["source"] = "derived_8.1 splits with soil_moisture_5cm <= 0.0 filtered out"
    meta["rows"] = {
        "train": len(train_pos),
        "val": len(val_pos),
        "test": len(test_pos)
    }
    meta["lia_csv"] = "data/splits/derived_8.1_pos/LIA/stations_lia.csv"
    
    with open(os.path.join(script_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print("Dataset compilation for derived_8.1_pos completed successfully.")

if __name__ == "__main__":
    main()
