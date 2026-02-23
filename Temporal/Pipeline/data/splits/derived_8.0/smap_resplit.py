import os
import json
import pandas as pd

DATE_COL = "date"

def main():

    BASE = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits"

    IN_DIR  = os.path.join(BASE, "derived_7.0")
    OUT_DIR = os.path.join(BASE, "derived_8.0")

    os.makedirs(OUT_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(IN_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(IN_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(IN_DIR, "test.csv"))

    full = pd.concat([train_df, val_df, test_df], ignore_index=True)

    full[DATE_COL] = pd.to_datetime(full[DATE_COL], errors="coerce")
    full["year"] = full[DATE_COL].dt.year.astype(int)

    train_years = set(range(2017, 2021))  # 2017–2020
    val_years   = set(range(2021, 2023))  # 2021–2022
    test_years  = set(range(2023, 2026))  # 2023–2025

    train_new = full[full["year"].isin(train_years)].copy()
    val_new   = full[full["year"].isin(val_years)].copy()
    test_new  = full[full["year"].isin(test_years)].copy()

    for name, df_ in [("train", train_new), ("val", val_new), ("test", test_new)]:
        if len(df_) == 0:
            raise ValueError(f"{name} split is empty. Check year coverage.")

    print("Rows:")
    print("Train:", len(train_new))
    print("Val:", len(val_new))
    print("Test:", len(test_new))

    print("\nYear breakdown:")
    print(train_new["year"].value_counts().sort_index())
    print(val_new["year"].value_counts().sort_index())
    print(test_new["year"].value_counts().sort_index())

    train_new.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_new.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_new.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    meta = {
        "source_split": "derived_7.0",
        "note": "Post-2016 stable SMAP-era split",
        "train_years": sorted(list(train_years)),
        "val_years": sorted(list(val_years)),
        "test_years": sorted(list(test_years)),
        "rows": {
            "train": int(len(train_new)),
            "val": int(len(val_new)),
            "test": int(len(test_new)),
        }
    }

    with open(os.path.join(OUT_DIR, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved derived_8.0 successfully.")

if __name__ == "__main__":
    main()
