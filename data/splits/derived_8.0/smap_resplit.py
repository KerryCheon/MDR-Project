import os
import json
import pandas as pd

DATE_COL = "date"
STATION_COL = "station_id"

LIA_CSV = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits/derived_8.0/LIA/stations_lia.csv"
LIA_COLS = [
    STATION_COL,
    "lia_mean_asc_deg",
    "lia_std_asc_deg",
    "lia_mean_desc_deg",
    "lia_std_desc_deg",
]

def main():
    BASE = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits"

    IN_DIR = os.path.join(BASE, "derived_7.0")
    OUT_DIR = os.path.join(BASE, "derived_8.0")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(IN_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(IN_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(IN_DIR, "test.csv"))

    full = pd.concat([train_df, val_df, test_df], ignore_index=True)

    if STATION_COL not in full.columns:
        raise ValueError(
            f"Expected '{STATION_COL}' column in split CSVs, but it was not found. "
            f"Columns: {list(full.columns)}"
        )
    if DATE_COL not in full.columns:
        raise ValueError(
            f"Expected '{DATE_COL}' column in split CSVs, but it was not found. "
            f"Columns: {list(full.columns)}"
        )

    full[DATE_COL] = pd.to_datetime(full[DATE_COL], errors="coerce")
    if full[DATE_COL].isna().any():
        bad = int(full[DATE_COL].isna().sum())
        raise ValueError(f"{bad} rows have invalid '{DATE_COL}' values. Fix before splitting.")
    full["year"] = full[DATE_COL].dt.year.astype(int)

    if not os.path.exists(LIA_CSV):
        raise FileNotFoundError(
            f"LIA CSV not found at: {LIA_CSV}\n"
            "Update LIA_CSV to the correct path."
        )

    lia = pd.read_csv(LIA_CSV)

    missing_cols = [c for c in LIA_COLS if c not in lia.columns]
    if missing_cols:
        raise ValueError(
            f"LIA CSV is missing required columns: {missing_cols}\n"
            f"Found columns: {list(lia.columns)}"
        )

    lia = lia[LIA_COLS].copy()
    lia[STATION_COL] = lia[STATION_COL].astype(str)
    full[STATION_COL] = full[STATION_COL].astype(str)

    dup = lia[STATION_COL].duplicated().sum()
    if dup:
        raise ValueError(f"LIA CSV has {int(dup)} duplicate station_id rows. It must be one row per station.")

    full = full.merge(lia, on=STATION_COL, how="left")

    miss_rate = float(full["lia_mean_asc_deg"].isna().mean())
    if miss_rate > 0:
        missing_stations = sorted(full.loc[full["lia_mean_asc_deg"].isna(), STATION_COL].unique().tolist())
        raise ValueError(
            f"LIA merge produced missing values (rate={miss_rate:.4%}).\n"
            f"Stations missing LIA: {missing_stations}\n"
            "Fix station_id mismatch between splits and stations_lia.csv."
        )

    train_years = set(range(2017, 2021))  # 2017–2020
    val_years = set(range(2021, 2023))    # 2021–2022
    test_years = set(range(2023, 2026))   # 2023–2025

    train_new = full[full["year"].isin(train_years)].copy()
    val_new = full[full["year"].isin(val_years)].copy()
    test_new = full[full["year"].isin(test_years)].copy()

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

    print("\nLIA columns added:")
    print([c for c in LIA_COLS if c != STATION_COL])

    train_new.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_new.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_new.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    meta = {
        "source_split": "derived_7.0",
        "note": "Post-2016 stable SMAP-era split + station-level pass-specific LIA features",
        "train_years": sorted(list(train_years)),
        "val_years": sorted(list(val_years)),
        "test_years": sorted(list(test_years)),
        "lia_csv": LIA_CSV,
        "lia_cols": [c for c in LIA_COLS if c != STATION_COL],
        "rows": {
            "train": int(len(train_new)),
            "val": int(len(val_new)),
            "test": int(len(test_new)),
        },
    }

    with open(os.path.join(OUT_DIR, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved derived_8.0 successfully.")


if __name__ == "__main__":
    main()
