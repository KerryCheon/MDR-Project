import os
import json
import shutil
import numpy as np
import pandas as pd

DATE_COL   = "date"
TARGET_COL = "soil_moisture_5cm"
GROUP_COL  = "station_id"

API_COL  = "G_API"
SMAP_COL = "SMAP_sm_pm_interp_ema02"

DRIFT_COLS = ["year_frac", "sin_year", "cos_year", "API_x_year", "SMAP_x_year"]

def add_drift_features(df, date_col=DATE_COL, ref_min_year=None, ref_max_year=None,
                       api_col=API_COL, smap_col=SMAP_COL):

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df[date_col].dt.year.astype(float)

    if ref_min_year is None:
        ref_min_year = float(df["year"].min())
    if ref_max_year is None:
        ref_max_year = float(df["year"].max())

    denom = (ref_max_year - ref_min_year) if (ref_max_year > ref_min_year) else 1.0
    df["year_frac"] = (df["year"] - ref_min_year) / denom

    theta = 2 * np.pi * df["year_frac"]
    df["sin_year"] = np.sin(theta)
    df["cos_year"] = np.cos(theta)

    if api_col in df.columns:
        df["API_x_year"] = df[api_col] * df["year_frac"]
    else:
        df["API_x_year"] = np.nan

    if smap_col in df.columns:
        df["SMAP_x_year"] = df[smap_col] * df["year_frac"]
    else:
        df["SMAP_x_year"] = np.nan

    return df, ref_min_year, ref_max_year


def main():
    SPLITS_BASE = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/splits"

    IN_DIR  = os.path.join(SPLITS_BASE, "derived_6.0")
    OUT_DIR = os.path.join(SPLITS_BASE, "derived_7.0")

    os.makedirs(OUT_DIR, exist_ok=True)

    train_path = os.path.join(IN_DIR, "train.csv")
    val_path   = os.path.join(IN_DIR, "val.csv")
    test_path  = os.path.join(IN_DIR, "test.csv")

    out_train_path = os.path.join(OUT_DIR, "train.csv")
    out_val_path   = os.path.join(OUT_DIR, "val.csv")
    out_test_path  = os.path.join(OUT_DIR, "test.csv")
    out_meta_path  = os.path.join(OUT_DIR, "split_meta.json")

    static_in = os.path.join(IN_DIR, "station_static_features.csv")
    static_cwd = os.path.join(os.getcwd(), "station_static_features.csv")
    static_out = os.path.join(OUT_DIR, "station_static_features.csv")

    tr = pd.read_csv(train_path, low_memory=False)
    va = pd.read_csv(val_path, low_memory=False)
    te = pd.read_csv(test_path, low_memory=False)

    full = pd.concat([tr, va, te], ignore_index=True)
    full[DATE_COL] = pd.to_datetime(full[DATE_COL], errors="coerce")
    if full[DATE_COL].isna().any():
        bad = int(full[DATE_COL].isna().sum())
        raise ValueError(f"{bad} rows have invalid dates in '{DATE_COL}'")

    full["year"] = full[DATE_COL].dt.year.astype(int)

    train_years = set(range(2014, 2020))
    val_years   = set(range(2020, 2023))
    test_years  = set(range(2023, 2026))

    train_df = full[full["year"].isin(train_years)].copy()
    val_df   = full[full["year"].isin(val_years)].copy()
    test_df  = full[full["year"].isin(test_years)].copy()

    for name, df_ in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(df_) == 0:
            raise ValueError(f"{name} split is empty. Check year ranges vs data coverage.")

    train_df_d, y0, y1 = add_drift_features(train_df, ref_min_year=None, ref_max_year=None)
    val_df_d, _, _     = add_drift_features(val_df,   ref_min_year=y0, ref_max_year=y1)
    test_df_d, _, _    = add_drift_features(test_df,  ref_min_year=y0, ref_max_year=y1)

    missing_api = float(train_df_d["API_x_year"].isna().mean())
    missing_smap = float(train_df_d["SMAP_x_year"].isna().mean())

    print("Added drift features:", DRIFT_COLS)
    print(f"Train year range for scaling: {y0} to {y1}")
    print("Missing rate (train) API_x_year:", missing_api)
    print("Missing rate (train) SMAP_x_year:", missing_smap)

    train_df_d.to_csv(out_train_path, index=False)
    val_df_d.to_csv(out_val_path, index=False)
    test_df_d.to_csv(out_test_path, index=False)

    if os.path.exists(static_in):
        shutil.copyfile(static_in, static_out)
    elif os.path.exists(static_cwd):
        shutil.copyfile(static_cwd, static_out)
    else:
        print("[WARN] station_static_features.csv not found in IN_DIR or CWD, skipping copy.")

    meta = {
        "source_split": "derived_6.0",
        "derived_features": "reused (no recompute), plus drift features added",
        "year_split": {
            "train_years": sorted(list(train_years)),
            "val_years": sorted(list(val_years)),
            "test_years": sorted(list(test_years)),
        },
        "drift_cols_added": DRIFT_COLS,
        "drift_year_scaling_train_minmax": [float(y0), float(y1)],
        "required_for_drift": {"api_col": API_COL, "smap_col": SMAP_COL},
        "rows": {"train": int(len(train_df_d)), "val": int(len(val_df_d)), "test": int(len(test_df_d))},
    }

    with open(out_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved derived_7.0 splits to:")
    print(" ", out_train_path)
    print(" ", out_val_path)
    print(" ", out_test_path)
    print("Meta:")
    print(" ", out_meta_path)


if __name__ == "__main__":
    main()
