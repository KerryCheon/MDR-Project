from pathlib import Path
import pandas as pd
import numpy as np

SPLIT_DIR = Path("data/splits/derived_8.4_ece_v3").resolve()
REF_PATH = Path("data/splits/derived_8.4/train.csv").resolve()


def test_derived_8_4_ece_v3_shape_and_schema():
    assert (SPLIT_DIR / "test.csv").exists(), "test.csv must exist"
    assert (SPLIT_DIR / "eval.csv").exists(), "eval.csv must exist"
    assert (SPLIT_DIR / "split_meta.json").exists(), "split_meta.json must exist"

    df = pd.read_csv(SPLIT_DIR / "test.csv", low_memory=False)
    ref_df = pd.read_csv(REF_PATH, nrows=5)

    assert len(df) == 150, f"Expected 150 rows, got {len(df)}"
    assert len(df.columns) == 499, f"Expected 499 columns, got {len(df.columns)}"
    assert list(df.columns) == list(ref_df.columns), "Column schema and ordering must match derived_8.4 exactly"

    # Verify 30 rows per station
    counts = df["station_id"].value_counts().to_dict()
    assert len(counts) == 5
    assert all(c == 30 for c in counts.values())

    # Verify date range
    dates = pd.to_datetime(df["date"])
    assert dates.min() == pd.Timestamp("2026-07-20")
    assert dates.max() == pd.Timestamp("2026-08-19")


def test_derived_8_4_ece_v3_smap_native_missing():
    df = pd.read_csv(SPLIT_DIR / "test.csv", low_memory=False)

    smap_cols = [c for c in df.columns if "SMAP" in c]
    mask_cols = [c for c in smap_cols if c.endswith("_mask")]
    val_cols = [c for c in smap_cols if c not in mask_cols]

    assert len(smap_cols) == 85
    assert len(mask_cols) == 3
    assert len(val_cols) == 82

    # Zero values in SMAP value columns are strictly forbidden
    assert (df[val_cols] == 0.0).sum().sum() == 0, "No spurious 0.0 values allowed in SMAP value columns"
    # All values must be NaN
    assert df[val_cols].isna().all().all(), "All SMAP value columns must be native NaN"
    # All masks must be 0
    assert df[mask_cols].eq(0).all().all(), "All SMAP observation masks must be 0"


def test_derived_8_4_ece_v3_warmup_continuity():
    df = pd.read_csv(SPLIT_DIR / "test.csv", low_memory=False)

    # 30-day rolling minimum of API must be populated on both Aug 18 and Aug 19
    col = "V_rollmin_G_API_kobs30"
    assert col in df.columns

    aug18 = df[df["date"] == "2026-08-18"][col].values
    aug19 = df[df["date"] == "2026-08-19"][col].values

    assert np.all(~np.isnan(aug18)), f"{col} must be non-NaN on Aug 18 (warmup working)"
    assert np.all(~np.isnan(aug19)), f"{col} must be non-NaN on Aug 19 (no boundary drop)"
    assert np.all(aug18 > 0.0), f"{col} on Aug 18 must be physically positive"
    assert np.all(aug19 > 0.0), f"{col} on Aug 19 must be physically positive"
    # Verify smooth transition (ratio within physical decay factor ~0.90)
    decay_ratios = aug19 / aug18
    assert np.all(decay_ratios > 0.85) and np.all(decay_ratios < 1.05), "Daily decay must be continuous"
