#!/usr/bin/env python3
"""Quality Audit Script for derived_8.4_ece_v3.

Verifies:
1. Shape & Schema (150 rows, 499 columns matching derived_8.4)
2. Per-station row counts (30 rows each for all 5 stations)
3. Date range strictly spanning July 20 – August 19, 2026
4. Target column physical sanity (soil_moisture_5cm > 0.0, zero NaNs)
5. Strict Native-Missing SMAP Policy (0 zero-filled values in 82 SMAP value cols; 3 SMAP masks all 0)
6. Warmup Continuity (30-day rolling features are populated on Aug 18 AND Aug 19, preventing boundary drop)
7. Optical & SAR feature coverage (F_NDVI, SAR_ratio, etc. valid)

Usage:
    PYTHONPATH=. uv run python data/splits/derived_8.4_ece_v3/quality_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SPLIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SPLIT_DIR.parents[2]

REF_SPLIT_PATH = PROJECT_ROOT / "data/splits/derived_8.4/train.csv"
TEST_SPLIT_PATH = SPLIT_DIR / "test.csv"
EVAL_SPLIT_PATH = SPLIT_DIR / "eval.csv"

EXPECTED_STATIONS = [
    "ECE_BBG_Lost_Meadow",
    "ECE_BBG_Main_St",
    "ECE_Renton_Garden_North",
    "ECE_Renton_Garden_Shed",
    "ECE_Renton_Home",
]


def run_audit() -> bool:
    print("=" * 70)
    print("           derived_8.4_ece_v3 DATASET QUALITY AUDIT")
    print("=" * 70)

    if not TEST_SPLIT_PATH.exists():
        print(f"[FAIL] test.csv not found at {TEST_SPLIT_PATH}")
        return False

    df = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)
    ref_df = pd.read_csv(REF_SPLIT_PATH, nrows=5)

    all_passed = True

    # ── 1. Shape & Schema Check ──────────────────────────────────────────────
    print("\n--- 1. SHAPE & SCHEMA PARITY ---")
    if len(df) == 150:
        print(f"[PASS] Row count is exactly 150.")
    else:
        print(f"[FAIL] Expected 150 rows, got {len(df)}.")
        all_passed = False

    if len(df.columns) == 499:
        print(f"[PASS] Column count is exactly 499.")
    else:
        print(f"[FAIL] Expected 499 columns, got {len(df.columns)}.")
        all_passed = False

    col_order_match = list(df.columns) == list(ref_df.columns)
    if col_order_match:
        print(f"[PASS] Column names and ordering 100% identical to derived_8.4 reference.")
    else:
        diff = set(ref_df.columns) ^ set(df.columns)
        print(f"[FAIL] Column mismatch with derived_8.4 reference! Symmetric difference: {diff}")
        all_passed = False

    # ── 2. Station Distribution Check ─────────────────────────────────────────
    print("\n--- 2. STATION DISTRIBUTION ---")
    stations = sorted(df["station_id"].unique().tolist())
    if stations == sorted(EXPECTED_STATIONS):
        print(f"[PASS] Station set matches expected 5 in-situ ECE stations.")
    else:
        print(f"[FAIL] Station mismatch! Expected {EXPECTED_STATIONS}, got {stations}")
        all_passed = False

    counts = df["station_id"].value_counts().to_dict()
    for st in EXPECTED_STATIONS:
        cnt = counts.get(st, 0)
        if cnt == 30:
            print(f"  [PASS] {st:<28}: exactly 30 evaluation days.")
        else:
            print(f"  [FAIL] {st:<28}: expected 30 days, got {cnt}.")
            all_passed = False

    # ── 3. Temporal Coverage Check ────────────────────────────────────────────
    print("\n--- 3. TEMPORAL BOUNDS ---")
    dates = pd.to_datetime(df["date"])
    min_date = dates.min().strftime("%Y-%m-%d")
    max_date = dates.max().strftime("%Y-%m-%d")
    if min_date == "2026-07-20" and max_date == "2026-08-19":
        print(f"[PASS] Date span is strictly {min_date} to {max_date} (no partial edge days).")
    else:
        print(f"[FAIL] Unexpected date span: {min_date} to {max_date}.")
        all_passed = False

    # ── 4. Target Variable Check ──────────────────────────────────────────────
    print("\n--- 4. TARGET SANITY (soil_moisture_5cm) ---")
    sm = df["soil_moisture_5cm"]
    n_nan = sm.isna().sum()
    n_le_zero = (sm <= 0.0).sum()
    if n_nan == 0 and n_le_zero == 0:
        print(f"[PASS] Ground truth target has 0 NaNs and all values > 0.0 (min: {sm.min():.4f}, max: {sm.max():.4f}).")
    else:
        print(f"[FAIL] Target issues found: {n_nan} NaNs, {n_le_zero} values <= 0.0.")
        all_passed = False

    # ── 5. Strict Native-Missing SMAP Policy ──────────────────────────────────
    print("\n--- 5. STRICT SMAP NATIVE-MISSING AUDIT ---")
    smap_cols = [c for c in df.columns if "SMAP" in c]
    mask_cols = [c for c in smap_cols if c.endswith("_mask")]
    val_cols = [c for c in smap_cols if c not in mask_cols]

    print(f"  Total SMAP features: {len(smap_cols)} ({len(val_cols)} value columns, {len(mask_cols)} mask columns)")

    # Ensure 0 zero-fill in value columns
    zero_counts = (df[val_cols] == 0.0).sum().sum()
    if zero_counts == 0:
        print(f"[PASS] Zero spurious zero-fill values across all {len(val_cols)} SMAP value columns.")
    else:
        print(f"[FAIL] Found {zero_counts} physical 0.0 values in SMAP value columns! Must be native NaN.")
        all_passed = False

    # Ensure all values are NaN
    nan_counts = df[val_cols].isna().sum().sum()
    expected_nans = len(df) * len(val_cols)
    if nan_counts == expected_nans:
        print(f"[PASS] All {len(val_cols)} SMAP value columns are 100% native NaN (prevents tree model bias).")
    else:
        print(f"[FAIL] Expected {expected_nans} NaNs, got {nan_counts}.")
        all_passed = False

    # Ensure masks are all 0
    mask_sum = df[mask_cols].sum().sum()
    if mask_sum == 0:
        print(f"[PASS] All SMAP observation mask columns are strictly 0.")
    else:
        print(f"[FAIL] SMAP observation masks contain non-zero values (sum={mask_sum}).")
        all_passed = False

    # ── 6. Rolling Warmup Continuity Audit ─────────────────────────────────────
    print("\n--- 6. ROLLING WARMUP CONTINUITY (AUG 18 vs AUG 19) ---")
    # Verify that V_rollmin_G_API_kobs30 is populated on BOTH Aug 18 and Aug 19!
    api_kobs30_col = "V_rollmin_G_API_kobs30"
    if api_kobs30_col in df.columns:
        aug18 = df[df["date"] == "2026-08-18"][["station_id", api_kobs30_col]]
        aug19 = df[df["date"] == "2026-08-19"][["station_id", api_kobs30_col]]

        aug18_nans = aug18[api_kobs30_col].isna().sum()
        aug19_nans = aug19[api_kobs30_col].isna().sum()

        print(f"  {api_kobs30_col} on 2026-08-18: {5 - aug18_nans}/5 stations populated (NaNs: {aug18_nans})")
        print(f"  {api_kobs30_col} on 2026-08-19: {5 - aug19_nans}/5 stations populated (NaNs: {aug19_nans})")

        if aug18_nans == 0 and aug19_nans == 0:
            print(f"[PASS] Warmup scaffold eliminated boundary artifact: {api_kobs30_col} is continuous on both Aug 18 & 19!")
        else:
            print(f"[FAIL] Boundary artifact remains! Aug 18 NaNs: {aug18_nans}, Aug 19 NaNs: {aug19_nans}.")
            all_passed = False
    else:
        print(f"[WARN] {api_kobs30_col} not found in columns.")

    # ── 7. Remote Sensing & Derived Optical Features ───────────────────────────
    print("\n--- 7. SATELLITE & DERIVED FEATURES COVERAGE ---")
    for feat in ["F_NDVI", "F_NDMI", "F_MSI", "E_SAR_ratio", "LST_modis"]:
        if feat in df.columns:
            finite_rate = df[feat].notna().mean()
            print(f"  {feat:<16}: {finite_rate:.1%} populated (mean: {df[feat].mean():.4f})")
            if finite_rate < 0.90:
                print(f"[WARN] {feat} coverage is lower than 90%: {finite_rate:.1%}")
        else:
            print(f"[FAIL] Missing required derived feature: {feat}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print(">>> ALL AUDIT CHECKS PASSED: derived_8.4_ece_v3 is verified ready for modeling! <<<")
    else:
        print(">>> AUDIT FAILED: Please review failed checks above before using split. <<<")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
