import pandas as pd
import numpy as np
from pathlib import Path

# Paths
project_root = Path.cwd().resolve()
test_path = project_root / "data/splits/derived_8.2/test.csv"

# Load test data
test_df = pd.read_csv(test_path)
test_df["date"] = pd.to_datetime(test_df["date"])
test_df["year"] = test_df["date"].dt.year

print("===== TEST SET STATS BY YEAR =====")
for yr in [2023, 2024, 2025]:
    df_yr = test_df[test_df["year"] == yr]
    print(f"\n--- Year {yr} (N={len(df_yr)}) ---")
    print(f"  Target soil moisture mean: {df_yr['soil_moisture_5cm'].mean():.4f}, std: {df_yr['soil_moisture_5cm'].std():.4f}")
    print(f"  LST_modis mean:            {df_yr['LST_modis'].mean():.4f}, std: {df_yr['LST_modis'].std():.4f}")
    print(f"  SMAP_sm_pm_interp_lag1 mean: {df_yr['SMAP_sm_pm_interp_lag1'].mean():.4f}, std: {df_yr['SMAP_sm_pm_interp_lag1'].std():.4f}")
    print(f"  G_API mean:                {df_yr['G_API'].mean():.4f}, std: {df_yr['G_API'].std():.4f}")
