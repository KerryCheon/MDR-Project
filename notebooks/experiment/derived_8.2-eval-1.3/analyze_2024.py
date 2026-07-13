import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path

# Paths
project_root = Path.cwd().resolve()
test_path = project_root / "data/splits/derived_8.2/test.csv"
metadata_path = project_root / "data/splits/derived_8.2/dataset_metadata.py"

# Load metadata
spec = importlib.util.spec_from_file_location("dataset_metadata", metadata_path)
dataset_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_metadata)

v0_features = dataset_metadata.OVERALL_SELECTED_FEATURES_V0
target_col = "soil_moisture_5cm"

# Load test data
test_df = pd.read_csv(test_path)
test_df["date"] = pd.to_datetime(test_df["date"])
test_df["year"] = test_df["date"].dt.year

print(f"Loaded test dataset with shape: {test_df.shape}")

# Calculate correlation of V0 features with soil moisture by year
years = [2023, 2024, 2025]
correlations = {}

for yr in years:
    df_yr = test_df[test_df["year"] == yr]
    corr_series = df_yr[v0_features].corrwith(df_yr[target_col])
    correlations[yr] = corr_series

corr_df = pd.DataFrame(correlations)
corr_df["Diff_2024_vs_2023"] = corr_df[2024] - corr_df[2023]
corr_df["Diff_2024_vs_2025"] = corr_df[2024] - corr_df[2025]
corr_df["Abs_2024"] = corr_df[2024].abs()

print("\n===== TOP CORRELATIONS WITH TARGET IN 2024 =====")
print(corr_df.sort_values(by="Abs_2024", ascending=False).head(15).to_string())

print("\n===== FEATURES WITH HIGHER CORRELATION IN 2024 THAN OTHER YEARS =====")
print(corr_df.sort_values(by="Diff_2024_vs_2023", ascending=False).head(10).to_string())
