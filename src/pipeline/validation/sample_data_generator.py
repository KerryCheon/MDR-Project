# sample_data_generator.py
# Extracts baseline reference samples from processed datasets for regression testing.

import json
from pathlib import Path
import pandas as pd


def extract_reference_samples(
    csv_path: Path,
    out_json_path: Path,
    station_name: str = None,
    n_samples: int = 5
) -> dict:
    """Extracts reference weekly satellite samples from an existing final.csv file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")

    resolved_station = station_name or csv_path.parent.name

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    sat_columns = [
        "LST_modis", "NDVI_modis", "s1_vv", "s1_vh", "s1_vv_dB", "s1_vh_dB",
        "s2_b2", "s2_b3", "s2_b4", "s2_b8", "s2_b11", "s2_b12",
        "elev", "slope", "aspect",
        "SMAP_sm_am", "SMAP_sm_pm", "SMAP_qual_am", "SMAP_qual_pm"
    ]

    # Select columns that exist in the CSV
    available_sat_cols = [c for c in sat_columns if c in df.columns]

    df["week"] = df["date"].dt.to_period("W-SUN").astype(str)
    grouped = df.groupby("week")

    samples = {
        "station_name": resolved_station,
        "latitude": float(df["latitude"].median()),
        "longitude": float(df["longitude"].median()),
        "weeks": {}
    }

    count = 0
    for week, group in grouped:
        if count >= n_samples:
            break
        start = group["date"].min().strftime("%Y-%m-%d")
        end = (group["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        date_key = f"{start}_{end}"

        # Take first row of the group for satellite feature reference
        row = group.iloc[0]
        sat_data = {}
        for col in available_sat_cols:
            val = row[col]
            sat_data[col] = float(val) if pd.notna(val) else None

        samples["weeks"][date_key] = {
            "week": str(week),
            "start": start,
            "end": end,
            "expected_features": sat_data
        }
        count += 1

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(samples, f, indent=2)

    return samples


if __name__ == "__main__":
    src_csv = Path("src/pipeline/data/processed/quinault/final.csv")
    out_fixture = Path("tests/fixtures/satellite_regression_samples.json")
    if src_csv.exists():
        extract_reference_samples(src_csv, out_fixture, n_samples=10)
        print(f"Generated fixture at {out_fixture}")
    else:
        print(f"Source file {src_csv} not found.")
