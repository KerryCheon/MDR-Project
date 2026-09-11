"""ECE raw coverage audit.

Reads raw ECE CSVs (Seattle Time preferred), floors to calendar day,
and reports per-day/per-hour sample counts. No data are modified;
results are returned as DataFrames for the notebook to print and save.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ece_raw(
    filepath: Path,
    seattle_col: str = "Timestamp (Seattle Time)",
    utc_col: str = "Timestamp (UTC)",
    sm_col: str = "Soil Moisture (%)",
) -> pd.DataFrame:
    """Load one ECE raw file, preferring Seattle Time."""
    df_raw = pd.read_csv(filepath, skiprows=1)
    col_map = {str(c).strip(): c for c in df_raw.columns}
    time_col = col_map.get(seattle_col) or col_map.get(utc_col)
    value_col = col_map.get(sm_col)
    if time_col is None or value_col is None:
        raise ValueError(f"Missing time/sm columns in {filepath.name}: {list(df_raw.columns)}")
    out = pd.DataFrame(
        {
            "parsed_time": pd.to_datetime(df_raw[time_col], errors="coerce"),
            "sm_pct": pd.to_numeric(df_raw[value_col], errors="coerce"),
            "time_source": seattle_col if col_map.get(seattle_col) else utc_col,
        }
    ).dropna(subset=["parsed_time", "sm_pct"]).copy()
    out["date"] = out["parsed_time"].dt.floor("D")
    out["hour"] = out["parsed_time"].dt.hour
    return out


def daily_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day: n samples, n distinct hours, min/max time."""
    g = df.groupby("date")
    rows = []
    for date, sub in g:
        rows.append(
            {
                "date": pd.to_datetime(date),
                "n_samples": int(len(sub)),
                "n_hours": int(sub["hour"].nunique()),
                "first_time": sub["parsed_time"].min(),
                "last_time": sub["parsed_time"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def hourly_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Day x hour count matrix (24 columns)."""
    mat = df.groupby(["date", "hour"]).size().unstack(fill_value=0)
    for h in range(24):
        if h not in mat.columns:
            mat[h] = 0
    return mat[[h for h in range(24)]].sort_index()
