"""Fetch diagnostic reference weather (FAO ET0 + 2m temperature) for ECE sites.

Diagnostic-only context for the input-weather overlay figures: these series are
plotted alongside Best-1.1 predictions but are NEVER model inputs (no leakage,
no retraining). Source: Open-Meteo archive API (ERA5-based FAO Penman-Monteith
reference evapotranspiration + 2m air temperature).

Usage (from this directory):
    uv run --no-sync python fetch_ece_et_reference.py [--refresh]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "et0_fao_evapotranspiration",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
]
OUT_PATH = EXP_DIR / "ece_et_reference.csv"


def _station_coords() -> pd.DataFrame:
    ece_test = PROJECT_ROOT / "data/splits/derived_8.4_ece_v3/test.csv"
    df = pd.read_csv(
        ece_test, usecols=["station_id", "date", "longitude", "latitude"]
    )
    coords = (
        df.drop_duplicates("station_id")[["station_id", "longitude", "latitude"]]
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    window = (df["date"].min(), df["date"].max())
    return coords, window


def _fetch_station(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start,
        "end_date": end,
        "timezone": "auto",
        "daily": ",".join(DAILY_VARS),
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=60)
    data = r.json()
    if r.status_code != 200 or data.get("error"):
        raise RuntimeError(f"Open-Meteo error: {data.get('reason', r.status_code)}")
    daily = data.get("daily", {})
    missing = [v for v in DAILY_VARS if v not in daily] + (
        [] if "time" in daily else ["time"]
    )
    if missing:
        raise RuntimeError(f"Open-Meteo response missing fields: {missing}")
    frame = pd.DataFrame({k: daily[k] for k in ["time", *DAILY_VARS]})
    frame = frame.rename(
        columns={
            "time": "date",
            "et0_fao_evapotranspiration": "et0_fao_mm",
            "temperature_2m_max": "t2m_max_C",
            "temperature_2m_min": "t2m_min_C",
            "temperature_2m_mean": "t2m_mean_C",
        }
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if OUT_PATH.exists() and not args.refresh:
        cached = pd.read_csv(OUT_PATH)
        print(f"[ET] cache hit: {OUT_PATH} ({len(cached)} rows, use --refresh to refetch)")
        return
    coords, (start, end) = _station_coords()
    print(f"[ET] window {start} -> {end}, {len(coords)} stations")
    frames = []
    for row in coords.itertuples():
        frame = _fetch_station(row.latitude, row.longitude, start, end)
        frame.insert(0, "station_id", row.station_id)
        frames.append(frame)
        print(f"[ET] {row.station_id}: {len(frame)} days")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    expected = {(s, d) for s in coords["station_id"] for d in pd.date_range(start, end).strftime("%Y-%m-%d")}
    got = set(map(tuple, out[["station_id", "date"]].to_numpy()))
    if got != expected:
        raise RuntimeError(f"Coverage gap: {len(expected - got)} station-days missing")
    value_cols = ["et0_fao_mm", "t2m_max_C", "t2m_min_C", "t2m_mean_C"]
    if out[value_cols].isna().any().any():
        bad = out[out[value_cols].isna().any(axis=1)]
        raise RuntimeError(f"Missing reference values:\n{bad.to_string()}")
    out.to_csv(OUT_PATH, index=False)
    print(f"[ET] saved {OUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    sys.exit(main())
