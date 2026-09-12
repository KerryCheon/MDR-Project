"""Fetch hourly rainfall reference for ECE sites (Open-Meteo archive, ERA5-based).

Diagnostic-only context for `ece-input-diagnose-1.0`: the pipeline's
`weather_pipe.py` fetches these same hourly `rain,precipitation` series but
discards them after `resample("D").sum()`. This script keeps the hourly frame
so rain timing within the day can be compared against the hourly sensor means.
These series are NEVER model inputs (no leakage, no retraining).

Usage (from the notebooks/ directory):
    uv run python experiment/ece-input-diagnose-1.0/fetch_hourly_rain.py [--refresh]
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
HOURLY_VARS = ["rain", "precipitation"]
OUT_PATH = EXP_DIR / "hourly_rain_reference.csv"
START = "2026-07-19"
END = "2026-08-20"


def _station_coords() -> pd.DataFrame:
    ece_test = PROJECT_ROOT / "data/splits/derived_8.4_ece_v3/test.csv"
    df = pd.read_csv(ece_test, usecols=["station_id", "longitude", "latitude"])
    return (
        df.drop_duplicates("station_id")[["station_id", "longitude", "latitude"]]
        .sort_values("station_id")
        .reset_index(drop=True)
    )


def _fetch_station(lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": START,
        "end_date": END,
        "timezone": "auto",
        "hourly": ",".join(HOURLY_VARS),
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=60)
    data = r.json()
    if r.status_code != 200 or data.get("error"):
        raise RuntimeError(f"Open-Meteo error: {data.get('reason', r.status_code)}")
    hourly = data.get("hourly", {})
    missing = [v for v in HOURLY_VARS if v not in hourly] + (
        [] if "time" in hourly else ["time"]
    )
    if missing:
        raise RuntimeError(f"Open-Meteo response missing fields: {missing}")
    frame = pd.DataFrame({k: hourly[k] for k in ["time", *HOURLY_VARS]})
    return frame.rename(
        columns={"time": "timestamp", "rain": "rain_mm", "precipitation": "precip_mm"}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if OUT_PATH.exists() and not args.refresh:
        cached = pd.read_csv(OUT_PATH)
        print(f"[RAIN] cache hit: {OUT_PATH} ({len(cached)} rows, use --refresh to refetch)")
        return
    coords = _station_coords()
    print(f"[RAIN] window {START} -> {END}, {len(coords)} stations")
    frames = []
    for row in coords.itertuples():
        frame = _fetch_station(row.latitude, row.longitude)
        frame.insert(0, "station_id", row.station_id)
        frames.append(frame)
        print(f"[RAIN] {row.station_id}: {len(frame)} hours "
              f"{frame['timestamp'].iloc[0]}..{frame['timestamp'].iloc[-1]}")
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    expected_idx = pd.date_range(f"{START} 00:00", f"{END} 23:00", freq="h")
    for station, sub in out.groupby("station_id"):
        missing_ts = set(expected_idx) - set(sub["timestamp"])
        if missing_ts:
            raise RuntimeError(f"{station}: {len(missing_ts)} hourly timestamps missing")
    if out[["rain_mm", "precip_mm"]].isna().any().any():
        raise RuntimeError("Missing rainfall values in response")
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    out.to_csv(OUT_PATH, index=False)
    print(f"[RAIN] saved {OUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    sys.exit(main())
