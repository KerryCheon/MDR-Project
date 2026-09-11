"""SNOTEL hourly vs daily check.

Parses one sample .stm file (hourly rows), reproduces the SNOTEL pipe's
cross-sensor averaging for one timestamp, and compares candidate daily
reductions: daily mean of hourly values vs first-hour snapshot (what
MergePipe drop_duplicates on ["station_id","date"] would retain).
"""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path

import pandas as pd


def load_stm_hourly(filepath: Path) -> pd.DataFrame:
    """Parse .stm hourly rows: Date, Time, Value."""
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if re.match(r"^\d{4}/\d{2}/\d{2}", line):
                parts = line.split()
                if len(parts) >= 3:
                    rows.append(parts[:3])
    df = pd.DataFrame(rows, columns=["Date", "Time", "Value"])
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
    df["date"] = df["DateTime"].dt.floor("D")
    return df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)


def daily_reductions(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily mean vs first-hour snapshot per calendar day."""
    g = hourly.groupby("date")["Value"]
    out = pd.DataFrame(
        {
            "date": [pd.to_datetime(d) for d in g.mean().index],
            "n_hours": g.size().values,
            "daily_mean": g.mean().values,
            "first_hour": g.first().values,
        }
    )
    out["mean_minus_first"] = out["daily_mean"] - out["first_hour"]
    return out.sort_values("date").reset_index(drop=True)


def find_sample_stm(raw_dir: Path, depth: str = "0.050800") -> Path | None:
    """Find one 5cm .stm file in a SNOTEL raw dir."""
    cands = sorted(glob(str(raw_dir / f"*sm_{depth}_*.stm")))
    return Path(cands[0]) if cands else None
