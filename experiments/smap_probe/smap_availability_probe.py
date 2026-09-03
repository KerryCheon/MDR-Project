#!/usr/bin/env python3
"""SMAP GEE Availability Probe for ECE Evaluation Window.

Queries NASA/SMAP/SPL3SMP_E/005+006 for daily soil moisture availability
across the warmup (Jun 20 - Jul 19) and evaluation (Jul 20 - Aug 19, 2026)
periods for one representative ECE station (ECE_BBG_Main_St).

Usage (from project root):
    PYTHONPATH=. uv run python experiments/smap_probe/smap_availability_probe.py

Output:
    Per-week table of SMAP_sm_am / SMAP_sm_pm mean values or None.
    Saved to: experiments/smap_probe/results.json
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ee
from src.pipeline.utils.gee import initialize_ee
from src.pipeline.utils.logger import get_logger

# -- Station parameters (ECE_BBG_Main_St -- representative probe point) --------
LAT      = 47.6098164
LON      = -122.1824678
BUFFER_M = 1000  # same as pipeline

SMAP_005 = "NASA/SMAP/SPL3SMP_E/005"
SMAP_006 = "NASA/SMAP/SPL3SMP_E/006"

# -- Date windows to probe (weekly, matching pipeline batching logic) -----------
PROBE_WEEKS = [
    # (label, start, end)
    ("warmup_w1", "2026-06-20", "2026-06-27"),
    ("warmup_w2", "2026-06-27", "2026-07-04"),
    ("warmup_w3", "2026-07-04", "2026-07-11"),
    ("warmup_w4", "2026-07-11", "2026-07-18"),
    ("warmup_w5", "2026-07-18", "2026-07-20"),
    ("eval_w1",   "2026-07-20", "2026-07-27"),
    ("eval_w2",   "2026-07-27", "2026-08-03"),
    ("eval_w3",   "2026-08-03", "2026-08-10"),
    ("eval_w4",   "2026-08-10", "2026-08-17"),
    ("eval_w5",   "2026-08-17", "2026-08-20"),
]

PAD_DAYS = 3  # same as OptimizedSatellitePipe


def pad_dates(start: str, end: str) -> tuple[str, str]:
    s = date.fromisoformat(start) - timedelta(days=PAD_DAYS)
    e = date.fromisoformat(end)   + timedelta(days=PAD_DAYS)
    return s.isoformat(), e.isoformat()


def probe_smap_week(buffer: ee.Geometry, start: str, end: str) -> dict:
    """Query SMAP AM and PM for a single date range (with pipeline-style padding)."""
    pad_start, pad_end = pad_dates(start, end)

    smap = (
        ee.ImageCollection(SMAP_005)
        .merge(ee.ImageCollection(SMAP_006))
        .filterBounds(buffer)
        .filterDate(pad_start, pad_end)
    )

    n_images = smap.size().getInfo()

    if n_images == 0:
        return {
            "padded_start": pad_start,
            "padded_end":   pad_end,
            "n_images":     0,
            "SMAP_sm_am":   None,
            "SMAP_sm_pm":   None,
            "qual_am":      None,
            "qual_pm":      None,
        }

    stats = (
        smap.mean()
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=9000,
            bestEffort=True,
        )
        .getInfo()
    ) or {}

    return {
        "padded_start": pad_start,
        "padded_end":   pad_end,
        "n_images":     n_images,
        "SMAP_sm_am":   stats.get("soil_moisture_am"),
        "SMAP_sm_pm":   stats.get("soil_moisture_pm"),
        "qual_am":      stats.get("retrieval_qual_flag_am"),
        "qual_pm":      stats.get("retrieval_qual_flag_pm"),
    }


def main():
    logger = get_logger()
    initialize_ee(logger)

    point  = ee.Geometry.Point([LON, LAT])
    buffer = point.buffer(BUFFER_M)

    results = {}

    hdr = f"{'Week':<12} {'n_img':>6} {'sm_am':>10} {'sm_pm':>10} {'qual_am':>9}  padded_range"
    print(f"\n{hdr}")
    print("-" * 78)

    for label, start, end in PROBE_WEEKS:
        print(f"  Querying {label} ({start} -> {end}) ...", flush=True)
        try:
            r = probe_smap_week(buffer, start, end)
            results[label] = {"start": start, "end": end, **r}

            sm_am_str = f"{r['SMAP_sm_am']:.4f}" if r["SMAP_sm_am"] is not None else "    None"
            sm_pm_str = f"{r['SMAP_sm_pm']:.4f}" if r["SMAP_sm_pm"] is not None else "    None"
            qual_str  = f"{r['qual_am']:.3f}"     if r["qual_am"]    is not None else "    None"
            pad_rng   = f"{r['padded_start']} -> {r['padded_end']}"

            print(f"{label:<12} {r['n_images']:>6} {sm_am_str:>10} {sm_pm_str:>10} {qual_str:>9}  {pad_rng}")

        except Exception as e:
            results[label] = {"start": start, "end": end, "error": str(e)}
            print(f"{label:<12}  ERROR: {e}")

    print()

    # -- Save results -----------------------------------------------------------
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {out_path}")

    # -- Decision summary -------------------------------------------------------
    eval_weeks = [k for k in results if k.startswith("eval_")]
    has_data   = [k for k in eval_weeks if results[k].get("SMAP_sm_am") is not None]
    still_null = [k for k in eval_weeks if results[k].get("SMAP_sm_am") is None and "error" not in results[k]]
    errored    = [k for k in eval_weeks if "error" in results[k]]

    print("\n=== DECISION SUMMARY ===")
    print(f"Evaluation weeks with real SMAP data : {has_data or 'none'}")
    print(f"Evaluation weeks still null           : {still_null or 'none'}")
    print(f"Evaluation weeks errored              : {errored or 'none'}")

    if len(has_data) >= 3:
        print("\n[GO]  SMAP data available for >=3 eval weeks -- PROCEED with full pipeline re-run.")
    elif has_data:
        print(f"\n[PARTIAL]  SMAP data partial ({len(has_data)}/5 eval weeks) -- proceed, note SMAP will be partial.")
    else:
        print("\n[HOLD]  SMAP still all-null -- investigate further before full re-run.")


if __name__ == "__main__":
    main()
