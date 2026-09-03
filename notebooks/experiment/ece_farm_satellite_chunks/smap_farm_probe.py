#!/usr/bin/env python3
"""SMAP GEE Availability Probe for ECE Enumclaw Research Farm.

Probes NASA/SMAP/SPL3SMP_E/005 and 006 for daily soil moisture availability
at the Enumclaw Farm (Parcel 3420069035) across multiple seasonal windows,
strictly avoiding the May 14 – July 28, 2026 global outage window.
Also queries urban negative controls (Bellevue, Renton) and rural positive
controls (Pullman, Lind) to establish benchmark validity.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import ee
from src.pipeline.utils.gee import initialize_ee
from src.pipeline.utils.logger import get_logger

# Station / Probe Coordinates
PROBE_SITES = {
    "ece_enumclaw_farm": {
        "name": "ECE Enumclaw Farm (Target)",
        "lat": 47.181139,
        "lon": -122.032361,
        "type": "target_rural_farm"
    },
    "ece_bbg_main_st": {
        "name": "ECE BBG Main St (Bellevue)",
        "lat": 47.6098164,
        "lon": -122.1824678,
        "type": "urban_negative_control"
    },
    "ece_renton_home": {
        "name": "ECE Renton Home (Renton)",
        "lat": 47.4890,
        "lon": -122.1950,
        "type": "suburban_negative_control"
    },
    "wa_pullman_rural": {
        "name": "Pullman Agricultural Station",
        "lat": 46.7325,
        "lon": -117.1800,
        "type": "rural_positive_control"
    },
    "wa_lind_rural": {
        "name": "Lind Dryland Research Station",
        "lat": 46.9930,
        "lon": -118.6210,
        "type": "rural_positive_control"
    },
}

SMAP_005 = "NASA/SMAP/SPL3SMP_E/005"
SMAP_006 = "NASA/SMAP/SPL3SMP_E/006"

# Date Windows (strictly avoiding May 14 – July 28, 2026 outage)
TEST_WINDOWS = [
    # 2025 baseline (full seasonal spread)
    ("2025_spring", "2025-04-01", "2025-04-30"),
    ("2025_summer", "2025-07-01", "2025-07-31"),
    ("2025_fall",   "2025-09-01", "2025-09-30"),
    # 2026 pre-outage (before May 14, 2026)
    ("2026_pre_outage_spring", "2026-04-01", "2026-05-05"),
    # 2026 post-outage (after July 28, 2026)
    ("2026_post_outage_aug",   "2026-08-01", "2026-08-25"),
]

BUFFER_M = 1000  # Pipeline standard moving circular buffer


def probe_site_window(lat: float, lon: float, start: str, end: str) -> dict:
    point = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(BUFFER_M)

    # In pipeline, SPL3SMP_E 005 and 006 are merged
    smap_coll = (
        ee.ImageCollection(SMAP_005)
        .merge(ee.ImageCollection(SMAP_006))
        .filterBounds(buffer)
        .filterDate(start, end)
    )

    n_images = smap_coll.size().getInfo()
    if n_images == 0:
        return {
            "n_images": 0,
            "sm_am_mean": None,
            "sm_pm_mean": None,
            "qual_am_mean": None,
            "qual_pm_mean": None,
            "finite_am_count": 0,
            "finite_pm_count": 0,
            "daily_samples": [],
        }

    # Extract mean over the window
    mean_stats = (
        smap_coll.mean()
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=9000,
            bestEffort=True,
        )
        .getInfo()
    ) or {}

    # Query daily individual images to count valid days
    # (limit to first 30 images to keep response fast)
    img_list = smap_coll.toList(35)
    img_count = min(n_images, 35)

    daily_samples = []
    finite_am = 0
    finite_pm = 0

    for i in range(img_count):
        img = ee.Image(img_list.get(i))
        date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        val = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffer,
            scale=9000,
            bestEffort=True,
        ).getInfo() or {}

        sm_am = val.get("soil_moisture_am")
        sm_pm = val.get("soil_moisture_pm")
        q_am = val.get("retrieval_qual_flag_am")
        q_pm = val.get("retrieval_qual_flag_pm")
        s_flag_am = val.get("surface_flag_am")

        if sm_am is not None:
            finite_am += 1
        if sm_pm is not None:
            finite_pm += 1

        daily_samples.append({
            "date": date_str,
            "sm_am": sm_am,
            "sm_pm": sm_pm,
            "qual_am": q_am,
            "qual_pm": q_pm,
            "surface_flag_am": s_flag_am,
        })

    return {
        "n_images": n_images,
        "sm_am_mean": mean_stats.get("soil_moisture_am"),
        "sm_pm_mean": mean_stats.get("soil_moisture_pm"),
        "qual_am_mean": mean_stats.get("retrieval_qual_flag_am"),
        "qual_pm_mean": mean_stats.get("retrieval_qual_flag_pm"),
        "finite_am_count": finite_am,
        "finite_pm_count": finite_pm,
        "total_sampled_days": img_count,
        "daily_samples": daily_samples,
    }


def main():
    logger = get_logger()
    print("=== Initializing Google Earth Engine ===")
    initialize_ee(logger)
    print("Earth Engine successfully initialized.\n")

    results = {}

    print(f"{'Site':<22} {'Window':<24} {'Images':>6} {'AM Mean':>10} {'PM Mean':>10} {'Finite AM':>10} {'Status':<12}")
    print("-" * 100)

    for site_key, site_info in PROBE_SITES.items():
        results[site_key] = {
            "info": site_info,
            "windows": {}
        }
        for win_key, start_date, end_date in TEST_WINDOWS:
            try:
                res = probe_site_window(site_info["lat"], site_info["lon"], start_date, end_date)
                results[site_key]["windows"][win_key] = {
                    "start": start_date,
                    "end": end_date,
                    **res
                }
                am_mean_str = f"{res['sm_am_mean']:.4f}" if res['sm_am_mean'] is not None else "None"
                pm_mean_str = f"{res['sm_pm_mean']:.4f}" if res['sm_pm_mean'] is not None else "None"
                fin_str = f"{res['finite_am_count']}/{res['total_sampled_days']}"
                status = "VALID" if res['sm_am_mean'] is not None else "NULL/MASKED"

                print(f"{site_key:<22} {win_key:<24} {res['n_images']:>6} {am_mean_str:>10} {pm_mean_str:>10} {fin_str:>10} {status:<12}", flush=True)
            except Exception as e:
                print(f"{site_key:<22} {win_key:<24} ERROR: {e}", flush=True)
                results[site_key]["windows"][win_key] = {"error": str(e)}

    # Save output
    out_file = Path(__file__).parent / "smap_probe_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        # Don't bloat JSON with too many daily samples if not needed, but keep for target
        json.dump(results, f, indent=2)
    print(f"\nProbe completed. Full results saved to: {out_file}")


if __name__ == "__main__":
    main()
