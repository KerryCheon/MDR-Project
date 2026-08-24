# compare_satellite_pipes.py
# Side-by-side benchmarking and regression verification between SatellitePipe and OptimizedSatellitePipe.

import argparse
import sys
import time
import tempfile
import json
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from ..utils.config import load_config
from ..utils.logger import get_logger, setup_logger
from ..pipes.satellite_pipe import SatellitePipe
from ..pipes.optimized_satellite_pipe import OptimizedSatellitePipe


SAT_COLUMNS = [
    "LST_modis", "NDVI_modis", "s1_vv", "s1_vh", "s1_vv_dB", "s1_vh_dB",
    "s2_b2", "s2_b3", "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    "elev", "slope", "aspect",
    "SMAP_sm_am", "SMAP_sm_pm", "SMAP_qual_am", "SMAP_qual_pm"
]


def create_test_dataframe(station_cfg: dict, n_weeks: int = 4) -> pd.DataFrame:
    """Creates a sample input DataFrame for testing satellite pipes."""
    parse_cfg = station_cfg.get("parse", {})
    lat = parse_cfg.get("latitude", 47.51)
    lon = parse_cfg.get("longitude", -123.81)

    # 4-week date range starting 2016-01-01
    dates = pd.date_range(start="2016-01-01", periods=n_weeks * 7, freq="D")
    df = pd.DataFrame({
        "station_id": station_cfg.get("request", {}).get("station", "TEST_STATION"),
        "date": dates,
        "latitude": lat,
        "longitude": lon,
        "air_temp_mean": 5.0,
        "precipitation": 0.0,
        "soil_moisture_5cm": 0.25,
    })
    return df


def compare_satellite_pipes(
    station_name: str = "quinault_4_ne",
    config_path: str = None,
    n_weeks: int = 4,
    use_existing_cache: bool = False,
    mock_offline: bool = False
) -> bool:
    logger = get_logger("compare_satellite")
    config = load_config(config_path) if config_path else load_config()
    setup_logger(config)

    if mock_offline:
        import os
        import ee
        os.environ["GEE_PROJECT_ID"] = os.environ.get("GEE_PROJECT_ID", "mock-project")
        ee.Initialize = lambda project=None: None
        ee.Authenticate = lambda: None

    station_cfg = config.get("stations", {}).get(station_name, {})
    if not station_cfg:
        logger.warning(f"Station {station_name} not found in config, using defaults.")

    test_df = create_test_dataframe(station_cfg, n_weeks=n_weeks)
    logger.info(f"Generated test DataFrame with {len(test_df)} rows ({n_weeks} weeks) for {station_name}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Config for V1
        cfg_v1 = json.loads(json.dumps(config))
        cache_v1 = tmp_path / "v1_cache.json" if not use_existing_cache else Path(f"src/pipeline/data/cache/{station_name}_satellite_cache.json")
        cfg_v1["satellite"]["cache_path"] = str(cache_v1)

        # Config for V2
        cfg_v2 = json.loads(json.dumps(config))
        cache_v2 = tmp_path / "v2_cache.json" if not use_existing_cache else Path(f"src/pipeline/data/cache/{station_name}_satellite_cache.json")
        cfg_v2["satellite"]["cache_path"] = str(cache_v2)

        # -------------------------------------------------------------
        # Benchmark V1 (Baseline SatellitePipe)
        # -------------------------------------------------------------
        logger.info("Running Baseline SatellitePipe (v1)...")
        pipe_v1 = SatellitePipe(config=cfg_v1, station_name=station_name)
        t0 = time.time()
        try:
            res_v1 = pipe_v1.run(test_df.copy())
            time_v1 = time.time() - t0
            logger.info(f"Baseline SatellitePipe finished in {time_v1:.2f}s")
            v1_ok = True
        except Exception as e:
            logger.error(f"Baseline SatellitePipe failed: {e}")
            time_v1 = float("nan")
            res_v1 = None
            v1_ok = False

        # -------------------------------------------------------------
        # Benchmark V2 (Optimized SatellitePipe)
        # -------------------------------------------------------------
        logger.info("Running OptimizedSatellitePipe (v2)...")
        pipe_v2 = OptimizedSatellitePipe(config=cfg_v2, station_name=station_name)
        t0 = time.time()
        try:
            res_v2 = pipe_v2.run(test_df.copy())
            time_v2 = time.time() - t0
            logger.info(f"OptimizedSatellitePipe finished in {time_v2:.2f}s")
            v2_ok = True
        except Exception as e:
            logger.error(f"OptimizedSatellitePipe failed: {e}")
            time_v2 = float("nan")
            res_v2 = None
            v2_ok = False

    if not v1_ok or not v2_ok:
        logger.error("Comparison could not be completed because one or both pipes failed.")
        return False

    # -------------------------------------------------------------
    # Schema & Numeric Parity Checks
    # -------------------------------------------------------------
    speedup = time_v1 / time_v2 if (time_v2 > 0 and not np.isnan(time_v1)) else 1.0

    print("\n" + "=" * 80)
    print(f" SATELLITE PIPELINE OPTIMIZATION BENCHMARK & REGRESSION REPORT")
    print("=" * 80)
    print(f" Station: {station_name}")
    print(f" Rows Processed: {len(test_df)} ({n_weeks} weekly batches)")
    print(f" Baseline (v1) Runtime:   {time_v1:.2f}s")
    print(f" Optimized (v2) Runtime:  {time_v2:.2f}s")
    print(f" Speedup Factor:          {speedup:.2f}x")
    print("-" * 80)

    rows = []
    all_passed = True

    for col in SAT_COLUMNS:
        in_v1 = col in res_v1.columns
        in_v2 = col in res_v2.columns

        if not in_v1 or not in_v2:
            status = "MISSING"
            all_passed = False
            max_diff = "N/A"
            mean_diff = "N/A"
            v1_non_null = res_v1[col].notna().sum() if in_v1 else 0
            v2_non_null = res_v2[col].notna().sum() if in_v2 else 0
        else:
            s1 = res_v1[col].astype(float)
            s2 = res_v2[col].astype(float)
            v1_non_null = int(s1.notna().sum())
            v2_non_null = int(s2.notna().sum())

            # Both null or both matching
            both_na = s1.isna() & s2.isna()
            both_valid = s1.notna() & s2.notna()
            mismatch_na = (s1.notna() & s2.isna()) | (s1.isna() & s2.notna())

            if mismatch_na.any():
                status = "NA_MISMATCH"
                all_passed = False
                max_diff = "N/A"
                mean_diff = "N/A"
            elif both_valid.sum() == 0:
                status = "ALL_NA_MATCH"
                max_diff = "0.0"
                mean_diff = "0.0"
            else:
                diffs = np.abs(s1[both_valid] - s2[both_valid])
                max_d = float(diffs.max())
                mean_d = float(diffs.mean())
                max_diff = f"{max_d:.6e}"
                mean_diff = f"{mean_d:.6e}"

                # Allow tolerance up to 1e-4 for reduction order
                if max_d < 1e-4:
                    status = "PASSED"
                else:
                    status = "DIFF_FAIL"
                    all_passed = False

        rows.append([col, v1_non_null, v2_non_null, max_diff, mean_diff, status])

    headers = ["Feature", "V1 Non-Null", "V2 Non-Null", "Max Abs Diff", "Mean Abs Diff", "Status"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print(f"{headers[0]:<15} | {headers[1]:<11} | {headers[2]:<11} | {headers[3]:<12} | {headers[4]:<13} | {headers[5]}")
        print("-" * 80)
        for r in rows:
            print(f"{r[0]:<15} | {r[1]:<11} | {r[2]:<11} | {r[3]:<12} | {r[4]:<13} | {r[5]}")

    print("=" * 80)
    if all_passed:
        print(" [PASSED] Zero regression detected: OptimizedSatellitePipe matches baseline outputs.")
    else:
        print(" [FAILED] Parity check detected discrepancies between v1 and v2.")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SatellitePipe and OptimizedSatellitePipe.")
    parser.add_argument("--station", "-s", default="quinault_4_ne", help="Station key from config")
    parser.add_argument("--weeks", "-w", type=int, default=3, help="Number of test weeks to process")
    parser.add_argument("--config", "-c", default="src/pipeline/config.yaml", help="Path to config file")
    parser.add_argument("--use-existing-cache", action="store_true", help="Use existing station cache if available")
    parser.add_argument("--mock-offline", action="store_true", help="Run in mock/offline mode without live Earth Engine connection")
    args = parser.parse_args()

    success = compare_satellite_pipes(
        station_name=args.station,
        config_path=args.config,
        n_weeks=args.weeks,
        use_existing_cache=args.use_existing_cache,
        mock_offline=args.mock_offline
    )
    sys.exit(0 if success else 1)
