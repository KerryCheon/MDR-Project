"""
LSTM v23 - engineered vs. raw features.

Jakob's 38-feature MDR-v25 set (train_v20.py) contains 22 hand-engineered
rolling/lag/ema/delta/gradient features derived from a handful of underlying
raw time series. Since the LSTM already ingests a sliding window and can
learn temporal dynamics via its own recurrence, this experiment tests
whether those hand-engineered transforms are redundant: drop all 22
rolling/lag/ema/delta/gradient columns, keep static/calendar/interaction/
base-instantaneous features, and add back the raw underlying series they
were derived from.

Runs at seq_len=10 (apples-to-apples vs v20 full38 seq10, test R2=0.696) and
seq_len=30 (apples-to-apples vs the current-best v20 top25 seq30, R2=0.790).

Usage:
    python -m Models.Temporal.lstm.train_v23_raw_vs_engineered
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.train_v20 import MAX_EPOCHS, PATIENCE, REPORT_FIG_DIR, SEED, _json_safe, _strip_runtime, train_model

LSTM_DIR = Path(__file__).parent
OUT_DIR = LSTM_DIR / "outputs_v23_raw_vs_engineered"
OUT_DIR.mkdir(exist_ok=True)

STATIC_FEATURES = [
    "slope", "elev", "K_slope_sin", "K_slope_cos", "K_aspect_cos",
    "J_clay_wfrac_b0", "J_sand_wfrac_b0",
]  # 7
CALENDAR_FEATURES = ["D_sin_DOY", "year_frac", "sin_year", "cos_year"]  # 4
INTERACTION_FEATURES = ["API_x_year", "SMAP_x_year"]  # 2
BASE_FEATURES = ["G_API", "G_DSLR", "SMAP_ampm_diff_interp"]  # 3
RAW_FEATURES = [
    "LST_modis", "precip_mm", "s2_b11", "E_SAR_diff",
    "SMAP_sm_am_interp", "SMAP_sm_pm_interp", "SMAP_sm_interp",
]  # 7

FEATURE_COLS = STATIC_FEATURES + CALENDAR_FEATURES + INTERACTION_FEATURES + BASE_FEATURES + RAW_FEATURES  # 23

ANCHORS = {
    "v9_test_r2": 0.747,
    "v20_full38_seq10_test_r2": 0.696,
    "v20_top25_seq30_test_r2": 0.790,
    "jakob_xgboost_test_r2": 0.822,
}


def main():
    print(f"[v23_raw_vs_engineered] {len(FEATURE_COLS)} features (dropped 22 rolling/lag/ema/delta/gradient cols)")

    run_seq10 = train_model(
        feature_cols=FEATURE_COLS,
        seq_len=10,
        variant="v23_raw_seq10",
        out_dir=OUT_DIR / "seq10",
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        seed=SEED,
        save_report_curve=REPORT_FIG_DIR / "training_curve_v23_raw_seq10.png",
    )

    run_seq30 = train_model(
        feature_cols=FEATURE_COLS,
        seq_len=30,
        variant="v23_raw_seq30",
        out_dir=OUT_DIR / "seq30",
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        seed=SEED,
        save_report_curve=REPORT_FIG_DIR / "training_curve_v23_raw_seq30.png",
    )

    seq10_stripped = _strip_runtime(run_seq10)
    seq30_stripped = _strip_runtime(run_seq30)
    best_variant = "seq10" if seq10_stripped["results"]["test"]["r2"] > seq30_stripped["results"]["test"]["r2"] else "seq30"

    combined = {
        "anchors": ANCHORS,
        "seq10": seq10_stripped,
        "seq30": seq30_stripped,
        "best_variant": best_variant,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(combined, f, indent=2, default=_json_safe)

    print(f"\n[summary] seq10 test R2={seq10_stripped['results']['test']['r2']:.4f} "
          f"RMSE={seq10_stripped['results']['test']['rmse']:.5f}")
    print(f"[summary] seq30 test R2={seq30_stripped['results']['test']['r2']:.4f} "
          f"RMSE={seq30_stripped['results']['test']['rmse']:.5f}")
    print(f"[summary] best_variant={best_variant}")
    print(f"[saved] {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
