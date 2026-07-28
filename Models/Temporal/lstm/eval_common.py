"""
Shared evaluation harness for LSTM Track A experiments.

Adds per-station / per-year / Touchet-excluded breakdowns on top of
train_v9.compute_metrics, and prediction-CSV I/O so ensembling (A4) can
average predictions across seeds without retraining.

Usage (self-check + legacy-checkpoint reproduction):
    python -m Models.Temporal.lstm.eval_common
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.dataset import TARGET, _build_sequences
from Models.Temporal.lstm.train_v9 import compute_metrics

TOUCHET_NAME = "Touchet_WA_824"


def evaluate(y_true, y_pred, station_ids, dates, touchet_name: str = TOUCHET_NAME) -> dict:
    """Overall + per-station + per-year + Touchet-excluded metrics.

    mean_per_station_r2 excludes touchet_name by default: its per-station R2
    is inherently volatile (small target variance, ~seq_len test windows per
    seq_len that cross its ~1437-day val-period gap), so an unweighted mean
    across stations would let it dominate/distort the average. Use
    mean_per_station_r2_all if you want every station included.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    station_ids = np.asarray(station_ids)
    dates = pd.to_datetime(np.asarray(dates))

    overall = compute_metrics(y_true, y_pred)

    touchet_mask = station_ids != touchet_name
    touchet_excluded = compute_metrics(y_true[touchet_mask], y_pred[touchet_mask])

    per_station = {}
    for sid in sorted(set(station_ids)):
        mask = station_ids == sid
        per_station[sid] = compute_metrics(y_true[mask], y_pred[mask])

    non_touchet_r2 = [m["r2"] for sid, m in per_station.items() if sid != touchet_name]
    all_r2 = [m["r2"] for m in per_station.values()]
    mean_per_station_r2 = float(np.mean(non_touchet_r2)) if non_touchet_r2 else float("nan")
    mean_per_station_r2_all = float(np.mean(all_r2)) if all_r2 else float("nan")

    years = np.asarray(dates.year)
    per_year = {}
    for yr in sorted(set(years)):
        mask = years == yr
        per_year[str(yr)] = compute_metrics(y_true[mask], y_pred[mask])

    return {
        "overall": overall,
        "touchet_excluded": touchet_excluded,
        "mean_per_station_r2": mean_per_station_r2,
        "mean_per_station_r2_all": mean_per_station_r2_all,
        "per_station": per_station,
        "per_year": per_year,
    }


def save_predictions_csv(path: Path, station_ids, dates, y_true, y_pred):
    dates = pd.to_datetime(np.asarray(dates))
    df = pd.DataFrame(
        {
            "station_id": np.asarray(station_ids),
            "date": dates.strftime("%Y-%m-%d"),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        }
    )
    df.to_csv(path, index=False)


def reconstruct_legacy_meta(df: pd.DataFrame, seq_len: int, stride: int = 1):
    """station_id/date aligned to the sequences dataset._build_sequences
    would build from `df` alone (the old, per-split-independent windowing).

    Mirrors dataset._build_sequences's loop exactly so it can only be used
    to retroactively add per-station/per-year breakdowns to a checkpoint
    trained/evaluated with the OLD build_datasets (pre-A1). Kept as a real
    function (not a one-off script) since any pre-A1 checkpoint (v9-v22)
    might need this later, not just the one used in the self-check below.
    """
    station_ids, dates = [], []
    for station, grp in df.groupby("station_id", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        for i in range(seq_len, len(grp), stride):
            station_ids.append(station)
            dates.append(grp["date"].iloc[i])
    return np.array(station_ids, dtype=object), np.array(dates)


def _self_check_legacy_meta_matches_build_sequences():
    """Regression guard: reconstruct_legacy_meta's (station_id, date) order
    must line up 1:1 with _build_sequences' y_vals, so a drift in
    _build_sequences (if it's ever edited) is caught here rather than
    silently producing mismatched per-station/year breakdowns."""
    synth = pd.DataFrame(
        {
            "station_id": ["A"] * 8 + ["B"] * 6,
            "date": pd.date_range("2020-01-01", periods=8).tolist()
            + pd.date_range("2020-01-01", periods=6).tolist(),
            "f1": np.arange(14, dtype=float),
            TARGET: np.arange(100, 114, dtype=float),
        }
    )
    seq_len = 3
    X, y = _build_sequences(synth, ["f1"], seq_len, stride=1)
    sid, dates = reconstruct_legacy_meta(synth, seq_len, stride=1)
    assert len(sid) == len(y) == len(dates)
    for i in range(len(y)):
        row = synth[(synth["station_id"] == sid[i]) & (synth["date"] == dates[i])]
        assert len(row) == 1
        assert row[TARGET].iloc[0] == y[i], "reconstruct_legacy_meta desynced from _build_sequences"
    print(f"[self-check] reconstruct_legacy_meta matches _build_sequences ({len(y)} sequences)")


def _reproduce_known_checkpoint():
    """A2 verification: rebuild the top25/seq30 checkpoint's test predictions
    via the OLD build_datasets + reconstruct_legacy_meta, run evaluate(...),
    and confirm it reproduces the checkpoint's recorded test R2 exactly."""
    import torch
    from torch.utils.data import DataLoader

    from Models.Temporal.lstm.dataset import build_datasets
    from Models.Temporal.lstm.train_v9 import (
        DROPOUT,
        HIDDEN_SIZE,
        NUM_LAYERS,
        PROJ_SIZE,
        SEED,
        BiLSTMAttn,
        apply_preprocessors,
        fit_preprocessors,
        predict_loader,
        set_seed,
    )
    from Models.Temporal.lstm.train_v20 import DATA_DIR, TRAIN_STRIDE, load_splits

    LSTM_DIR = Path(__file__).parent
    checkpoint = LSTM_DIR / "outputs_v20" / "window_sweep" / "top25" / "seq30" / "best_model.pt"
    seq_len = 30
    feature_cols = [
        "V_rollmin_LST_modis_kobs30", "C_lag_LST_modis_kobs30", "C_lag_LST_modis_kobs6",
        "V_rollmean_G_API_kobs14", "K_slope_sin", "D_sin_DOY", "V_ema_G_API_kobs14",
        "V_rollmin_G_API_kobs30", "G_rain_sum_3d", "V_ema_LST_modis_kobs7", "G_rain_sum_7d",
        "G_DSLR", "cos_year", "SMAP_x_year", "G_API", "V_ema_G_API_kobs7", "V_rollmax_G_API_kobs30",
        "C_lag_G_API_kobs1", "elev", "V_rollmean_G_API_kobs7", "V_rollstd_SMAP_sm_interp_kobs30",
        "V_ema_G_API_kobs30", "V_rollmean_s2_b11_kobs7", "A_d_E_SAR_diff_kobs14", "K_slope_cos",
    ]
    expected_test_r2 = 0.7900784611701965
    expected_test_n = 3866

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = load_splits(DATA_DIR)
    imputer, scaler = fit_preprocessors(train_df, feature_cols)
    train_p = apply_preprocessors(train_df, feature_cols, imputer, scaler)
    test_p = apply_preprocessors(test_df, feature_cols, imputer, scaler)

    _, _, ds_test = build_datasets(train_p, apply_preprocessors(val_df, feature_cols, imputer, scaler),
                                    test_p, feature_cols=feature_cols, seq_len=seq_len, train_stride=TRAIN_STRIDE)
    test_loader = DataLoader(ds_test, batch_size=256, shuffle=False)

    model = BiLSTMAttn(n_features=len(feature_cols), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                        dropout=DROPOUT, proj_size=PROJ_SIZE).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    y_true, y_pred = predict_loader(model, test_loader, device)
    station_ids, dates = reconstruct_legacy_meta(test_df, seq_len, stride=1)
    assert len(station_ids) == len(y_true), (
        f"legacy meta length {len(station_ids)} != prediction length {len(y_true)}"
    )

    result = evaluate(y_true, y_pred, station_ids, dates)
    r2 = result["overall"]["r2"]
    n = result["overall"]["n"]
    print(f"[reproduce] test R2={r2!r}  n={n}  (expected R2={expected_test_r2!r}  n={expected_test_n})")
    assert n == expected_test_n, f"n mismatch: {n} != {expected_test_n}"
    assert abs(r2 - expected_test_r2) < 1e-6, f"R2 mismatch: {r2} != {expected_test_r2}"
    print("[reproduce] PASSED — eval_common.evaluate exactly reproduces the known checkpoint's test R2")
    return result


if __name__ == "__main__":
    _self_check_legacy_meta_matches_build_sequences()
    _reproduce_known_checkpoint()
