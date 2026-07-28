"""
LSTM v23 - is BiLSTMAttn over-parameterized? Small RNN/GRU baseline.

BiLSTMAttn (2-layer, bidirectional, hidden=80, + attention pooling) has
283,538 parameters trained on ~6.9k sequences over 5 stations. This
experiment trains a deliberately small, single-layer, unidirectional
GRU/RNN (SimpleRNNRegressor in model.py, no attention) on the exact same
feature set/seq_len as the current-best checkpoint (top25/seq30) to see
whether the extra capacity actually buys anything.

Usage:
    python -m Models.Temporal.lstm.train_v23_small_rnn
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.model import SimpleRNNRegressor
from Models.Temporal.lstm.train_v20 import MAX_EPOCHS, PATIENCE, REPORT_FIG_DIR, SEED, _json_safe, train_model

LSTM_DIR = Path(__file__).parent
OUT_DIR = LSTM_DIR / "outputs_v23_small_rnn"
OUT_DIR.mkdir(exist_ok=True)

# Same feature set + seq_len as the v20 top25/seq30 checkpoint (test R2=0.790,
# 283,538 params), for a direct, fair comparison against BiLSTMAttn.
FEATURE_COLS = [
    "V_rollmin_LST_modis_kobs30",
    "C_lag_LST_modis_kobs30",
    "C_lag_LST_modis_kobs6",
    "V_rollmean_G_API_kobs14",
    "K_slope_sin",
    "D_sin_DOY",
    "V_ema_G_API_kobs14",
    "V_rollmin_G_API_kobs30",
    "G_rain_sum_3d",
    "V_ema_LST_modis_kobs7",
    "G_rain_sum_7d",
    "G_DSLR",
    "cos_year",
    "SMAP_x_year",
    "G_API",
    "V_ema_G_API_kobs7",
    "V_rollmax_G_API_kobs30",
    "C_lag_G_API_kobs1",
    "elev",
    "V_rollmean_G_API_kobs7",
    "V_rollstd_SMAP_sm_interp_kobs30",
    "V_ema_G_API_kobs30",
    "V_rollmean_s2_b11_kobs7",
    "A_d_E_SAR_diff_kobs14",
    "K_slope_cos",
]
SEQ_LEN = 30

HIDDEN_SWEEP = [16, 32, 48]

BILSTM_ATTN_REFERENCE = {
    "architecture": "BiLSTMAttn",
    "n_params": 283538,
    "test_r2": 0.7900784611701965,
    "test_rmse": 0.04334788769483566,
    "val_rmse": 0.034279510378837585,
    "source": "outputs_v20/window_sweep/top25/seq30/metrics.json",
}


def make_factory(rnn_type: str, hidden_size: int):
    def factory(n_features: int):
        return SimpleRNNRegressor(n_features=n_features, hidden_size=hidden_size, rnn_type=rnn_type)
    return factory


def model_config(rnn_type: str, hidden_size: int):
    return {
        "architecture": "SimpleRNNRegressor",
        "rnn_type": rnn_type,
        "hidden_size": hidden_size,
        "num_layers": 1,
        "bidirectional": False,
        "attention": False,
    }


def main():
    runs = {}

    for h in HIDDEN_SWEEP:
        variant = f"v23_smallrnn_gru_h{h}"
        run = train_model(
            feature_cols=FEATURE_COLS,
            seq_len=SEQ_LEN,
            variant=variant,
            out_dir=OUT_DIR / f"gru_h{h}",
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            seed=SEED,
            save_report_curve=REPORT_FIG_DIR / f"training_curve_v23_smallrnn_gru_h{h}.png",
            model_factory=make_factory("gru", h),
            model_config=model_config("gru", h),
        )
        runs[f"gru_h{h}"] = run

    best_h = min(HIDDEN_SWEEP, key=lambda h: runs[f"gru_h{h}"]["results"]["val"]["rmse"])
    print(f"\n[gru sweep] best hidden_size by val RMSE = {best_h}")

    variant = f"v23_smallrnn_rnn_h{best_h}"
    rnn_run = train_model(
        feature_cols=FEATURE_COLS,
        seq_len=SEQ_LEN,
        variant=variant,
        out_dir=OUT_DIR / f"rnn_h{best_h}",
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        seed=SEED,
        save_report_curve=REPORT_FIG_DIR / f"training_curve_v23_smallrnn_rnn_h{best_h}.png",
        model_factory=make_factory("rnn", best_h),
        model_config=model_config("rnn", best_h),
    )
    runs[f"rnn_h{best_h}"] = rnn_run

    combined = {"bilstm_attn_reference": BILSTM_ATTN_REFERENCE, "runs": {}}
    for name, run in runs.items():
        combined["runs"][name] = {
            "variant": run["variant"],
            "n_params": run["results"]["config"]["n_params"],
            "results": run["results"],
        }
    combined["best_gru_hidden_size"] = best_h

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(combined, f, indent=2, default=_json_safe)

    print("\n[summary]")
    print(f"  {'variant':22s} {'n_params':>10s} {'val_r2':>8s} {'test_r2':>8s} {'test_rmse':>10s}")
    print(f"  {'BiLSTMAttn (ref)':22s} {BILSTM_ATTN_REFERENCE['n_params']:>10,d} "
          f"{'-':>8s} {BILSTM_ATTN_REFERENCE['test_r2']:>8.4f} {BILSTM_ATTN_REFERENCE['test_rmse']:>10.5f}")
    for name, run in runs.items():
        res = run["results"]
        print(f"  {name:22s} {res['config']['n_params']:>10,d} "
              f"{res['val']['r2']:>8.4f} {res['test']['r2']:>8.4f} {res['test']['rmse']:>10.5f}")

    print(f"\n[saved] {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
