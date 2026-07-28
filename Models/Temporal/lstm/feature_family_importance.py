"""
LSTM v23 - feature FAMILY importance on the current-best checkpoint.

Extends train_v20.permutation_importance (which ablates one feature at a
time) to ablate whole signal families jointly - all columns belonging to a
family are simultaneously replaced by their train-mean, which avoids
within-family correlation masking an individual feature's contribution
(e.g. the 10 G_API lag/rolling/ema variants are highly correlated with each
other, so one-at-a-time ablation understates how much the API signal as a
whole matters).

No retraining: loads the existing best checkpoint
(outputs_v20/window_sweep/top25/seq30/best_model.pt) and reproduces its
exact preprocessing + seq_len=30 windows.

Usage:
    python -m Models.Temporal.lstm.feature_family_importance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.dataset import SoilMoistureDataset, build_datasets
from Models.Temporal.lstm.train_v9 import (
    DROPOUT,
    HIDDEN_SIZE,
    NUM_LAYERS,
    PROJ_SIZE,
    SEED,
    BiLSTMAttn,
    apply_preprocessors,
    compute_metrics,
    fit_preprocessors,
    predict_loader,
    set_seed,
)
from Models.Temporal.lstm.train_v20 import (
    DATA_DIR,
    TRAIN_STRIDE,
    _json_safe,
    check_features,
    load_splits,
)

LSTM_DIR = Path(__file__).parent
OUT_DIR = LSTM_DIR / "outputs_v23_family_importance"
REPORT_FIG_DIR = LSTM_DIR / "report_figs"
OUT_DIR.mkdir(exist_ok=True)
REPORT_FIG_DIR.mkdir(exist_ok=True)

CHECKPOINT = LSTM_DIR / "outputs_v20" / "window_sweep" / "top25" / "seq30" / "best_model.pt"
SEQ_LEN = 30

# Exact feature list of the top25/seq30 checkpoint (order matters - matches
# column order the model was trained on). Hardcoded from
# outputs_v20/window_sweep/top25/seq30/metrics.json config.feature_cols
# to avoid a fragile runtime dependency on the window-sweep script's layout.
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

# Known baseline val RMSE from the checkpoint's own metrics.json - used to
# sanity-check that preprocessing/windowing has been reproduced exactly.
EXPECTED_BASELINE_VAL_RMSE = 0.034279510378837585

# Each of the 25 features assigned to exactly one signal-source family.
FAMILIES = {
    "static_terrain": ["K_slope_sin", "elev", "K_slope_cos"],
    "calendar": ["D_sin_DOY", "cos_year"],
    "interaction": ["SMAP_x_year"],
    "api_antecedent": [
        "V_rollmean_G_API_kobs14",
        "V_ema_G_API_kobs14",
        "V_rollmin_G_API_kobs30",
        "G_DSLR",
        "G_API",
        "V_ema_G_API_kobs7",
        "V_rollmax_G_API_kobs30",
        "C_lag_G_API_kobs1",
        "V_rollmean_G_API_kobs7",
        "V_ema_G_API_kobs30",
    ],
    "precip": ["G_rain_sum_3d", "G_rain_sum_7d"],
    "lst": [
        "V_rollmin_LST_modis_kobs30",
        "C_lag_LST_modis_kobs30",
        "C_lag_LST_modis_kobs6",
        "V_ema_LST_modis_kobs7",
    ],
    "smap": ["V_rollstd_SMAP_sm_interp_kobs30"],
    "optical_s2": ["V_rollmean_s2_b11_kobs7"],
    "sar": ["A_d_E_SAR_diff_kobs14"],
}


def _assert_family_coverage():
    seen = [c for cols in FAMILIES.values() for c in cols]
    assert sorted(seen) == sorted(FEATURE_COLS), (
        "FAMILIES must partition FEATURE_COLS exactly (no gaps, no overlaps)"
    )


@torch.no_grad()
def family_permutation_importance(
    *, model, loader, device, feature_cols, families, train_feature_means, out_json
):
    y_base, pred_base = predict_loader(model, loader, device)
    baseline = compute_metrics(y_base, pred_base)
    X_base = loader.dataset.X
    y = loader.dataset.y
    pin = device.type == "cuda"

    rows = []
    for family, cols in families.items():
        idx = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not idx:
            continue
        X_perm = X_base.copy()
        X_perm[:, :, idx] = train_feature_means[idx]
        perm_ds = SoilMoistureDataset(X_perm, y)
        perm_loader = DataLoader(perm_ds, batch_size=loader.batch_size, shuffle=False, pin_memory=pin)
        y_true, y_pred = predict_loader(model, perm_loader, device)
        m = compute_metrics(y_true, y_pred)
        rows.append(
            {
                "family": family,
                "n_features_ablated": len(idx),
                "features": [feature_cols[i] for i in idx],
                "baseline_val_rmse": baseline["rmse"],
                "ablated_val_rmse": m["rmse"],
                "delta_val_rmse": m["rmse"] - baseline["rmse"],
                "ablated_val_r2": m["r2"],
            }
        )

    rows.sort(key=lambda r: r["delta_val_rmse"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    payload = {
        "method": "jointly replace all columns in a family with train-mean at every timestep",
        "checkpoint": str(CHECKPOINT),
        "baseline_val": baseline,
        "ranking": rows,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    return payload


def save_family_plot(rows, out_path, title):
    ranked = sorted(rows, key=lambda r: r["delta_val_rmse"])
    labels = [f"{r['family']} (n={r['n_features_ablated']})" for r in ranked]
    values = [r["delta_val_rmse"] for r in ranked]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, values, color="tab:blue")
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_xlabel("Validation RMSE increase when family replaced by train mean")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    _assert_family_coverage()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_splits(DATA_DIR)
    check_features(train_df, FEATURE_COLS)

    imputer, scaler = fit_preprocessors(train_df, FEATURE_COLS)
    train_df = apply_preprocessors(train_df, FEATURE_COLS, imputer, scaler)
    val_df = apply_preprocessors(val_df, FEATURE_COLS, imputer, scaler)
    test_df = apply_preprocessors(test_df, FEATURE_COLS, imputer, scaler)

    ds_train, ds_val, ds_test = build_datasets(
        train_df, val_df, test_df, feature_cols=FEATURE_COLS, seq_len=SEQ_LEN, train_stride=TRAIN_STRIDE
    )
    pin = device.type == "cuda"
    val_loader = DataLoader(ds_val, batch_size=256, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(ds_test, batch_size=256, shuffle=False, pin_memory=pin)

    model = BiLSTMAttn(
        n_features=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        proj_size=PROJ_SIZE,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    train_feature_means = ds_train.X.mean(axis=(0, 1))

    val_payload = family_permutation_importance(
        model=model,
        loader=val_loader,
        device=device,
        feature_cols=FEATURE_COLS,
        families=FAMILIES,
        train_feature_means=train_feature_means,
        out_json=OUT_DIR / "family_importance.json",
    )

    baseline_rmse = val_payload["baseline_val"]["rmse"]
    diff = abs(baseline_rmse - EXPECTED_BASELINE_VAL_RMSE)
    print(f"[check] baseline val RMSE={baseline_rmse:.6f}  expected={EXPECTED_BASELINE_VAL_RMSE:.6f}  diff={diff:.2e}")
    if diff > 1e-4:
        print("[WARNING] baseline val RMSE does not match checkpoint's recorded metrics.json - "
              "preprocessing/windowing may not have been reproduced exactly.")

    # secondary diagnostic: same joint-family ablation against test
    test_payload = family_permutation_importance(
        model=model,
        loader=test_loader,
        device=device,
        feature_cols=FEATURE_COLS,
        families=FAMILIES,
        train_feature_means=train_feature_means,
        out_json=OUT_DIR / "family_importance_test.json",
    )

    # merge test ranking into the primary JSON for convenience
    combined = dict(val_payload)
    combined["test_baseline"] = test_payload["baseline_val"]
    combined["test_ranking"] = test_payload["ranking"]
    with open(OUT_DIR / "family_importance.json", "w") as f:
        json.dump(combined, f, indent=2, default=_json_safe)

    save_family_plot(val_payload["ranking"], REPORT_FIG_DIR / "lstm_family_importance.png",
                      "Feature family importance (val, top25/seq30 checkpoint)")

    print("\n[ranking] (val, most important first)")
    for r in val_payload["ranking"]:
        print(f"  {r['rank']:2d}. {r['family']:16s} n={r['n_features_ablated']:2d} "
              f"delta_val_rmse={r['delta_val_rmse']:+.5f} ablated_val_r2={r['ablated_val_r2']:.4f}")

    print(f"\n[saved] {OUT_DIR / 'family_importance.json'}")
    print(f"[saved] {REPORT_FIG_DIR / 'lstm_family_importance.png'}")


if __name__ == "__main__":
    main()
