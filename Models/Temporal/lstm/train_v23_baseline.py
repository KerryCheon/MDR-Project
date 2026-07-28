"""
LSTM Track A - 5-seed re-baseline on the coverage-fixed dataset builder.

Re-establishes a trustworthy baseline for (top25, seq_len=30) and
(full58, seq_len=10) using build_datasets_v2 (dataset.py) instead of the
old build_datasets, which silently dropped each station's first seq_len
val/test rows. Also reports per-seed variance, which determines whether
prior sessions' erratic window-sweep results were real signal or seed
noise.

train_one_seed() is intentionally NOT threaded through train_v20.train_model
(that function is already shared by 5+ scripts and hardcoded to the old
builder) - it duplicates train_model's training-loop structure exactly, but
swaps in build_datasets_v2 + eval_common. train_v23_window_sweep.py imports
this same function.

Usage:
    python -m Models.Temporal.lstm.train_v23_baseline
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.dataset import build_datasets_v2
from Models.Temporal.lstm.eval_common import evaluate, save_predictions_csv
from Models.Temporal.lstm.train_v9 import (
    BATCH_SIZE,
    DROPOUT,
    GRAD_CLIP,
    HIDDEN_SIZE,
    HUBER_DELTA,
    LR,
    NUM_LAYERS,
    PROJ_SIZE,
    WEIGHT_DECAY,
    BiLSTMAttn,
    apply_preprocessors,
    compute_metrics,
    fit_preprocessors,
    predict_loader,
    set_seed,
)
from Models.Temporal.lstm.train_v20 import (
    DATA_DIR,
    MAX_EPOCHS,
    PATIENCE,
    PLATEAU_FACTOR,
    PLATEAU_PATIENCE,
    REPORT_FIG_DIR,
    TRAIN_STRIDE,
    _json_safe,
    check_features,
    load_splits,
)
from Models.Temporal.lstm.train_v21 import ALL_FEATURES as FULL58_FEATURES

LSTM_DIR = Path(__file__).parent
OUT_DIR = LSTM_DIR / "outputs_v23_baseline"
OUT_DIR.mkdir(exist_ok=True)

TOP25_FEATURES = [
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

SEEDS = [42, 7, 123, 2024, 31337]

CONFIGS = {
    "top25_seq30": {"feature_cols": TOP25_FEATURES, "seq_len": 30},
    "full58_seq10": {"feature_cols": FULL58_FEATURES, "seq_len": 10},
}

ANCHORS = {
    "v20_top25_seq30_test_r2_old_builder": 0.7900784611701965,
    "v20_top25_seq30_test_n_old_builder": 3866,
    "v20_full38_seq10_test_r2_old_builder": 0.6963052749633789,
}


def train_one_seed(
    *,
    feature_cols: list[str],
    seq_len: int,
    variant: str,
    out_dir: Path,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> dict:
    """Mirrors train_v20.train_model's training loop (same BiLSTMAttn
    hyperparams / Huber / AdamW / ReduceLROnPlateau / early-stop / best-
    checkpoint-reload), but uses build_datasets_v2 (coverage-fixed) and
    eval_common (per-station/year breakdowns + prediction CSVs) instead of
    build_datasets. train_df/val_df/test_df may be passed pre-loaded to
    skip repeated CSV reads across a sweep; preprocessing still happens
    fresh inside this call.
    """
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_df is None or val_df is None or test_df is None:
        train_df, val_df, test_df = load_splits(DATA_DIR)
    check_features(train_df, feature_cols)
    print(f"\n[{variant}] seed={seed} {len(feature_cols)} features seq_len={seq_len} device={device}")

    imputer, scaler = fit_preprocessors(train_df, feature_cols)
    train_p = apply_preprocessors(train_df, feature_cols, imputer, scaler)
    val_p = apply_preprocessors(val_df, feature_cols, imputer, scaler)
    test_p = apply_preprocessors(test_df, feature_cols, imputer, scaler)

    ds_train, ds_val, ds_test, diagnostics = build_datasets_v2(
        train_p, val_p, test_p, feature_cols=feature_cols, seq_len=seq_len, train_stride=TRAIN_STRIDE
    )

    pin = device.type == "cuda"
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin, generator=gen)
    train_eval_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    model = BiLSTMAttn(
        n_features=len(feature_cols), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        dropout=DROPOUT, proj_size=PROJ_SIZE,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val_rmse = math.inf
    best_epoch = -1
    patience_ctr = 0
    history = []
    best_model_path = out_dir / "best_model.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running += loss.item() * len(y_batch)

        train_loss = running / len(ds_train)
        y_true_val, y_pred_val = predict_loader(model, val_loader, device)
        val_mse = float(np.mean((y_true_val - y_pred_val) ** 2))
        val_rmse = math.sqrt(val_mse)
        val_metrics = compute_metrics(y_true_val, y_pred_val)
        scheduler.step(val_rmse)

        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_mse,
             "val_rmse": val_rmse, "val_r2": val_metrics["r2"], "lr": lr_now}
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_ctr += 1

        if epoch == 1 or epoch % 20 == 0:
            print(f"  ep {epoch:3d} train={train_loss:.5f} val_rmse={val_rmse:.5f} "
                  f"val_r2={val_metrics['r2']:.4f} best={best_val_rmse:.5f} (ep{best_epoch})")

        if patience_ctr >= patience:
            print(f"[early stop] no improvement for {patience} epochs at epoch {epoch}")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    y_true_train, y_pred_train = predict_loader(model, train_eval_loader, device)
    y_true_val, y_pred_val = predict_loader(model, val_loader, device)
    y_true_test, y_pred_test = predict_loader(model, test_loader, device)

    train_metrics = compute_metrics(y_true_train, y_pred_train)
    val_eval = evaluate(y_true_val, y_pred_val, ds_val.station_id, ds_val.date)
    test_eval = evaluate(y_true_test, y_pred_test, ds_test.station_id, ds_test.date)

    val_csv = out_dir / "val_preds.csv"
    test_csv = out_dir / "test_preds.csv"
    save_predictions_csv(val_csv, ds_val.station_id, ds_val.date, y_true_val, y_pred_val)
    save_predictions_csv(test_csv, ds_test.station_id, ds_test.date, y_true_test, y_pred_test)

    print(f"  train R2={train_metrics['r2']:.4f} RMSE={train_metrics['rmse']:.5f}")
    print(f"  val   R2={val_eval['overall']['r2']:.4f} RMSE={val_eval['overall']['rmse']:.5f} n={val_eval['overall']['n']}")
    print(f"  test  R2={test_eval['overall']['r2']:.4f} RMSE={test_eval['overall']['rmse']:.5f} n={test_eval['overall']['n']}")

    results = {
        "train": train_metrics,
        "val": val_eval["overall"],
        "test": test_eval["overall"],
        "val_eval": val_eval,
        "test_eval": test_eval,
        "diagnostics": diagnostics,
        "config": {
            "variant": variant,
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "seq_len": seq_len,
            "train_stride": TRAIN_STRIDE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "proj_size": PROJ_SIZE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "huber_delta": HUBER_DELTA,
            "scheduler": f"ReduceLROnPlateau(factor={PLATEAU_FACTOR},patience={PLATEAU_PATIENCE})",
            "max_epochs": max_epochs,
            "patience": patience,
            "seed": seed,
            "n_params": n_params,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "stopped_by_patience": len(history) < max_epochs,
        },
        "history": history,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)

    return {
        "variant": variant,
        "seed": seed,
        "results": results,
        "val_pred_csv": val_csv,
        "test_pred_csv": test_csv,
        "out_dir": out_dir,
    }


def main():
    train_df, val_df, test_df = load_splits(DATA_DIR)

    combined = {"anchors": ANCHORS, "configs": {}}
    for config_name, cfg in CONFIGS.items():
        config_dir = OUT_DIR / config_name
        seed_results = []
        for seed in SEEDS:
            run = train_one_seed(
                feature_cols=cfg["feature_cols"],
                seq_len=cfg["seq_len"],
                variant=f"v23_baseline_{config_name}_seed{seed}",
                out_dir=config_dir / f"seed{seed}",
                seed=seed,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
            )
            seed_results.append(run)

        test_r2s = [r["results"]["test"]["r2"] for r in seed_results]
        test_rmses = [r["results"]["test"]["rmse"] for r in seed_results]
        test_maes = [r["results"]["test"]["mae"] for r in seed_results]

        combined["configs"][config_name] = {
            "feature_count": len(cfg["feature_cols"]),
            "seq_len": cfg["seq_len"],
            "seeds": SEEDS,
            "per_seed_test_r2": test_r2s,
            "test_r2_mean": float(np.mean(test_r2s)),
            "test_r2_std": float(np.std(test_r2s)),
            "test_rmse_mean": float(np.mean(test_rmses)),
            "test_rmse_std": float(np.std(test_rmses)),
            "test_mae_mean": float(np.mean(test_maes)),
            "test_mae_std": float(np.std(test_maes)),
            "seed_out_dirs": [str(r["out_dir"]) for r in seed_results],
        }

        print(f"\n[{config_name}] test R2 = {np.mean(test_r2s):.4f} +/- {np.std(test_r2s):.4f} "
              f"(seeds: {[round(x, 4) for x in test_r2s]})")

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(combined, f, indent=2, default=_json_safe)
    print(f"\n[saved] {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
