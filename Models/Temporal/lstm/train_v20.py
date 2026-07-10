"""
LSTM v20 - feature-aligned BiLSTM+Attention audit on derived_8.0.

This run keeps the v9 architecture fixed but replaces the hand-picked v9
inputs with Jakob's 38-feature MDR-v25 tabular feature set. All features are
fed as per-timestep sequence inputs so the only intended change from v9 is the
feature surface plus the scheduler/convergence instrumentation.

Default usage runs the full audit:
    python -m Models.Temporal.lstm.train_v20

Smoke-test usage:
    python -m Models.Temporal.lstm.train_v20 --max-epochs 2 --patience 2 --no-prune
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.dataset import SoilMoistureDataset, build_datasets
from Models.Temporal.lstm.train_v9 import (
    BATCH_SIZE,
    DROPOUT,
    GRAD_CLIP,
    HIDDEN_SIZE,
    HUBER_DELTA,
    LR,
    NUM_LAYERS,
    PROJ_SIZE,
    SEED,
    TRAIN_STRIDE,
    WEIGHT_DECAY,
    BiLSTMAttn,
    apply_preprocessors,
    compute_metrics,
    fit_preprocessors,
    predict_loader,
    set_seed,
)

DATA_DIR = REPO_ROOT / "data/splits/derived_8.0"
LSTM_DIR = Path(__file__).parent
OUT_DIR = LSTM_DIR / "outputs_v20"
REPORT_FIG_DIR = LSTM_DIR / "report_figs"
OUT_DIR.mkdir(exist_ok=True)
REPORT_FIG_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "SMAP_sm_pm_interp_ema02",
    "V_rollmin_LST_modis_kobs30",
    "D_sin_DOY",
    "G_rain_sum_3d",
    "V_ema_G_API_kobs7",
    "V_rollmin_G_API_kobs30",
    "G_rain_sum_7d",
    "C_lag_LST_modis_kobs30",
    "C_lag_G_API_kobs1",
    "V_ema_G_API_kobs14",
    "V_rollmean_G_API_kobs14",
    "G_API",
    "G_DSLR",
    "SMAP_ampm_diff_interp",
    "V_rollmax_G_API_kobs30",
    "V_ema_G_API_kobs30",
    "V_rollmean_s2_b11_kobs7",
    "V_ema_LST_modis_kobs7",
    "V_rollmean_G_API_kobs7",
    "C_lag_s2_b11_kobs30",
    "A_d_E_SAR_diff_kobs14",
    "C_lag_LST_modis_kobs6",
    "A_d_LST_modis_kobs14",
    "A_d_SMAP_sm_interp_kobs14",
    "V_rollstd_SMAP_sm_interp_kobs30",
    "SMAP_sm_interp_grad7",
    "year_frac",
    "sin_year",
    "cos_year",
    "API_x_year",
    "SMAP_x_year",
    "slope",
    "elev",
    "K_slope_sin",
    "K_slope_cos",
    "K_aspect_cos",
    "J_clay_wfrac_b0",
    "J_sand_wfrac_b0",
]

STATIC_FEATURES = [
    "slope",
    "elev",
    "K_slope_sin",
    "K_slope_cos",
    "K_aspect_cos",
    "J_clay_wfrac_b0",
    "J_sand_wfrac_b0",
]
TIME_FEATURES = [c for c in FEATURE_COLS if c not in STATIC_FEATURES]
ALL_FEATURES = FEATURE_COLS

SEQ_LEN = 10
MAX_EPOCHS = 300
PATIENCE = 60
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 8

ANCHORS = {
    "v9_test_r2": 0.747,
    "jakob_xgboost_test_r2": 0.822,
    "v9_feature_count": 30,
    "jakob_feature_count": 38,
}


def _json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_splits(data_dir: Path = DATA_DIR):
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, val_df, test_df


def check_features(df: pd.DataFrame, feature_cols: list[str]):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def save_training_curve(history: list[dict], out_path: Path, title: str):
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train Huber")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val MSE")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["val_rmse"] for h in history], color="tab:red")
    axes[1].set_ylabel("Val RMSE")

    axes[2].plot(epochs, [h["val_r2"] for h in history], color="tab:green")
    axes[2].set_ylabel("Val R2")
    axes[2].set_xlabel("Epoch")

    best = min(history, key=lambda h: h["val_rmse"])
    for ax in axes:
        ax.axvline(best["epoch"], color="0.5", linestyle="--", linewidth=1)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_importance_plot(rows: list[dict], out_path: Path):
    ranked = sorted(rows, key=lambda r: r["delta_val_rmse"])
    labels = [r["feature"] for r in ranked]
    values = [r["delta_val_rmse"] for r in ranked]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(labels, values, color="tab:blue")
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_xlabel("Validation RMSE increase when replaced by train mean")
    ax.set_title("v20 feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_loaders(train_df, val_df, test_df, feature_cols, seq_len):
    ds_train, ds_val, ds_test = build_datasets(
        train_df,
        val_df,
        test_df,
        feature_cols=feature_cols,
        seq_len=seq_len,
        train_stride=TRAIN_STRIDE,
    )
    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin),
        "train_eval": DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin),
        "val": DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin),
        "test": DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin),
    }
    return ds_train, ds_val, ds_test, loaders


def train_model(
    *,
    feature_cols: list[str],
    seq_len: int,
    variant: str,
    out_dir: Path,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    seed: int = SEED,
    save_report_curve: Path | None = None,
    split_transform=None,
):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_splits()
    transform_metadata = None
    if split_transform is not None:
        train_df, val_df, test_df, transform_metadata = split_transform(
            train_df,
            val_df,
            test_df,
            feature_cols,
        )
    check_features(train_df, feature_cols)
    print(f"\n[{variant}] {len(feature_cols)} features  seq_len={seq_len}  device={device}")
    print(f"[load] train={train_df.shape}  val={val_df.shape}  test={test_df.shape}")

    imputer, scaler = fit_preprocessors(train_df, feature_cols)
    train_df = apply_preprocessors(train_df, feature_cols, imputer, scaler)
    val_df = apply_preprocessors(val_df, feature_cols, imputer, scaler)
    test_df = apply_preprocessors(test_df, feature_cols, imputer, scaler)

    ds_train, _, _, loaders = _make_loaders(train_df, val_df, test_df, feature_cols, seq_len)

    model = BiLSTMAttn(
        n_features=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        proj_size=PROJ_SIZE,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] BiLSTMAttn params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
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
        for X_batch, y_batch in loaders["train"]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running += loss.item() * len(y_batch)

        train_loss = running / len(ds_train)
        y_true_val, y_pred_val = predict_loader(model, loaders["val"], device)
        val_mse = float(np.mean((y_true_val - y_pred_val) ** 2))
        val_rmse = math.sqrt(val_mse)
        val_metrics = compute_metrics(y_true_val, y_pred_val)
        scheduler.step(val_rmse)

        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_mse,
                "val_rmse": val_rmse,
                "val_r2": val_metrics["r2"],
                "lr": lr_now,
            }
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_ctr += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  ep {epoch:3d} train={train_loss:.5f} "
                f"val_rmse={val_rmse:.5f} val_r2={val_metrics['r2']:.4f} "
                f"best={best_val_rmse:.5f} (ep{best_epoch}) lr={lr_now:.2e}"
            )

        if patience_ctr >= patience:
            print(f"[early stop] no improvement for {patience} epochs at epoch {epoch}")
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    results = {}
    for name in ["train_eval", "val", "test"]:
        y_true, y_pred = predict_loader(model, loaders[name], device)
        metric_name = "train" if name == "train_eval" else name
        metrics = compute_metrics(y_true, y_pred)
        results[metric_name] = metrics
        print(
            f"  {metric_name:5s} R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.5f} "
            f"ubRMSE={metrics['ubrmse']:.5f} Bias={metrics['bias']:+.5f} "
            f"MAE={metrics['mae']:.5f} Q90={metrics['q90']:.5f} n={metrics['n']}"
        )

    convergence = {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_by_patience": len(history) < max_epochs,
        "improving_at_old_stop_13": _improving_at_epoch(history, 13),
        "old_short_stop_reference_epoch": 13,
    }
    results["config"] = {
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
        "split_transform": transform_metadata,
        **convergence,
    }
    results["history"] = history

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)

    save_training_curve(history, out_dir / "training_curve.png", f"{variant} training curve")
    if save_report_curve is not None:
        save_training_curve(history, save_report_curve, f"{variant} training curve")

    train_feature_means = ds_train.X.mean(axis=(0, 1))
    return {
        "variant": variant,
        "model": model,
        "device": device,
        "loaders": loaders,
        "feature_cols": feature_cols,
        "train_feature_means": train_feature_means,
        "results": results,
        "out_dir": out_dir,
    }


def _improving_at_epoch(history: list[dict], epoch: int) -> bool | None:
    if len(history) <= epoch:
        return None
    before = [h["val_rmse"] for h in history[:epoch]]
    after = [h["val_rmse"] for h in history[epoch:]]
    return min(after) < min(before)


@torch.no_grad()
def permutation_importance(
    *,
    model,
    loader,
    device,
    feature_cols: list[str],
    train_feature_means: np.ndarray,
    out_json: Path,
    out_png: Path,
):
    y_base, pred_base = predict_loader(model, loader, device)
    baseline = compute_metrics(y_base, pred_base)
    dataset = loader.dataset
    X_base = dataset.X
    y = dataset.y
    pin = device.type == "cuda"

    rows = []
    for j, feature in enumerate(feature_cols):
        X_perm = X_base.copy()
        X_perm[:, :, j] = train_feature_means[j]
        perm_ds = SoilMoistureDataset(X_perm, y)
        perm_loader = DataLoader(perm_ds, batch_size=loader.batch_size, shuffle=False, pin_memory=pin)
        y_true, y_pred = predict_loader(model, perm_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        rows.append(
            {
                "feature": feature,
                "rank": None,
                "baseline_val_rmse": baseline["rmse"],
                "ablated_val_rmse": metrics["rmse"],
                "delta_val_rmse": metrics["rmse"] - baseline["rmse"],
                "ablated_val_r2": metrics["r2"],
            }
        )

    rows = sorted(rows, key=lambda r: r["delta_val_rmse"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    payload = {
        "method": "replace feature with train-sequence mean across every validation timestep",
        "baseline_val": baseline,
        "ranking": rows,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    save_importance_plot(rows, out_png)
    print(f"[importance] saved {out_json}")
    print(f"[importance] saved {out_png}")
    return payload


def top_k_features(importance_payload: dict, k: int):
    ranking = importance_payload["ranking"]
    return [row["feature"] for row in ranking[:k]]


def write_combined_metrics(payload: dict):
    metrics_path = OUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    print(f"[saved] {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run LSTM v20 feature-alignment audit.")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--keep-ks", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--no-prune", action="store_true", help="Run only full-38 + importance.")
    parser.add_argument("--skip-importance", action="store_true", help="Skip permutation importance.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        f"[v20] feature-aligned BiLSTM+Attn on derived_8.0: "
        f"{len(ALL_FEATURES)} features ({len(TIME_FEATURES)} time-varying + "
        f"{len(STATIC_FEATURES)} station-static)"
    )
    print(f"[anchors] v9 test R2={ANCHORS['v9_test_r2']:.3f} | XGBoost test R2={ANCHORS['jakob_xgboost_test_r2']:.3f}")

    full = train_model(
        feature_cols=ALL_FEATURES,
        seq_len=args.seq_len,
        variant="v20_full38",
        out_dir=OUT_DIR / "full38",
        max_epochs=args.max_epochs,
        patience=args.patience,
        save_report_curve=REPORT_FIG_DIR / "training_curve.png",
    )

    combined = {
        "anchors": ANCHORS,
        "full38": _strip_runtime(full),
        "importance": None,
        "pruned_candidates": {},
        "selected_pruned_run": None,
        "selected_pruned_features": None,
        "selected_pruned_feature_count": None,
        "meets_full38_test_r2": None,
    }

    importance_payload = None
    if not args.skip_importance:
        importance_payload = permutation_importance(
            model=full["model"],
            loader=full["loaders"]["val"],
            device=full["device"],
            feature_cols=full["feature_cols"],
            train_feature_means=full["train_feature_means"],
            out_json=OUT_DIR / "feature_importance.json",
            out_png=REPORT_FIG_DIR / "lstm_feature_importance.png",
        )
        combined["importance"] = {
            "json": str(OUT_DIR / "feature_importance.json"),
            "figure": str(REPORT_FIG_DIR / "lstm_feature_importance.png"),
            "ranking": importance_payload["ranking"],
        }

    if importance_payload is not None and not args.no_prune:
        for k in args.keep_ks:
            if k >= len(ALL_FEATURES):
                raise ValueError(f"Pruned keep-K must be < {len(ALL_FEATURES)}; got {k}")
            feature_cols = top_k_features(importance_payload, k)
            run = train_model(
                feature_cols=feature_cols,
                seq_len=args.seq_len,
                variant=f"v20b_top{k}",
                out_dir=OUT_DIR / f"pruned_top{k}",
                max_epochs=args.max_epochs,
                patience=args.patience,
                save_report_curve=REPORT_FIG_DIR / f"training_curve_v20b_top{k}.png",
            )
            combined["pruned_candidates"][f"top{k}"] = _strip_runtime(run)

        selected_name, selected = min(
            combined["pruned_candidates"].items(),
            key=lambda item: item[1]["results"]["val"]["rmse"],
        )
        selected_features = selected["feature_cols"]
        full_test_r2 = combined["full38"]["results"]["test"]["r2"]
        selected_test_r2 = selected["results"]["test"]["r2"]
        combined["selected_pruned_run"] = selected_name
        combined["selected_pruned_features"] = selected_features
        combined["selected_pruned_feature_count"] = len(selected_features)
        combined["meets_full38_test_r2"] = selected_test_r2 >= full_test_r2
        print(
            f"[selected] {selected_name}: {len(selected_features)} features, "
            f"val RMSE={selected['results']['val']['rmse']:.5f}, "
            f"test R2={selected_test_r2:.4f}, full38 test R2={full_test_r2:.4f}"
        )

    write_combined_metrics(combined)


def _strip_runtime(run: dict):
    return {
        "variant": run["variant"],
        "feature_cols": run["feature_cols"],
        "feature_count": len(run["feature_cols"]),
        "results": run["results"],
        "out_dir": str(run["out_dir"]),
    }


if __name__ == "__main__":
    main()
