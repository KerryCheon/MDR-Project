#!/usr/bin/env python3
"""Per-cluster affine calibration of MLP predictions (offline, no retraining).

1.2 documented a systematic positive test bias for the 2-regime MLPs
(2regime_96 median bias ~ +0.021 vs XGBoost +0.0065; bias^2 ~ 10-17% of MSE).
This script quantifies how much of that gap is recoverable HONESTLY: fit a
per-cluster (and a global) affine map y' = a*y + b on the VAL predictions of
every saved model (best-epoch weights), apply it to the saved test predictions,
and re-metric. Fit is strictly on val (2021-22) — no test leakage.

Reads an experiment dir (default: derived_8.4-eval-mlp-1.2; run again on 1.3
after its sweep with --exp-dir) and writes, to --out:
  calibration_<tag>_summary.csv   per-config raw vs calibrated metrics
  calibrated_preds_<tag>/         per-config per-cluster-calibrated seed-mean
                                  test predictions (.npy), for the leaderboard

Usage:
  python analyze_calibration.py [--exp-dir <1.2|1.3 dir>] [--out <1.3 dir>]
                                [--tag 12] [--max-configs N] [--cpu-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXP_DIR_12 = Path(__file__).resolve().parent / "derived_8.4-eval-mlp-1.2"
EXP_DIR_13 = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR_13.parent / "derived_8.4-eval-1.1"))
sys.path.insert(0, str(EXP_DIR_13))

from eval11.evaluator import compute_metrics  # noqa: E402
from mlp13.model import build_model  # noqa: E402

CLUSTERS = ("0", "1")


def load_tensor(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "n_features": int(data["X_train"].shape[1]),
    }


def load_best_model(cfg: dict, n_features: int, spec_dir: Path, device: str = "cpu"):
    """Reconstruct the model and load best-epoch weights (EMA-aware)."""
    model = build_model(cfg, n_features).to(device)
    best_path = spec_dir / "best_model.pt"
    if not best_path.exists():
        return None
    if cfg.get("ema", False):
        ema = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ema, strict=False)
    else:
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_val(model, X_val, device: str = "cpu") -> np.ndarray:
    with torch.no_grad():
        model.eval()
        out = []
        for i in range(0, X_val.shape[0], 1024):
            Xb = torch.from_numpy(X_val[i : i + 1024]).to(device)
            out.append(model(Xb).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def affine_fit(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares y ~ a*p + b on val rows. Returns (a, b)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.size < 2:
        return 1.0, 0.0
    a, b = np.polyfit(p, y, 1)
    return float(a), float(b)


def analyze_exp(exp_dir: Path, out: Path, tag: str, max_configs: int | None, cpu_only: bool) -> pd.DataFrame:
    device = "cpu" if cpu_only or not torch.cuda.is_available() else "cuda"
    artifacts = exp_dir / "artifacts"
    models_root = exp_dir / "models"

    test_meta = np.load(artifacts / "test_meta.npz", allow_pickle=True)
    y_test = test_meta["y_test"]
    labels_te = np.load(artifacts / "labels_test.npy") if (artifacts / "labels_test.npy").exists() else None

    rows = []
    cal_pred_dir = out / f"calibrated_preds_{tag}"
    cal_pred_dir.mkdir(parents=True, exist_ok=True)

    for family_dir in sorted([p for p in models_root.iterdir() if p.is_dir()]):
        family = family_dir.name
        if not family.startswith("2regime"):
            continue
        for cdir in sorted([p for p in family_dir.iterdir() if p.is_dir()]):
            config_id = cdir.name
            # aggregate meta (n_seeds, per_cluster n_test)
            agg_path = cdir / "meta.json"
            if not agg_path.exists():
                continue
            agg = json.loads(agg_path.read_text(encoding="utf-8"))
            if agg.get("config", {}).get("architecture", "mlp") != "mlp":
                continue
            # 1.3-only exclusions: EMA configs (inherited EMA trainer is broken
            # — documented negative, EMA eval never converges) and target-
            # centered configs (best_model.pt predicts the residual, so a
            # val-fit affine on raw preds would be on the wrong scale).
            if agg.get("config", {}).get("ema", False) or agg.get("config", {}).get("center_target", False):
                continue
            seeds = sorted(
                [int(p.name.split("_")[1]) for p in cdir.iterdir() if p.is_dir() and p.name.startswith("seed_")]
            )
            if not seeds:
                continue
            # per-cluster val preds from each seed's best-epoch weights
            cal_test_seed_preds = []  # per-seed per-cluster-calibrated test preds
            cal_test_global_preds = []
            raw_test_seed_preds = []
            for s in seeds:
                sdir = cdir / f"seed_{s}"
                seed_meta_path = sdir / "meta.json"
                if not seed_meta_path.exists():
                    continue
                seed_meta = json.loads(seed_meta_path.read_text(encoding="utf-8"))
                cfg = seed_meta["config"]
                raw_test = np.load(sdir / "preds.npy")  # full test preds (original units)
                raw_test_seed_preds.append(raw_test)

                val_preds_full = np.zeros_like(raw_test)
                cal_per_cluster = np.zeros_like(raw_test)
                cal_global = np.zeros_like(raw_test)
                a_pool: list[float] = []
                b_pool: list[float] = []
                p_pool: list[np.ndarray] = []
                y_pool: list[np.ndarray] = []
                ok = True
                for cl in CLUSTERS:
                    spec_dir = sdir / f"spec_{cl}"
                    tens = load_tensor(artifacts / f"tensors_{family}_cluster{cl}.npz")
                    model = load_best_model(cfg, tens["n_features"], spec_dir, device)
                    if model is None:
                        ok = False
                        break
                    vp = predict_val(model, tens["X_val"], device)
                    a, b = affine_fit(vp, tens["y_val"])
                    a_pool.append(a)
                    b_pool.append(b)
                    p_pool.append(vp)
                    y_pool.append(tens["y_val"])
                if not ok:
                    continue
                # per-cluster calibration applied to the seed's test preds
                if labels_te is not None:
                    for cl, (a, b) in zip(CLUSTERS, zip(a_pool, b_pool)):
                        mask = labels_te == int(cl)
                        cal_per_cluster[mask] = a * raw_test[mask] + b
                # global affine on pooled val
                vp_all = np.concatenate(p_pool)
                yv_all = np.concatenate(y_pool)
                a_g, b_g = affine_fit(vp_all, yv_all)
                cal_global = a_g * raw_test + b_g
                cal_test_seed_preds.append(cal_per_cluster)
                cal_test_global_preds.append(cal_global)

            if not cal_test_seed_preds:
                continue

            raw_mean = np.mean(raw_test_seed_preds, axis=0)
            cal_pc_mean = np.mean(cal_test_seed_preds, axis=0)
            cal_g_mean = np.mean(cal_test_global_preds, axis=0)

            m_raw = compute_metrics(y_test, raw_mean)
            m_pc = compute_metrics(y_test, cal_pc_mean)
            m_g = compute_metrics(y_test, cal_g_mean)

            rows.append({
                "family": family,
                "config_id": config_id,
                "n_seeds": len(seeds),
                "raw_r2": m_raw["r2"],
                "raw_rmse": m_raw["rmse"],
                "raw_bias": m_raw["bias"],
                "cal_pc_r2": m_pc["r2"],
                "cal_pc_rmse": m_pc["rmse"],
                "cal_pc_bias": m_pc["bias"],
                "cal_g_r2": m_g["r2"],
                "cal_g_rmse": m_g["rmse"],
                "cal_g_bias": m_g["bias"],
            })
            np.save(cal_pred_dir / f"{family}__{config_id}.npy", cal_pc_mean)
            print(f"[cal] {family}/{config_id} raw_r2={m_raw['r2']:.4f} "
                  f"cal_pc_r2={m_pc['r2']:.4f} cal_g_r2={m_g['r2']:.4f} "
                  f"raw_bias={m_raw['bias']:+.4f} cal_pc_bias={m_pc['bias']:+.4f}", flush=True)
            if max_configs and len(rows) >= max_configs:
                break
        if max_configs and len(rows) >= max_configs:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["family", "cal_pc_r2"], ascending=[True, False]).reset_index(drop=True)
        df.to_csv(out / f"calibration_{tag}_summary.csv", index=False)
        print(f"\n[cal] wrote {out / f'calibration_{tag}_summary.csv'} ({len(df)} rows)", flush=True)
        for family in ["2regime_96", "2regime_54"]:
            sub = df[df["family"] == family]
            if sub.empty:
                continue
            print(f"\n--- {family}: raw vs per-cluster-calibrated (top-8 by cal_pc_r2) ---", flush=True)
            cols = ["config_id", "n_seeds", "raw_r2", "cal_pc_r2", "cal_g_r2", "raw_rmse", "cal_pc_rmse", "raw_bias", "cal_pc_bias"]
            print(sub.head(8)[cols].to_string(index=False), flush=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR_12,
                        help="experiment dir whose models/artifacts to analyze (default: 1.2)")
    parser.add_argument("--out", type=Path, default=EXP_DIR_13,
                        help="output dir for CSVs + calibrated preds (default: this 1.3 dir)")
    parser.add_argument("--tag", default="12", help="output tag (12 | 13)")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    analyze_exp(args.exp_dir, args.out, args.tag, args.max_configs, args.cpu_only)


if __name__ == "__main__":
    main()
