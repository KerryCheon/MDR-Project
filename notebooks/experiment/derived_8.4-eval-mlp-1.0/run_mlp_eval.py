#!/usr/bin/env python3
"""Aggregate the MLP sweep into the derived_8.4-eval-1.1-style report.

No retraining: the sweep already produced test predictions/models per job.
This script picks the winners per family, builds the leaderboard
(metrics_summary.csv), per-regime breakdown, selected_features.json, all
figures, and the timing summary.

Usage:
    python run_mlp_eval.py [--config config.yaml] [--out .] [--top-n 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
EVAL11_DIR = EXP_DIR.parent / "derived_8.4-eval-1.1"
sys.path.insert(0, str(EVAL11_DIR))
sys.path.insert(0, str(EXP_DIR))

from eval11.data import load_experiment_data  # noqa: E402
from eval11.evaluator import compute_metrics  # noqa: E402
from eval11.routers import get_router  # noqa: E402
from generate_station_year_table import generate_table_from_preds  # noqa: E402
from mlp10.plots import (  # noqa: E402
    plot_diagnostics,
    plot_mlp_loss_curves,
    plot_per_regime_diagnostics,
    plot_sweep_summary,
    plot_yearly_performance_linechart,
)
from run_mlp_sweep import load_cluster_deltas  # noqa: E402

YEARS = [2023, 2024, 2025]


def load_test_meta(artifacts: Path) -> dict:
    data = np.load(artifacts / "test_meta.npz", allow_pickle=True)
    return {
        "y_test": data["y_test"],
        "year": data["year"],
        "station": data["station"],
    }


def yearly_metrics(preds: np.ndarray, test_meta: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for y in YEARS:
        mask = test_meta["year"] == y
        m = compute_metrics(test_meta["y_test"][mask], preds[mask])
        out[f"year_{y}_r2"] = m["r2"]
    return out


def load_xgb_references(config: dict, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """XGBoost reference rows from eval-1.1 (leaderboard + per-regime)."""
    ref_file = PROJECT_ROOT / Path(config["xgboost_reference_file"])
    df = pd.read_csv(ref_file)
    df = df[df["model_name"].isin([
        "Global Single Model (54 Backbone)",
        "Clustering_V0_Full_k2 (Winner c0=0, c1=10)",
    ])].copy()
    df["strategy_name"] = "XGBoost_Reference"
    per_reg = pd.read_csv(EVAL11_DIR / "per_regime_metrics_summary.csv")
    per_reg = per_reg[per_reg["strategy_name"].isin(["Global_Single", "Clustering_V0_Full_k2"])].copy()
    per_reg["strategy_name"] = "XGBoost_Reference"
    return df, per_reg


def mlp_leaderboard_row(
    family: str, config_id: str, meta: dict, test_meta: dict, config: dict, c1_deltas: list[str]
) -> dict:
    backbone = list(config["shared_backbone_54"])
    test = meta["test"]
    row = {
        "model_name": f"MLP {'1-Regime' if family == '1regime' else '2-Regime'} ({config_id})",
        "strategy_name": "MLP_Global" if family == "1regime" else "MLP_Clustering_V0_Full_k2",
        "candidate_id": f"{family}_{config_id}",
        "pooled_r2": test["r2"],
        "pooled_rmse": test["rmse"],
        "pooled_ubrmse": test["ubrmse"],
        "pooled_bias": test["bias"],
        "pooled_mae": test["mae"],
        "pooled_pearson": test["pearson"],
        "global_feature_count": len(backbone),
        "cluster_0_additions": "",
        "cluster_1_additions": ";".join(c1_deltas) if family == "2regime" else "",
        "cluster_0_feature_count": len(backbone),
        "cluster_1_feature_count": len(backbone) + (len(c1_deltas) if family == "2regime" else 0),
        "train_time_s": meta["train_time_s"],
        "epochs": meta["epochs"],
        "best_epoch": meta["best_epoch"],
        "holdout_rmse": meta["holdout_rmse"],
        "config_id": config_id,
        "family": family,
    }
    preds = np.load(EXP_DIR / "models" / family / config_id / "preds.npy")
    row.update(yearly_metrics(preds, test_meta))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--per-regime-n", type=int, default=3)
    args = parser.parse_args()

    t0 = time.perf_counter()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    artifacts = args.out / "artifacts"
    test_meta = load_test_meta(artifacts)
    c1_deltas = load_cluster_deltas(config, EXP_DIR)

    sweep = pd.read_csv(args.out / "sweep_results.csv")
    xgb_df, xgb_per_reg = load_xgb_references(config, args.out)

    # ---- winners per family (ranked by holdout RMSE, already sorted) ----
    rows = []
    winners: dict[str, str] = {}
    for family in ["1regime", "2regime"]:
        sub = sweep[sweep["family"] == family].sort_values("holdout_rmse")
        if sub.empty:
            print(f"[warn] no results for family {family}", flush=True)
            continue
        winners[family] = sub.iloc[0]["config_id"]
        for _, r in sub.head(args.top_n).iterrows():
            meta_path = args.out / "models" / family / r["config_id"] / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append(mlp_leaderboard_row(family, r["config_id"], meta, test_meta, config, c1_deltas))

    # ---- leaderboard: MLP + XGBoost references ----
    df_mlp = pd.DataFrame(rows)
    df_summary = pd.concat([df_mlp, xgb_df], ignore_index=True)
    df_summary = df_summary.sort_values("pooled_r2", ascending=False).reset_index(drop=True)
    df_summary.to_csv(args.out / "metrics_summary.csv", index=False)
    print(f"[eval] wrote metrics_summary.csv ({len(df_summary)} rows)", flush=True)

    # ---- per-regime breakdown ----
    per_reg_rows = []
    for family in ["1regime", "2regime"]:
        for _, r in sweep[sweep["family"] == family].sort_values("holdout_rmse").head(args.per_regime_n).iterrows():
            meta_path = args.out / "models" / family / r["config_id"] / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            mname = f"MLP {'1-Regime' if family == '1regime' else '2-Regime'} ({r['config_id']})"
            if family == "1regime":
                n_train = int(np.load(artifacts / "tensors_global.npz")["train_idx"].shape[0])
                per_reg_rows.append({
                    "strategy_name": "MLP_Global", "model_name": mname, "cluster": 0,
                    "n_train": n_train,
                    "n_test": len(test_meta["y_test"]),
                    **meta["test"],
                })
            else:
                for cl in ["0", "1"]:
                    cm = meta["per_cluster"][cl]
                    per_reg_rows.append({
                        "strategy_name": "MLP_Clustering_V0_Full_k2", "model_name": mname,
                        "cluster": int(cl), "n_train": cm["n_train"], "n_test": cm["n_test"],
                        **cm["test"],
                    })
    df_per_reg = pd.concat([pd.DataFrame(per_reg_rows), xgb_per_reg], ignore_index=True)
    df_per_reg.to_csv(args.out / "per_regime_metrics_summary.csv", index=False)
    print(f"[eval] wrote per_regime_metrics_summary.csv ({len(df_per_reg)} rows)", flush=True)

    # ---- selected_features.json (eval-1.1 schema) ----
    data = load_experiment_data(PROJECT_ROOT, config)
    selected = {
        "shared_backbone_54": list(config["shared_backbone_54"]),
        "baseline_v0_50": list(data.v0_features),
        "cluster_1_delta_features": c1_deltas,
        "mlp_winners": winners,
        "leaderboard": json.loads(df_summary.to_json(orient="records")),
    }
    with open(args.out / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    print(f"[eval] wrote selected_features.json (winners: {winners})", flush=True)

    # ---- figures ----
    y_test = test_meta["y_test"]
    labels_te = np.load(artifacts / "labels_test.npy") if (artifacts / "labels_test.npy").exists() else None

    loss_curves: dict[str, list[float]] = {}
    for family in ["1regime", "2regime"]:
        for _, r in sweep[sweep["family"] == family].sort_values("holdout_rmse").head(3).iterrows():
            cid = r["config_id"]
            meta_path = args.out / "models" / family / cid / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if family == "1regime":
                curves = np.load(args.out / "models" / family / cid / "curves.npy")
                loss_curves[f"{family}/{cid}"] = curves[1].tolist()  # row 1 = test RMSE
            else:
                total = sum(c["n_test"] for c in meta["per_cluster"].values())
                per_cl = []
                for cl in ["0", "1"]:
                    curves = np.load(args.out / "models" / family / cid / f"spec_{cl}" / "curves.npy")
                    per_cl.append((meta["per_cluster"][cl]["n_test"] / total, curves[1]))
                # Align to the longest curve by flat-extending the shorter ones.
                max_len = max(len(c) for _, c in per_cl)
                combined = np.zeros(max_len, dtype=float)
                for w, c in per_cl:
                    c_ext = np.full(max_len, float(c[-1]))
                    c_ext[: len(c)] = c
                    combined += w * c_ext ** 2
                loss_curves[f"{family}/{cid}"] = np.sqrt(combined).tolist()

    plot_mlp_loss_curves(loss_curves, args.out)

    # diagnostics for winners + XGBoost refs
    for family, tag in [("1regime", "1regime"), ("2regime", "2regime")]:
        if family not in winners:
            continue
        cid = winners[family]
        preds = np.load(args.out / "models" / family / cid / "preds.npy")
        mname = f"MLP {'1-Regime' if family == '1regime' else '2-Regime'} ({cid})"
        plot_diagnostics(mname, y_test, preds, args.out)
        if family == "2regime" and labels_te is not None:
            plot_per_regime_diagnostics(mname, y_test, preds, labels_te, args.out)
        generate_table_from_preds(
            data.test, preds, mname,
            f"station_year_metrics_{tag}.png", f" ({mname})",
        )

    # XGBoost reference diagnostics (reuse eval-1.1 saved preds)
    for ref_name, pred_file in [
        ("Global Single Model (54 Backbone)", "Global_Single_backbone_0_0_preds.npy"),
        ("Clustering_V0_Full_k2 (Winner c0=0, c1=10)", "Clustering_V0_Full_k2_winner_preds.npy"),
    ]:
        p = EVAL11_DIR / "models" / pred_file
        if p.exists():
            plot_diagnostics(ref_name, y_test, np.load(p), args.out)

    plot_yearly_performance_linechart(df_summary, args.out)
    plot_sweep_summary(sweep, args.out)

    # ---- timing summary ----
    timing = json.loads((args.out / "timing_log.json").read_text(encoding="utf-8"))
    timing["eval_wall_s"] = time.perf_counter() - t0
    timing["gpu"] = {"device": "NVIDIA H100 PCIe 80GB", "n_parallel": config["sweep"]["n_parallel"]}
    with open(args.out / "timing_log.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    trows = []
    for family in ["1regime", "2regime"]:
        for _, r in sweep[sweep["family"] == family].iterrows():
            meta_path = args.out / "models" / family / r["config_id"] / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            trows.append({
                "family": family, "config_id": r["config_id"],
                "train_time_s": meta["train_time_s"],
                "wall_time_s": r.get("wall_time_s", float("nan")),
                "epochs": meta["epochs"], "best_epoch": meta["best_epoch"],
                "holdout_rmse": meta["holdout_rmse"], "test_r2": meta["test"]["r2"],
            })
    pd.DataFrame(trows).to_csv(args.out / "timing_summary.csv", index=False)
    print(f"[eval] wrote timing_summary.csv + timing_log.json (eval wall {timing['eval_wall_s']:.1f}s)", flush=True)

    # ---- print leaderboard ----
    cols = ["model_name", "strategy_name", "pooled_r2", "pooled_rmse", "pooled_bias", "pooled_mae"]
    print("\n" + "=" * 70, flush=True)
    print("FINAL MODEL LEADERBOARD (derived_8.4-eval-mlp-1.0)", flush=True)
    print("=" * 70, flush=True)
    print(df_summary[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
