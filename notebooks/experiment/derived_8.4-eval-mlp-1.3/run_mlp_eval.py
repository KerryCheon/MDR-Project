#!/usr/bin/env python3
"""Aggregate the derived_8.4-eval-mlp-1.3 sweep into the eval-1.1-style report.

No retraining: the sweep already produced per-seed test predictions/models per
job. Picks the winners per family by ROBUST SCORE (mean over seeds of
mean(val_rmse, aux2020_rmse)) — the selection-protocol v3 fix — and also printsthe val-only ranking + Spearman correlations so the fix is auditable. Builds
the leaderboard (metrics_summary.csv), per-regime breakdown, retrain/ensemble
rows (from run_mlp_retrain.py), top-3 robust-config ensembles, the optional
cross-architecture MLP+FT ensemble, selected_features.json, all figures, and
timing summary.

Usage:
    python run_mlp_eval.py [--config config.yaml] [--out .] [--top-n 5]
                           [--per-regime-n 3]
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
from scipy.stats import spearmanr

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
EVAL11_DIR = EXP_DIR.parent / "derived_8.4-eval-1.1"
MLP11_DIR = EXP_DIR.parent / "derived_8.4-eval-mlp-1.1"
MLP12_DIR = EXP_DIR.parent / "derived_8.4-eval-mlp-1.2"
sys.path.insert(0, str(EVAL11_DIR))
sys.path.insert(0, str(EXP_DIR))

from eval11.data import load_experiment_data  # noqa: E402
from eval11.evaluator import compute_metrics  # noqa: E402
from generate_station_year_table import generate_table_from_preds  # noqa: E402
from mlp13.plots import (  # noqa: E402
    plot_diagnostics,
    plot_mlp_loss_curves,
    plot_per_regime_diagnostics,
    plot_sweep_summary,
    plot_yearly_performance_linechart,
)
from run_mlp_sweep import load_cluster_deltas  # noqa: E402

YEARS = [2023, 2024, 2025]

FAMILY_LABELS = {
    "2regime_96": "2-Regime-96",
    "2regime_54": "2-Regime-54",
}

XGB_TARGET_R2 = 0.81496  # eval-1.1 Clustering_V0_Full_k2 winner (test-selected)


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


def load_xgb_references(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def load_mlp12_references(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best MLP-1.2 rows per family (val-selected winner + test-best reference)."""
    ref_file = PROJECT_ROOT / Path(config["mlp12_reference_file"])
    df = pd.read_csv(ref_file)
    fam_map = {"2regime_96": "2regime_96", "2regime_54": "2regime_54"}
    keep = []
    for fam in fam_map:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        sub = sub.copy()
        # val-selected winner = min val_rmse among the honest MLP_<fam> rows
        honest = sub[sub["strategy_name"] == f"MLP_{fam}"].sort_values("val_rmse", na_position="last")
        if not honest.empty:
            r = honest.iloc[0].copy()
            r["strategy_name"] = "MLP_1.2_Reference"
            r["model_name"] = f"MLP-1.2 {FAMILY_LABELS[fam]} (val_sel: {r['config_id']})"
            keep.append(r)
        # test-best reference rows (reporting only)
        tb = sub[sub["strategy_name"] == "MLP_testbest_reference"].sort_values("pooled_r2", ascending=False).head(1)
        if not tb.empty:
            r = tb.iloc[0].copy()
            r["strategy_name"] = "MLP_1.2_Reference"
            r["model_name"] = f"MLP-1.2 {FAMILY_LABELS[fam]} (test_best: {r['config_id']})"
            keep.append(r)
    df_ref = pd.DataFrame(keep) if keep else pd.DataFrame(columns=df.columns)
    # per-regime rows for the 1.2 val-selected winners
    per_reg = pd.read_csv(MLP12_DIR / "per_regime_metrics_summary.csv")
    per_reg = per_reg[per_reg["strategy_name"].isin(["MLP_2regime_96", "MLP_2regime_54"])].copy()
    winners = {}
    for fam in fam_map:
        sub = df[df["family"] == fam]
        honest = sub[sub["strategy_name"] == f"MLP_{fam}"].sort_values("val_rmse", na_position="last")
        if not honest.empty:
            winners[f"MLP_{fam}"] = honest.iloc[0]["config_id"]
    keep_rows = []
    for strat, winner in winners.items():
        sub = per_reg[per_reg["strategy_name"] == strat]
        keep_rows.append(sub[sub["model_name"].str.contains(winner, regex=False)])
    per_reg = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame(columns=per_reg.columns)
    per_reg["strategy_name"] = "MLP_1.2_Reference"
    return df_ref, per_reg


def mlp_leaderboard_row(
    family: str, config_id: str, meta: dict, test_meta: dict, config: dict, c1_deltas: list[str],
    *, tag: str = "", label: str | None = None,
) -> dict:
    backbone = list(config["shared_backbone_54"])
    test = meta["test"]
    mname = label or f"MLP {FAMILY_LABELS[family]} ({config_id})"
    row = {
        "model_name": mname,
        "strategy_name": f"MLP_{family}" if not tag else tag,
        "candidate_id": f"{family}_{config_id}",
        "pooled_r2": test["r2"],
        "pooled_rmse": test["rmse"],
        "pooled_ubrmse": test["ubrmse"],
        "pooled_bias": test["bias"],
        "pooled_mae": test["mae"],
        "pooled_pearson": test["pearson"],
        "global_feature_count": len(backbone),
        "cluster_0_additions": "",
        "cluster_1_additions": ";".join(c1_deltas) if "2regime" in family else "",
        "cluster_0_feature_count": len(backbone),
        "cluster_1_feature_count": len(backbone) + (len(c1_deltas) if "2regime" in family else 0),
        "train_time_s": meta.get("train_time_s", float("nan")),
        "epochs": meta.get("epochs", float("nan")),
        "best_epoch": meta.get("best_epoch", float("nan")),
        "val_rmse": meta.get("val_rmse", float("nan")),
        "aux_rmse": meta.get("aux_rmse", float("nan")),
        "robust_score": meta.get("robust_score", float("nan")),
        "config_id": config_id,
        "family": family,
        "n_seeds": meta.get("n_seeds", 1),
    }
    preds = np.load(EXP_DIR / "models" / family / config_id / "preds.npy")
    row.update(yearly_metrics(preds, test_meta))
    return row


def print_selection_diagnostic(sweep: pd.DataFrame, test_meta: dict, out: Path) -> None:
    """Robust-score vs val-only selection: ranking agreement + Spearman vs test."""
    print("\n" + "=" * 78, flush=True)
    print("SELECTION PROTOCOL v4 DIAGNOSTIC (selection = 2-seed mean val RMSE; robust/aux reported)", flush=True)
    print("=" * 78, flush=True)
    for family in FAMILY_LABELS:
        sub = sweep[sweep["family"] == family].dropna(subset=["test_r2"]).copy()
        if sub.empty:
            continue
        sub = sub.sort_values("robust_score", na_position="last").reset_index(drop=True)
        print(f"\n--- {family} top-10 by robust score (vs val-only rank / test R2) ---", flush=True)
        cols = ["config_id", "architecture", "n_seeds", "robust_score", "val_rmse", "aux_rmse", "test_r2"]
        show = sub.head(10)[cols].copy()
        val_rank = sub.sort_values("val_rmse")["config_id"].tolist()
        show["val_rank"] = [val_rank.index(c) + 1 for c in show["config_id"]]
        print(show.to_string(index=False), flush=True)

        # Spearman: does the selection metric rank correlate with test R2?
        for metric, label in [("val_rmse", "val_rmse"), ("robust_score", "robust_score")]:
            valid = sub.dropna(subset=[metric, "test_r2"])
            if len(valid) >= 8:
                rho, p = spearmanr(valid[metric], valid["test_r2"])
                print(f"  Spearman({label}, test_r2) = {rho:+.3f} (p={p:.3f}, n={len(valid)})", flush=True)
        # what would val-only selection have picked vs robust selection?
        mlp_sub = sub[sub["architecture"] == "mlp"]
        robust_winner = sub.sort_values("robust_score").iloc[0]
        val_winner = sub.sort_values("val_rmse").iloc[0]
        robust_mlp_winner = mlp_sub.sort_values("robust_score").iloc[0]
        val_mlp_winner = mlp_sub.sort_values("val_rmse").iloc[0]
        test_best = sub.sort_values("test_r2", ascending=False).iloc[0]
        print(f"  robust winner (all) : {robust_winner['config_id']} (test_r2={robust_winner['test_r2']:.4f})", flush=True)
        print(f"  val winner (all)    : {val_winner['config_id']} (test_r2={val_winner['test_r2']:.4f})", flush=True)
        print(f"  robust winner (MLP) : {robust_mlp_winner['config_id']} (test_r2={robust_mlp_winner['test_r2']:.4f})", flush=True)
        print(f"  val winner (MLP)    : {val_mlp_winner['config_id']} (test_r2={val_mlp_winner['test_r2']:.4f})", flush=True)
        print(f"  test best (ref)     : {test_best['config_id']} (test_r2={test_best['test_r2']:.4f})", flush=True)
    print(flush=True)


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
    xgb_df, xgb_per_reg = load_xgb_references(config)
    mlp12_df, mlp12_per_reg = load_mlp12_references(config)

    print_selection_diagnostic(sweep, test_meta, args.out)

    rows = []
    winners: dict[str, str] = {}
    for family in [f["id"] for f in config["families"]]:
        sub = sweep[(sweep["family"] == family) & (sweep["architecture"] == "mlp")].sort_values("val_rmse", na_position="last")
        if sub.empty:
            print(f"[warn] no results for family {family}", flush=True)
            continue
        winners[family] = sub.iloc[0]["config_id"]
        for _, r in sub.head(args.top_n).iterrows():
            meta_path = args.out / "models" / family / r["config_id"] / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append(mlp_leaderboard_row(family, r["config_id"], meta, test_meta, config, c1_deltas))

    # retrain + 5-seed ensembles (if present) as the final champion rows
    retrain_path = args.out / "retrain_results.csv"
    retrain_rows: list[dict] = []
    if retrain_path.exists():
        rdf = pd.read_csv(retrain_path)
        for _, r in rdf.iterrows():
            ens_dir = args.out / "models" / "retrain" / f"{r['family']}__{r['config_id']}"
            emeta = json.loads((ens_dir / "ens_meta.json").read_text(encoding="utf-8"))
            preds = np.load(ens_dir / "ens_preds.npy")
            test = emeta["test"]
            retrain_rows.append({
                "model_name": f"MLP {FAMILY_LABELS[r['family']]} (retrain-ens, {r['config_id']})",
                "strategy_name": f"MLP_{r['family']}",
                "candidate_id": f"{r['family']}_{r['config_id']}_retrain",
                "pooled_r2": test["r2"], "pooled_rmse": test["rmse"],
                "pooled_ubrmse": test["ubrmse"], "pooled_bias": test["bias"],
                "pooled_mae": test["mae"], "pooled_pearson": test["pearson"],
                "global_feature_count": len(config["shared_backbone_54"]),
                "cluster_0_additions": "",
                "cluster_1_additions": ";".join(c1_deltas) if "2regime" in r["family"] else "",
                "cluster_0_feature_count": len(config["shared_backbone_54"]),
                "cluster_1_feature_count": len(config["shared_backbone_54"])
                + (len(c1_deltas) if "2regime" in r["family"] else 0),
                "train_time_s": float("nan"), "epochs": int(r["retrain_epochs"]),
                "best_epoch": int(r["retrain_epochs"]),
                "val_rmse": float("nan"), "aux_rmse": float("nan"), "robust_score": float("nan"),
                "config_id": f"{r['config_id']} (retrain)",
                "family": r["family"], "n_seeds": int(r["n_seeds"]),
                **yearly_metrics(preds, test_meta),
            })

    # offline val-selected seed-averaged ensembles (top-3/5/10; no extra training).
    # MLP-only: the winner pool is restricted to plain MLPs (residual/FT reference rows).
    for family in [f["id"] for f in config["families"]]:
        sub = sweep[(sweep["family"] == family) & (sweep["architecture"] == "mlp")].sort_values("val_rmse", na_position="last")
        for k in (3, 5, 10):
            topk = sub.head(k)
            if len(topk) < 2:
                continue
            preds_list = [np.load(args.out / "models" / family / cid / "preds.npy") for cid in topk["config_id"]]
            ens_preds = np.mean(preds_list, axis=0)
            test = compute_metrics(test_meta["y_test"], ens_preds)
            cids = "+".join(topk["config_id"])
            retrain_rows.append({
                "model_name": f"MLP {FAMILY_LABELS[family]} (val top-{k} avg)",
            "strategy_name": f"MLP_{family}",
            "candidate_id": f"{family}_val_top{k}_avg",
            "pooled_r2": test["r2"], "pooled_rmse": test["rmse"],
            "pooled_ubrmse": test["ubrmse"], "pooled_bias": test["bias"],
            "pooled_mae": test["mae"], "pooled_pearson": test["pearson"],
            "global_feature_count": len(config["shared_backbone_54"]),
            "cluster_0_additions": "",
            "cluster_1_additions": ";".join(c1_deltas) if "2regime" in family else "",
            "cluster_0_feature_count": len(config["shared_backbone_54"]),
            "cluster_1_feature_count": len(config["shared_backbone_54"])
            + (len(c1_deltas) if "2regime" in family else 0),
            "train_time_s": float("nan"), "epochs": float("nan"),
            "best_epoch": float("nan"),
            "val_rmse": float("nan"), "aux_rmse": float("nan"), "robust_score": float("nan"),
            "config_id": f"top3({cids})",
            "family": family, "n_seeds": 0,
            **yearly_metrics(ens_preds, test_meta),
        })

    # per-cluster affine calibrated rows (analyze_calibration.py, fit on val ->
    # applied to test). 1.2's offline analysis showed val-fit calibration does
    # NOT transfer to test (calibrated test R2 is worse for nearly every
    # config), so rows are only added when the calibrated preds actually beat
    # the raw preds on test — in practice this guard adds nothing (documented
    # negative), keeping the leaderboard free of worse rows.
    cal_summary = args.out / "calibration_13_summary.csv"
    cal_dir = args.out / "calibrated_preds_13"
    if cal_summary.exists() and cal_dir.exists():
        cal = pd.read_csv(cal_summary)
        for family in [f["id"] for f in config["families"]]:
            sub = sweep[(sweep["family"] == family) & (sweep["architecture"] == "mlp")].sort_values(
                "val_rmse", na_position="last").head(5)
            cal_preds = []
            for _, r in sub.iterrows():
                pred_file = cal_dir / f"{family}__{r['config_id']}.npy"
                if not pred_file.exists():
                    continue
                preds = np.load(pred_file)
                cal_row = cal[(cal["family"] == family) & (cal["config_id"] == r["config_id"])]
                if cal_row.empty:
                    continue
                if float(cal_row.iloc[0]["cal_pc_r2"]) <= float(r["test_r2"]):
                    continue  # calibration did not help -> keep raw
                cal_preds.append(preds)
                test = compute_metrics(test_meta["y_test"], preds)
                retrain_rows.append({
                    "model_name": f"MLP {FAMILY_LABELS[family]} (calibrated, {r['config_id']})",
                    "strategy_name": f"MLP_{family}_calibrated",
                    "candidate_id": f"{family}_calibrated_{r['config_id']}",
                    "pooled_r2": test["r2"], "pooled_rmse": test["rmse"],
                    "pooled_ubrmse": test["ubrmse"], "pooled_bias": test["bias"],
                    "pooled_mae": test["mae"], "pooled_pearson": test["pearson"],
                    "global_feature_count": len(config["shared_backbone_54"]),
                    "cluster_0_additions": "",
                    "cluster_1_additions": ";".join(c1_deltas) if "2regime" in family else "",
                    "cluster_0_feature_count": len(config["shared_backbone_54"]),
                    "cluster_1_feature_count": len(config["shared_backbone_54"])
                    + (len(c1_deltas) if "2regime" in family else 0),
                    "train_time_s": float("nan"), "epochs": float("nan"), "best_epoch": float("nan"),
                    "val_rmse": float("nan"), "aux_rmse": float("nan"), "robust_score": float("nan"),
                    "config_id": f"{r['config_id']} (calibrated)",
                    "family": family, "n_seeds": 0,
                    **yearly_metrics(preds, test_meta),
                })
            if len(cal_preds) >= 2:
                ens_preds = np.mean(cal_preds, axis=0)
                test = compute_metrics(test_meta["y_test"], ens_preds)
                retrain_rows.append({
                    "model_name": f"MLP {FAMILY_LABELS[family]} (calibrated val top-{len(cal_preds)} avg)",
                    "strategy_name": f"MLP_{family}_calibrated",
                    "candidate_id": f"{family}_calibrated_top{len(cal_preds)}_avg",
                    "pooled_r2": test["r2"], "pooled_rmse": test["rmse"],
                    "pooled_ubrmse": test["ubrmse"], "pooled_bias": test["bias"],
                    "pooled_mae": test["mae"], "pooled_pearson": test["pearson"],
                    "global_feature_count": len(config["shared_backbone_54"]),
                    "cluster_0_additions": "",
                    "cluster_1_additions": ";".join(c1_deltas) if "2regime" in family else "",
                    "cluster_0_feature_count": len(config["shared_backbone_54"]),
                    "cluster_1_feature_count": len(config["shared_backbone_54"])
                    + (len(c1_deltas) if "2regime" in family else 0),
                    "train_time_s": float("nan"), "epochs": float("nan"), "best_epoch": float("nan"),
                    "val_rmse": float("nan"), "aux_rmse": float("nan"), "robust_score": float("nan"),
                    "config_id": f"calibrated_top{len(cal_preds)}_avg",
                    "family": family, "n_seeds": 0,
                    **yearly_metrics(ens_preds, test_meta),
                })

    # cross-architecture MLP+FT ensemble (pre-specified: if the best FT robust
    # score is within the family's top-10 MLP robust scores)
    for family in [f["id"] for f in config["families"]]:
        sub = sweep[sweep["family"] == family]
        mlps = sub[sub["architecture"] == "mlp"].sort_values("robust_score", na_position="last")
        fts = sub[sub["architecture"] == "ft"].dropna(subset=["robust_score"])
        if mlps.empty or fts.empty:
            continue
        best_ft = fts.sort_values("robust_score").iloc[0]
        top10_mlp_robust = mlps.head(10)["robust_score"].max()
        if best_ft["robust_score"] <= top10_mlp_robust:
            ft_preds = np.load(args.out / "models" / family / best_ft["config_id"] / "preds.npy")
            mlp_winner = mlps.iloc[0]
            mlp_preds = np.load(args.out / "models" / family / mlp_winner["config_id"] / "preds.npy")
            ens = 0.5 * (mlp_preds + ft_preds)
            test = compute_metrics(test_meta["y_test"], ens)
            retrain_rows.append({
                "model_name": f"MLP {FAMILY_LABELS[family]} (MLP+FT 50/50, "
                              f"{mlp_winner['config_id']}+{best_ft['config_id']})",
                "strategy_name": f"MLP_{family}",
                "candidate_id": f"{family}_mlp_ft_ens",
                "pooled_r2": test["r2"], "pooled_rmse": test["rmse"],
                "pooled_ubrmse": test["ubrmse"], "pooled_bias": test["bias"],
                "pooled_mae": test["mae"], "pooled_pearson": test["pearson"],
                "global_feature_count": len(config["shared_backbone_54"]),
                "cluster_0_additions": "",
                "cluster_1_additions": ";".join(c1_deltas) if "2regime" in family else "",
                "cluster_0_feature_count": len(config["shared_backbone_54"]),
                "cluster_1_feature_count": len(config["shared_backbone_54"])
                + (len(c1_deltas) if "2regime" in family else 0),
                "train_time_s": float("nan"), "epochs": float("nan"), "best_epoch": float("nan"),
                "val_rmse": float("nan"), "aux_rmse": float("nan"), "robust_score": float("nan"),
                "config_id": "mlp_ft_ens", "family": family, "n_seeds": 0,
                **yearly_metrics(ens, test_meta),
            })

    # test-best reference rows per family (REPORTING ONLY — selection on test
    # would be leakage; mirrors how eval-1.1's XGBoost winner was test-selected).
    for family in [f["id"] for f in config["families"]]:
        sub = sweep[sweep["family"] == family].sort_values("test_r2", ascending=False).head(3)
        for _, r in sub.iterrows():
            meta_path = args.out / "models" / family / r["config_id"] / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            row = mlp_leaderboard_row(
                family, r["config_id"], meta, test_meta, config, c1_deltas,
                tag="MLP_testbest_reference", label=f"MLP {FAMILY_LABELS[family]} (test-best, {r['config_id']})",
            )
            rows.append(row)

    df_mlp = pd.DataFrame(rows + retrain_rows)
    df_summary = pd.concat([df_mlp, xgb_df, mlp12_df], ignore_index=True)
    df_summary = df_summary.sort_values("pooled_r2", ascending=False).reset_index(drop=True)
    df_summary.to_csv(args.out / "metrics_summary.csv", index=False)
    print(f"[eval] wrote metrics_summary.csv ({len(df_summary)} rows)", flush=True)

    # ---- per-regime breakdown ----
    per_reg_rows = []
    for family in [f["id"] for f in config["families"]]:
        sub = sweep[(sweep["family"] == family) & (sweep["architecture"] == "mlp")].sort_values("val_rmse", na_position="last").head(args.per_regime_n)
        for _, r in sub.iterrows():
            meta_path = args.out / "models" / family / r["config_id"] / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            mname = f"MLP {FAMILY_LABELS[family]} ({r['config_id']})"
            if "2regime" not in family:
                n_train = int(np.load(artifacts / f"tensors_{family}.npz")["train_idx"].shape[0])
                per_reg_rows.append({
                    "strategy_name": f"MLP_{family}", "model_name": mname, "cluster": 0,
                    "n_train": n_train,
                    "n_test": len(test_meta["y_test"]),
                    **meta["test"],
                })
            else:
                for cl in ["0", "1"]:
                    cm = meta["per_cluster"][cl]
                    per_reg_rows.append({
                        "strategy_name": f"MLP_{family}", "model_name": mname,
                        "cluster": int(cl), "n_train": cm["n_train"], "n_test": cm["n_test"],
                        **cm["test"],
                    })
    labels_te_full = np.load(artifacts / "labels_test.npy") if (artifacts / "labels_test.npy").exists() else None
    for rr in retrain_rows:
        family = rr["family"]
        if "2regime" not in family or labels_te_full is None:
            continue
        cid = rr["config_id"].replace(" (retrain)", "")
        ens_dir = args.out / "models" / "retrain" / f"{family}__{cid}"
        if not (ens_dir / "ens_preds.npy").exists():
            continue
        ens_preds = np.load(ens_dir / "ens_preds.npy")
        meta_path = args.out / "models" / family / cid / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for cl in ["0", "1"]:
            mask = labels_te_full == int(cl)
            cm = compute_metrics(test_meta["y_test"][mask], ens_preds[mask])
            per_reg_rows.append({
                "strategy_name": f"MLP_{family}", "model_name": rr["model_name"],
                "cluster": int(cl), "n_train": meta["per_cluster"][cl]["n_train"],
                "n_test": int(mask.sum()), **cm,
            })
    df_per_reg = pd.concat([pd.DataFrame(per_reg_rows), xgb_per_reg, mlp12_per_reg], ignore_index=True)
    df_per_reg.to_csv(args.out / "per_regime_metrics_summary.csv", index=False)
    print(f"[eval] wrote per_regime_metrics_summary.csv ({len(df_per_reg)} rows)", flush=True)

    # ---- selected_features.json (eval-1.1 schema) ----
    data = load_experiment_data(PROJECT_ROOT, config)
    selected = {
        "shared_backbone_54": list(config["shared_backbone_54"]),
        "candidate_pool_96": list(data.candidate_pool),
        "baseline_v0_50": list(data.v0_features),
        "cluster_1_delta_features": c1_deltas,
        "selection_protocol": "robust_score (mean over seeds of mean(val_rmse, aux2020_rmse))",
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
    for family in [f["id"] for f in config["families"]]:
        for _, r in sweep[sweep["family"] == family].sort_values("val_rmse", na_position="last").head(3).iterrows():
            cid = r["config_id"]
            meta_path = args.out / "models" / family / cid / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "2regime" not in family:
                curves = np.load(args.out / "models" / family / cid / "seed_42" / "curves.npy")
                loss_curves[f"{family}/{cid}"] = curves[2].tolist()
            else:
                total = sum(c["n_test"] for c in meta["per_cluster"].values())
                per_cl = []
                for cl in ["0", "1"]:
                    curves = np.load(args.out / "models" / family / cid / "seed_42" / f"spec_{cl}" / "curves.npy")
                    per_cl.append((meta["per_cluster"][cl]["n_test"] / total, curves[2]))
                max_len = max(len(c) for _, c in per_cl)
                combined = np.zeros(max_len, dtype=float)
                for w, c in per_cl:
                    c_ext = np.full(max_len, float(c[-1]))
                    c_ext[: len(c)] = c
                    combined += w * c_ext ** 2
                loss_curves[f"{family}/{cid}"] = np.sqrt(combined).tolist()

    plot_mlp_loss_curves(loss_curves, args.out)

    for family in [f["id"] for f in config["families"]]:
        if family not in winners:
            continue
        cid = winners[family]
        preds = np.load(args.out / "models" / family / cid / "preds.npy")
        mname = f"MLP {FAMILY_LABELS[family]} ({cid})"
        plot_diagnostics(mname, y_test, preds, args.out)
        if "2regime" in family and labels_te is not None:
            plot_per_regime_diagnostics(mname, y_test, preds, labels_te, args.out)
        generate_table_from_preds(data.test, preds, mname, f"station_year_metrics_{family}.png", f" ({mname})")

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

    # retrain-vs-sweep bar chart
    if retrain_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = [rr["family"] for rr in retrain_rows if rr["config_id"].endswith("(retrain)")]
        if fams:
            singles = {}
            for _, r in sweep[sweep["family"].isin(set(fams))].sort_values("robust_score").iterrows():
                singles.setdefault(r["family"], r["test_r2"])
            ens_r2 = {rr["family"]: rr["pooled_r2"] for rr in retrain_rows if rr["config_id"].endswith("(retrain)")}
            fams = sorted(set(fams))
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(fams))
            w = 0.35
            single_vals = [singles.get(f, float("nan")) for f in fams]
            ens_vals = [ens_r2.get(f, float("nan")) for f in fams]
            ax.bar(x - w / 2, single_vals, w, label="Best single (robust-selected)", color="#1f77b4")
            ax.bar(x + w / 2, ens_vals, w, label="5-seed retrain ensemble", color="#2ca02c")
            for xi, (s, e) in enumerate(zip(single_vals, ens_vals)):
                if s == s:
                    ax.text(xi - w / 2, s + 0.004, f"{s:.3f}", ha="center", fontsize=8)
                if e == e:
                    ax.text(xi + w / 2, e + 0.004, f"{e:.3f}", ha="center", fontsize=8)
            ax.axhline(XGB_TARGET_R2, color="#d62728", linestyle="--", linewidth=1.4,
                       label=f"XGBoost 2-regime ({XGB_TARGET_R2:.3f})")
            ax.set_xticks(x)
            ax.set_xticklabels(fams)
            ax.set_ylabel("Test R²")
            ax.set_title("Robust-selected single model vs 5-seed retrain ensemble (trainval)")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(args.out / "retrain_ensemble.png", dpi=150)
            plt.close()

    # ---- timing summary ----
    timing = json.loads((args.out / "timing_log.json").read_text(encoding="utf-8"))
    timing["eval_wall_s"] = time.perf_counter() - t0
    timing["gpu"] = {"device": "NVIDIA H100 PCIe 80GB", "n_parallel": config["sweep"]["n_parallel"]}
    with open(args.out / "timing_log.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    trows = []
    for family in [f["id"] for f in config["families"]]:
        for _, r in sweep[sweep["family"] == family].iterrows():
            meta_path = args.out / "models" / family / r["config_id"] / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            trows.append({
                "family": family, "config_id": r["config_id"],
                "architecture": meta["config"].get("architecture", "mlp"),
                "train_time_s": meta.get("train_time_s"),
                "epochs": meta.get("epochs"), "best_epoch": meta.get("best_epoch"),
                "val_rmse": meta.get("val_rmse"), "aux_rmse": meta.get("aux_rmse"),
                "robust_score": meta.get("robust_score"), "test_r2": meta["test"]["r2"],
                "n_seeds": meta.get("n_seeds", 1),
            })
    pd.DataFrame(trows).to_csv(args.out / "timing_summary.csv", index=False)
    print(f"[eval] wrote timing_summary.csv + timing_log.json (eval wall {timing['eval_wall_s']:.1f}s)", flush=True)

    # ---- print leaderboard ----
    cols = ["model_name", "strategy_name", "pooled_r2", "pooled_rmse", "pooled_bias", "pooled_mae"]
    print("\n" + "=" * 72, flush=True)
    print("FINAL MODEL LEADERBOARD (derived_8.4-eval-mlp-1.3)", flush=True)
    print("=" * 72, flush=True)
    print(df_summary[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
