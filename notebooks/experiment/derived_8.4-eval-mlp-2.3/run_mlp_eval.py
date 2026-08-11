#!/usr/bin/env python3
"""Aggregate the derived_8.4-eval-mlp-2.3 sweep into the eval-1.1-style report.

No retraining: the sweep already produced per-seed test predictions/models per
job. Picks the winners per family by 3-seed mean val RMSE among the honest
architectures (mlp / fg / plr — only mlp configs run in 2.3; fg/plr are
documented negatives from 2.0, swa a documented negative from 2.1, and
54-family 3-layer a documented 2.2 negative — only the 3 bit-identity
anchors + 2 re-check probes use it), reports the selection diagnostics +
Spearman correlations so the fix is auditable, and builds the leaderboard
(metrics_summary.csv), per-regime breakdown, offline val top-k and cross-family
ensembles (the 54/96/mixed families are complementary: 54 near-unbiased, 96
extrapolation, mixed per-cluster-optimal), selected_features.json, all figures,
and the timing summary. Reference rows include the XGBoost eval-1.1 baselines,
the mlp-1.3 rows, the mlp-2.0 rows, the mlp-2.1 rows, and NEW mlp-2.2 rows
(val-selected winners, test-best refs, val top-k ensembles, 5-seed champ,
cross-family ensemble) read from the 2.2 metrics_summary.csv via
config.mlp22_reference_file.

Usage:
    python run_mlp_eval.py [--config config.yaml] [--out .] [--top-n 5]
                           [--per-regime-n 3]
"""

from __future__ import annotations

import argparse
import json
import re
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
MLP13_DIR = EXP_DIR.parent / "derived_8.4-eval-mlp-1.3"
MLP20_DIR = EXP_DIR.parent / "derived_8.4-eval-mlp-2.0"
MLP21_DIR = EXP_DIR.parent / "derived_8.4-eval-mlp-2.1"
MLP22_DIR = EXP_DIR.parent / "derived_8.4-eval-mlp-2.2"
sys.path.insert(0, str(EVAL11_DIR))
sys.path.insert(0, str(EXP_DIR))

from eval11.data import load_experiment_data  # noqa: E402
from eval11.evaluator import compute_metrics  # noqa: E402
from generate_station_year_table import generate_table_from_preds  # noqa: E402
from mlp23.plots import (  # noqa: E402
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
    "2regime_mixed": "2-Regime-Mixed",
}

# Architectures that may enter the winner pool (residual/ft are reference-only).
HONEST_ARCHS = ("mlp", "fg", "plr")

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


def load_mlp13_references(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best MLP-1.3 rows per family (val-selected winner + test-best reference).

    mlp-1.3's metrics_summary.csv uses strategy_name "MLP_<family>" for both
    individual config rows and offline val top-k ensembles; the honest
    val-selected winner is the row with the minimum (non-NaN) val_rmse.
    """
    ref_file = PROJECT_ROOT / Path(config["mlp13_reference_file"])
    df = pd.read_csv(ref_file)
    fam_map = {"2regime_96": "2regime_96", "2regime_54": "2regime_54"}
    keep = []
    for fam in fam_map:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        sub = sub.copy()
        # val-selected winner = min val_rmse among the honest MLP_<fam> rows
        honest = sub[sub["strategy_name"] == f"MLP_{fam}"].dropna(subset=["val_rmse"]).sort_values("val_rmse")
        if not honest.empty:
            r = honest.iloc[0].copy()
            r["strategy_name"] = "MLP_1.3_Reference"
            r["model_name"] = f"MLP-1.3 {FAMILY_LABELS[fam]} (val_sel: {r['config_id']})"
            keep.append(r)
        # test-best reference rows (reporting only)
        tb = sub[sub["strategy_name"] == "MLP_testbest_reference"].sort_values("pooled_r2", ascending=False).head(1)
        if not tb.empty:
            r = tb.iloc[0].copy()
            r["strategy_name"] = "MLP_1.3_Reference"
            r["model_name"] = f"MLP-1.3 {FAMILY_LABELS[fam]} (test_best: {r['config_id']})"
            keep.append(r)
        # best offline ensemble row (val top-10 avg)
        ens = sub[sub["strategy_name"] == f"MLP_{fam}"].dropna(subset=["val_rmse"])
        ens_topk = sub[sub["model_name"].str.contains("val top-10 avg", regex=False)]
        if not ens_topk.empty:
            r = ens_topk.sort_values("pooled_r2", ascending=False).iloc[0].copy()
            r["strategy_name"] = "MLP_1.3_Reference"
            r["model_name"] = f"MLP-1.3 {FAMILY_LABELS[fam]} (val top-10 avg)"
            keep.append(r)
    df_ref = pd.DataFrame(keep) if keep else pd.DataFrame(columns=df.columns)
    # per-regime rows for the 1.3 val-selected winners
    per_reg = pd.read_csv(MLP13_DIR / "per_regime_metrics_summary.csv")
    per_reg = per_reg[per_reg["strategy_name"].isin(["MLP_2regime_96", "MLP_2regime_54"])].copy()
    winners = {}
    for fam in fam_map:
        sub = df[df["family"] == fam]
        honest = sub[sub["strategy_name"] == f"MLP_{fam}"].dropna(subset=["val_rmse"]).sort_values("val_rmse")
        if not honest.empty:
            winners[f"MLP_{fam}"] = honest.iloc[0]["config_id"]
    keep_rows = []
    for strat, winner in winners.items():
        sub = per_reg[per_reg["strategy_name"] == strat]
        keep_rows.append(sub[sub["model_name"].str.contains(winner, regex=False)])
    per_reg = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame(columns=per_reg.columns)
    per_reg["strategy_name"] = "MLP_1.3_Reference"
    return df_ref, per_reg


def load_mlp20_references(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best MLP-2.0 rows (val-selected winner, test-best ref, val top-5 ensemble,
    cross-family ensemble) from the 2.0 metrics_summary.csv.

    2.0's metrics_summary uses strategy_name "MLP_<family>" for individual
    config rows and offline val top-k ensembles; the honest val-selected
    winner is the row with the minimum (non-NaN) val_rmse. 2.1's headline
    comparison is against 2.0's mixed val top-5 ensemble (0.8003) and its
    2-seed honest single (0.7903).
    """
    ref_file = PROJECT_ROOT / Path(config["mlp20_reference_file"])
    df = pd.read_csv(ref_file)
    fam_map = {"2regime_96": "2regime_96", "2regime_54": "2regime_54", "2regime_mixed": "2regime_mixed"}
    keep = []
    for fam in fam_map:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        sub = sub.copy()
        # val-selected winner = min val_rmse among the honest MLP_<fam> rows
        honest = sub[sub["strategy_name"] == f"MLP_{fam}"].dropna(subset=["val_rmse"]).sort_values("val_rmse")
        if not honest.empty:
            r = honest.iloc[0].copy()
            r["strategy_name"] = "MLP_2.0_Reference"
            r["model_name"] = f"MLP-2.0 {FAMILY_LABELS[fam]} (val_sel: {r['config_id']})"
            keep.append(r)
        # test-best reference rows (reporting only)
        tb = sub[sub["strategy_name"] == "MLP_testbest_reference"].sort_values("pooled_r2", ascending=False).head(1)
        if not tb.empty:
            r = tb.iloc[0].copy()
            r["strategy_name"] = "MLP_2.0_Reference"
            r["model_name"] = f"MLP-2.0 {FAMILY_LABELS[fam]} (test_best: {r['config_id']})"
            keep.append(r)
        # best offline ensemble row (val top-k avg) per family — keep its name
        ens_topk = sub[(sub["strategy_name"] == f"MLP_{fam}")
                       & sub["model_name"].str.contains("val top-", regex=False)]
        if not ens_topk.empty:
            r = ens_topk.sort_values("pooled_r2", ascending=False).iloc[0].copy()
            r["strategy_name"] = "MLP_2.0_Reference"
            r["model_name"] = "MLP-2.0 " + r["model_name"].removeprefix("MLP ")
            keep.append(r)
    # cross-family ensemble row (2.0's strongest cross-family result)
    cf = df[df["strategy_name"] == "MLP_cross_family"].sort_values("pooled_r2", ascending=False)
    if not cf.empty:
        r = cf.iloc[0].copy()
        r["strategy_name"] = "MLP_2.0_Reference"
        r["model_name"] = "MLP-2.0 cross-family (val winners)"
        keep.append(r)
    df_ref = pd.DataFrame(keep) if keep else pd.DataFrame(columns=df.columns)

    # per-regime rows for the 2.0 val-selected winners
    per_reg = pd.read_csv(MLP20_DIR / "per_regime_metrics_summary.csv")
    per_reg = per_reg[per_reg["strategy_name"].isin([f"MLP_{f}" for f in fam_map])].copy()
    winners = {}
    for fam in fam_map:
        sub = df[df["family"] == fam]
        honest = sub[sub["strategy_name"] == f"MLP_{fam}"].dropna(subset=["val_rmse"]).sort_values("val_rmse")
        if not honest.empty:
            winners[f"MLP_{fam}"] = honest.iloc[0]["config_id"]
    keep_rows = []
    for strat, winner in winners.items():
        sub = per_reg[per_reg["strategy_name"] == strat]
        # strict match: winner + ")" so a winner id that is a prefix of another
        # (e.g. ..._lr1e-3 vs ..._lr1e-3_swa) does not pull in the sibling rows
        keep_rows.append(sub[sub["model_name"].str.contains(re.escape(winner) + r"\)", regex=True)])
    per_reg = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame(columns=per_reg.columns)
    per_reg["strategy_name"] = "MLP_2.0_Reference"
    return df_ref, per_reg


def load_mlp21_references(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best MLP-2.1 rows (val-selected winner, test-best ref, val top-k
    ensemble, 5-seed champ, cross-family ensemble) from the 2.1
    metrics_summary.csv.

    2.1's metrics_summary uses the same schema as 2.0's ("MLP_<family>" for
    individual config rows and offline ensembles; "MLP_testbest_reference"
    for test-best rows; "MLP_cross_family" for the cross-family ensemble).
    2.2's headline comparison is against 2.1's mixed val winner (0.7844) and
    its test-best (0.7940), plus the 2.0 mixed val top-5 ensemble (0.8003).
    """
    ref_file = PROJECT_ROOT / Path(config["mlp21_reference_file"])
    df = pd.read_csv(ref_file)
    fam_map = {"2regime_96": "2regime_96", "2regime_54": "2regime_54", "2regime_mixed": "2regime_mixed"}
    keep = []
    for fam in fam_map:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        sub = sub.copy()
        # val-selected winner = min val_rmse among the honest MLP_<fam> rows
        honest = sub[sub["strategy_name"] == f"MLP_{fam}"].dropna(subset=["val_rmse"]).sort_values("val_rmse")
        if not honest.empty:
            r = honest.iloc[0].copy()
            r["strategy_name"] = "MLP_2.1_Reference"
            r["model_name"] = f"MLP-2.1 {FAMILY_LABELS[fam]} (val_sel: {r['config_id']})"
            keep.append(r)
        # test-best reference rows (reporting only)
        tb = sub[sub["strategy_name"] == "MLP_testbest_reference"].sort_values("pooled_r2", ascending=False).head(1)
        if not tb.empty:
            r = tb.iloc[0].copy()
            r["strategy_name"] = "MLP_2.1_Reference"
            r["model_name"] = f"MLP-2.1 {FAMILY_LABELS[fam]} (test_best: {r['config_id']})"
            keep.append(r)
        # best offline ensemble row (val top-k avg) per family — keep its name
        ens_topk = sub[(sub["strategy_name"] == f"MLP_{fam}")
                       & sub["model_name"].str.contains("val top-", regex=False)]
        if not ens_topk.empty:
            r = ens_topk.sort_values("pooled_r2", ascending=False).iloc[0].copy()
            r["strategy_name"] = "MLP_2.1_Reference"
            r["model_name"] = "MLP-2.1 " + r["model_name"].removeprefix("MLP ")
            keep.append(r)
        # 5-seed champion ensemble (if present)
        champ = sub[sub["model_name"].str.contains("5-seed champ", regex=False)]
        if not champ.empty:
            r = champ.sort_values("pooled_r2", ascending=False).iloc[0].copy()
            r["strategy_name"] = "MLP_2.1_Reference"
            r["model_name"] = "MLP-2.1 " + r["model_name"].removeprefix("MLP ")
            keep.append(r)
    # cross-family ensemble row (2.1's strongest cross-family result)
    cf = df[df["strategy_name"] == "MLP_cross_family"].sort_values("pooled_r2", ascending=False)
    if not cf.empty:
        r = cf.iloc[0].copy()
        r["strategy_name"] = "MLP_2.1_Reference"
        r["model_name"] = "MLP-2.1 cross-family (val winners)"
        keep.append(r)
    df_ref = pd.DataFrame(keep) if keep else pd.DataFrame(columns=df.columns)

    # per-regime rows for the 2.1 val-selected winners
    per_reg = pd.read_csv(MLP21_DIR / "per_regime_metrics_summary.csv")
    per_reg = per_reg[per_reg["strategy_name"].isin([f"MLP_{f}" for f in fam_map])].copy()
    winners = {}
    for fam in fam_map:
        sub = df[df["family"] == fam]
        honest = sub[sub["strategy_name"] == f"MLP_{fam}"].dropna(subset=["val_rmse"]).sort_values("val_rmse")
        if not honest.empty:
            winners[f"MLP_{fam}"] = honest.iloc[0]["config_id"]
    keep_rows = []
    for strat, winner in winners.items():
        sub = per_reg[per_reg["strategy_name"] == strat]
        keep_rows.append(sub[sub["model_name"].str.contains(re.escape(winner) + r"\)", regex=True)])
    per_reg = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame(columns=per_reg.columns)
    per_reg["strategy_name"] = "MLP_2.1_Reference"
    return df_ref, per_reg


def _strip_mlp_prefix(name: str) -> str:
    """Strip a leading 'MLP ' or 'MLP-x.y ' prefix (2.2's summary embeds
    earlier-version reference rows whose names already carry their own
    'MLP-2.1 ' prefix)."""
    return re.sub(r"^MLP(?:-\d\.\d)? ", "", name)


def load_mlp22_references(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best MLP-2.2 rows (val-selected winner, test-best ref, val top-k
    ensemble, 5-seed champ, cross-family ensemble) from the 2.2
    metrics_summary.csv.

    NEW in 2.3: 2.3's headline comparison is against 2.2's results — the
    mixed val winner (0.7809), the cross-family ensemble (0.7885), the
    series-best single test row (54 w320x320_d0.4_huber0.2_gelu_lr6e-4,
    0.7973, reporting only), and the 2.0 mixed val top-5 ensemble (0.8003).
    """
    ref_file = PROJECT_ROOT / Path(config["mlp22_reference_file"])
    df = pd.read_csv(ref_file)
    fam_map = {"2regime_96": "2regime_96", "2regime_54": "2regime_54", "2regime_mixed": "2regime_mixed"}
    keep = []
    for fam in fam_map:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        sub = sub.copy()
        # val-selected winner = min val_rmse among the honest MLP_<fam> rows
        honest = sub[sub["strategy_name"] == f"MLP_{fam}"].dropna(subset=["val_rmse"]).sort_values("val_rmse")
        if not honest.empty:
            r = honest.iloc[0].copy()
            r["strategy_name"] = "MLP_2.2_Reference"
            r["model_name"] = f"MLP-2.2 {FAMILY_LABELS[fam]} (val_sel: {r['config_id']})"
            keep.append(r)
        # test-best reference rows (reporting only)
        tb = sub[sub["strategy_name"] == "MLP_testbest_reference"].sort_values("pooled_r2", ascending=False).head(1)
        if not tb.empty:
            r = tb.iloc[0].copy()
            r["strategy_name"] = "MLP_2.2_Reference"
            r["model_name"] = f"MLP-2.2 {FAMILY_LABELS[fam]} (test_best: {r['config_id']})"
            keep.append(r)
        # best offline ensemble row (val top-k avg) per family — keep its name
        ens_topk = sub[(sub["strategy_name"] == f"MLP_{fam}")
                       & sub["model_name"].str.contains("val top-", regex=False)]
        if not ens_topk.empty:
            r = ens_topk.sort_values("pooled_r2", ascending=False).iloc[0].copy()
            r["strategy_name"] = "MLP_2.2_Reference"
            r["model_name"] = "MLP-2.2 " + _strip_mlp_prefix(r["model_name"])
            keep.append(r)
        # 5-seed champion ensemble (if present)
        champ = sub[sub["model_name"].str.contains("5-seed champ", regex=False)]
        if not champ.empty:
            r = champ.sort_values("pooled_r2", ascending=False).iloc[0].copy()
            r["strategy_name"] = "MLP_2.2_Reference"
            r["model_name"] = "MLP-2.2 " + _strip_mlp_prefix(r["model_name"])
            keep.append(r)
    # cross-family ensemble row (2.2's strongest cross-family result)
    cf = df[df["strategy_name"] == "MLP_cross_family"].sort_values("pooled_r2", ascending=False)
    if not cf.empty:
        r = cf.iloc[0].copy()
        r["strategy_name"] = "MLP_2.2_Reference"
        r["model_name"] = "MLP-2.2 cross-family (val winners)"
        keep.append(r)
    df_ref = pd.DataFrame(keep) if keep else pd.DataFrame(columns=df.columns)

    # per-regime rows for the 2.2 val-selected winners
    per_reg = pd.read_csv(MLP22_DIR / "per_regime_metrics_summary.csv")
    per_reg = per_reg[per_reg["strategy_name"].isin([f"MLP_{f}" for f in fam_map])].copy()
    winners = {}
    for fam in fam_map:
        sub = df[df["family"] == fam]
        honest = sub[sub["strategy_name"] == f"MLP_{fam}"].dropna(subset=["val_rmse"]).sort_values("val_rmse")
        if not honest.empty:
            winners[f"MLP_{fam}"] = honest.iloc[0]["config_id"]
    keep_rows = []
    for strat, winner in winners.items():
        sub = per_reg[per_reg["strategy_name"] == strat]
        keep_rows.append(sub[sub["model_name"].str.contains(re.escape(winner) + r"\)", regex=True)])
    per_reg = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame(columns=per_reg.columns)
    per_reg["strategy_name"] = "MLP_2.2_Reference"
    return df_ref, per_reg


def mlp_leaderboard_row(
    family: str, config_id: str, meta: dict, test_meta: dict, config: dict, c1_deltas: list[str],
    *, tag: str = "", label: str | None = None, out_dir: Path | None = None, n_c0: int | None = None,
) -> dict:
    backbone = list(config["shared_backbone_54"])
    # cluster-0 feature count: 96-pool families use the candidate pool (96),
    # the 54 family (and 2.0's "global" rows) use the shared 54-feature backbone.
    if n_c0 is None:
        n_c0 = len(backbone)
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
        "global_feature_count": n_c0,
        "cluster_0_additions": "",
        "cluster_1_additions": ";".join(c1_deltas) if "2regime" in family else "",
        "cluster_0_feature_count": n_c0,
        "cluster_1_feature_count": n_c0 + (len(c1_deltas) if "2regime" in family else 0),
        "train_time_s": meta.get("train_time_s", float("nan")),
        "epochs": meta.get("epochs", float("nan")),
        "best_epoch": meta.get("best_epoch", float("nan")),
        "val_rmse": meta.get("val_rmse", float("nan")),
        "val_rmse_live": meta.get("val_rmse_live", float("nan")),
        "val_rmse_swa": meta.get("val_rmse_swa", float("nan")),
        "deployed": meta.get("deployed", "live"),
        "aux_rmse": meta.get("aux_rmse", float("nan")),
        "robust_score": meta.get("robust_score", float("nan")),
        "config_id": config_id,
        "family": family,
        "n_seeds": meta.get("n_seeds", 1),
    }
    preds = np.load((out_dir or EXP_DIR) / "models" / family / config_id / "preds.npy")
    row.update(yearly_metrics(preds, test_meta))
    return row


def print_selection_diagnostic(sweep: pd.DataFrame, test_meta: dict, out: Path) -> None:
    """Selection diagnostics: val-RMSE ranking + Spearman vs test (aux reported only)."""
    print("\n" + "=" * 78, flush=True)
    print("SELECTION PROTOCOL v9 DIAGNOSTIC (selection = 3-seed mean val RMSE; robust/aux reported)", flush=True)
    print("=" * 78, flush=True)
    for family in FAMILY_LABELS:
        sub = sweep[sweep["family"] == family].dropna(subset=["test_r2"]).copy()
        if sub.empty:
            continue
        sub = sub.sort_values("val_rmse", na_position="last").reset_index(drop=True)
        print(f"\n--- {family} top-10 by val RMSE (selection metric; test R2 for reference) ---", flush=True)
        cols = ["config_id", "architecture", "n_seeds", "val_rmse", "aux_rmse", "test_r2", "test_bias"]
        show = sub.head(10)[cols].copy()
        print(show.to_string(index=False), flush=True)

        # Spearman: does the selection metric rank correlate with test R2?
        for metric, label in [("val_rmse", "val_rmse"), ("robust_score", "robust_score")]:
            valid = sub.dropna(subset=[metric, "test_r2"])
            if len(valid) >= 8:
                rho, p = spearmanr(valid[metric], valid["test_r2"])
                print(f"  Spearman({label}, test_r2) = {rho:+.3f} (p={p:.3f}, n={len(valid)})", flush=True)
        # what would val-only selection have picked?
        honest_sub = sub[sub["architecture"].isin(HONEST_ARCHS)]
        val_winner = sub.sort_values("val_rmse").iloc[0]
        val_honest = honest_sub.sort_values("val_rmse").iloc[0]
        test_best = sub.sort_values("test_r2", ascending=False).iloc[0]
        print(f"  val winner (all)    : {val_winner['config_id']} (test_r2={val_winner['test_r2']:.4f})", flush=True)
        print(f"  val winner (honest) : {val_honest['config_id']} (test_r2={val_honest['test_r2']:.4f})", flush=True)
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

    data = load_experiment_data(PROJECT_ROOT, config)  # hoisted: needed for the leaderboard feature counts
    artifacts = args.out / "artifacts"
    test_meta = load_test_meta(artifacts)
    c1_deltas = load_cluster_deltas(config, EXP_DIR)

    sweep = pd.read_csv(args.out / "sweep_results.csv")
    xgb_df, xgb_per_reg = load_xgb_references(config)
    mlp13_df, mlp13_per_reg = load_mlp13_references(config)
    mlp20_df, mlp20_per_reg = load_mlp20_references(config)
    mlp21_df, mlp21_per_reg = load_mlp21_references(config)
    mlp22_df, mlp22_per_reg = load_mlp22_references(config)

    print_selection_diagnostic(sweep, test_meta, args.out)

    rows = []
    winners: dict[str, str] = {}
    for family in [f["id"] for f in config["families"]]:
        sub = sweep[(sweep["family"] == family) & sweep["architecture"].isin(HONEST_ARCHS)] \
            .sort_values("val_rmse", na_position="last")
        if sub.empty:
            print(f"[warn] no results for family {family}", flush=True)
            continue
        winners[family] = sub.iloc[0]["config_id"]
        for _, r in sub.head(args.top_n).iterrows():
            meta_path = args.out / "models" / family / r["config_id"] / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            n_c0 = len(data.candidate_pool) if family in ("2regime_96", "2regime_mixed") else len(config["shared_backbone_54"])
            rows.append(mlp_leaderboard_row(family, r["config_id"], meta, test_meta, config, c1_deltas,
                                            out_dir=args.out, n_c0=n_c0))

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
    # Winner pool = honest architectures (mlp/fg/plr; residual/FT reference rows).
    for family in [f["id"] for f in config["families"]]:
        sub = sweep[(sweep["family"] == family) & sweep["architecture"].isin(HONEST_ARCHS)] \
            .sort_values("val_rmse", na_position="last")
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

    # 5-seed champion ensembles (run_mlp_champion.py, extra stability seeds on
    # the val-selected winners — no trainval retrain, which is a documented
    # negative). Read the ens dirs if present.
    champion_root = args.out / "models" / "champion"
    if champion_root.exists():
        for ens_dir in sorted(champion_root.iterdir()):
            emeta_file = ens_dir / "ens_meta.json"
            if not emeta_file.exists():
                continue
            emeta = json.loads(emeta_file.read_text(encoding="utf-8"))
            preds = np.load(ens_dir / "ens_preds.npy")
            test = emeta["test"]
            fam, cid = emeta["family"], emeta["config_id"]
            retrain_rows.append({
                "model_name": f"MLP {FAMILY_LABELS[fam]} (5-seed champ, {cid})",
                "strategy_name": f"MLP_{fam}",
                "candidate_id": f"{fam}_{cid}_champ5",
                "pooled_r2": test["r2"], "pooled_rmse": test["rmse"],
                "pooled_ubrmse": test["ubrmse"], "pooled_bias": test["bias"],
                "pooled_mae": test["mae"], "pooled_pearson": test["pearson"],
                "global_feature_count": len(config["shared_backbone_54"]),
                "cluster_0_additions": "",
                "cluster_1_additions": ";".join(c1_deltas) if "2regime" in fam else "",
                "cluster_0_feature_count": len(config["shared_backbone_54"]),
                "cluster_1_feature_count": len(config["shared_backbone_54"])
                + (len(c1_deltas) if "2regime" in fam else 0),
                "train_time_s": float("nan"), "epochs": float("nan"), "best_epoch": float("nan"),
                "val_rmse": float("nan"), "aux_rmse": float("nan"), "robust_score": float("nan"),
                "config_id": f"{cid} (5-seed champ)",
                "family": fam, "n_seeds": int(emeta.get("n_seeds", 5)),
                **yearly_metrics(preds, test_meta),
            })

    # NEW in 2.0 — cross-family ensembles (no extra training): average the
    # val-selected winner preds across families. The families are complementary
    # (54: near-unbiased; 96: extrapolation/OOD; mixed: per-cluster-optimal).
    cf_preds: list[np.ndarray] = []
    cf_labels: list[str] = []
    for family in [f["id"] for f in config["families"]]:
        sub = sweep[(sweep["family"] == family) & sweep["architecture"].isin(HONEST_ARCHS)] \
            .sort_values("val_rmse", na_position="last")
        if sub.empty:
            continue
        cid = sub.iloc[0]["config_id"]
        p = args.out / "models" / family / cid / "preds.npy"
        if p.exists():
            cf_preds.append(np.load(p))
            cf_labels.append(f"{family}/{cid}")
    if len(cf_preds) >= 2:
        ens_preds = np.mean(cf_preds, axis=0)
        test = compute_metrics(test_meta["y_test"], ens_preds)
        retrain_rows.append({
            "model_name": f"MLP cross-family (val winners: {' + '.join(cf_labels)})",
            "strategy_name": "MLP_cross_family",
            "candidate_id": "cross_family_val_winners",
            "pooled_r2": test["r2"], "pooled_rmse": test["rmse"],
            "pooled_ubrmse": test["ubrmse"], "pooled_bias": test["bias"],
            "pooled_mae": test["mae"], "pooled_pearson": test["pearson"],
            "global_feature_count": len(config["shared_backbone_54"]),
            "cluster_0_additions": "",
            "cluster_1_additions": ";".join(c1_deltas),
            "cluster_0_feature_count": len(config["shared_backbone_54"]),
            "cluster_1_feature_count": len(config["shared_backbone_54"]) + len(c1_deltas),
            "train_time_s": float("nan"), "epochs": float("nan"), "best_epoch": float("nan"),
            "val_rmse": float("nan"), "aux_rmse": float("nan"), "robust_score": float("nan"),
            "config_id": "cross_family", "family": "mixed", "n_seeds": 0,
            **yearly_metrics(ens_preds, test_meta),
        })

    # per-cluster affine calibrated rows (analyze_calibration.py, fit on val ->
    # applied to test). 1.2/1.3 documented that val-fit calibration does NOT
    # transfer to test (calibrated test R2 is worse for nearly every config),
    # so rows are only added when the calibrated preds actually beat the raw
    # preds on test — in practice this guard adds nothing (documented negative).
    # Not run in the 2.0 sweep; the guard stays so a tag-20 recheck would slot in.
    cal_summary = args.out / "calibration_20_summary.csv"
    cal_dir = args.out / "calibrated_preds_20"
    if cal_summary.exists() and cal_dir.exists():
        cal = pd.read_csv(cal_summary)
        for family in [f["id"] for f in config["families"]]:
            sub = sweep[(sweep["family"] == family) & sweep["architecture"].isin(HONEST_ARCHS)].sort_values(
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

    # cross-architecture MLP+FT ensemble (reference-only; FT is a documented
    # failure in mlp-1.1/1.2, so this usually produces no row)
    for family in [f["id"] for f in config["families"]]:
        sub = sweep[sweep["family"] == family]
        mlps = sub[sub["architecture"].isin(HONEST_ARCHS)].sort_values("robust_score", na_position="last")
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
                out_dir=args.out,
                n_c0=len(data.candidate_pool) if family in ("2regime_96", "2regime_mixed") else len(config["shared_backbone_54"]),
            )
            rows.append(row)

    df_mlp = pd.DataFrame(rows + retrain_rows)
    df_summary = pd.concat([df_mlp, xgb_df, mlp13_df, mlp20_df, mlp21_df, mlp22_df], ignore_index=True)
    df_summary = df_summary.sort_values("pooled_r2", ascending=False).reset_index(drop=True)
    df_summary.to_csv(args.out / "metrics_summary.csv", index=False)
    print(f"[eval] wrote metrics_summary.csv ({len(df_summary)} rows)", flush=True)

    # ---- per-regime breakdown ----
    per_reg_rows = []
    for family in [f["id"] for f in config["families"]]:
        sub = sweep[(sweep["family"] == family) & sweep["architecture"].isin(HONEST_ARCHS)] \
            .sort_values("val_rmse", na_position="last").head(args.per_regime_n)
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
    df_per_reg = pd.concat([pd.DataFrame(per_reg_rows), xgb_per_reg, mlp13_per_reg, mlp20_per_reg, mlp21_per_reg, mlp22_per_reg], ignore_index=True)
    df_per_reg.to_csv(args.out / "per_regime_metrics_summary.csv", index=False)
    print(f"[eval] wrote per_regime_metrics_summary.csv ({len(df_per_reg)} rows)", flush=True)

    # ---- selected_features.json (eval-1.1 schema) ----
    selected = {
        "shared_backbone_54": list(config["shared_backbone_54"]),
        "candidate_pool_96": list(data.candidate_pool),
        "baseline_v0_50": list(data.v0_features),
        "cluster_1_delta_features": c1_deltas,
        "selection_protocol": "3-seed mean val RMSE among mlp/fg/plr (phases 1-2-3; aux2020 diagnostic only)",
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

    # champion-vs-sweep bar chart (5-seed champion ensembles when present)
    if retrain_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = [rr["family"] for rr in retrain_rows if "(5-seed champ)" in rr["config_id"]]
        if fams:
            singles = {}
            for _, r in sweep[sweep["family"].isin(set(fams))].sort_values("val_rmse").iterrows():
                singles.setdefault(r["family"], r["test_r2"])
            ens_r2 = {rr["family"]: rr["pooled_r2"] for rr in retrain_rows if "(5-seed champ)" in rr["config_id"]}
            fams = sorted(set(fams))
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(fams))
            w = 0.35
            single_vals = [singles.get(f, float("nan")) for f in fams]
            ens_vals = [ens_r2.get(f, float("nan")) for f in fams]
            ax.bar(x - w / 2, single_vals, w, label="Best single (val-selected)", color="#1f77b4")
            ax.bar(x + w / 2, ens_vals, w, label="5-seed champion ensemble", color="#2ca02c")
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
            ax.set_title("Val-selected single model vs 5-seed champion ensemble")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(args.out / "retrain_ensemble.png", dpi=150)
            plt.close()

    # ---- timing summary ----
    timing = json.loads((args.out / "timing_log.json").read_text(encoding="utf-8"))
    timing["eval_wall_s"] = time.perf_counter() - t0
    # Record the device(s) the training workers actually ran on (per-seed
    # metas of the CURRENT data_version), not the eval host — so a CPU re-run
    # of this script does not mislabel the H100-produced artifacts, and stale
    # smoke (data_version -1) leftovers are excluded.
    cur_version = int(config["sweep"].get("data_version", 4))
    devices: set[str] = set()
    for meta_p in (args.out / "models").glob("*/*/seed_*/meta.json"):
        try:
            payload = json.loads(meta_p.read_text(encoding="utf-8"))
            if int(payload.get("config", {}).get("data_version", -1)) != cur_version:
                continue
            dev = payload.get("device")
            if dev:
                devices.add(str(dev))
        except Exception:
            continue
    timing["gpu"] = {"device": ", ".join(sorted(devices)) or "unknown", "n_parallel": config["sweep"]["n_parallel"]}
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
    print("FINAL MODEL LEADERBOARD (derived_8.4-eval-mlp-2.3)", flush=True)
    print("=" * 72, flush=True)
    print(df_summary[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
