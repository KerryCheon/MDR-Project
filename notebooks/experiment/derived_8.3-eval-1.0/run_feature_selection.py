#!/usr/bin/env python3
"""Run c1 (V6) feature selection on K=2 regime / cluster subsets for derived_8.3-eval-1.0.

Uses the recommended pipeline for derived_8.3 from feature-selection-1.0:
  configs/config_c1_baseline_bypass_off.yaml
  (MI → ElasticNet → stability, bypass OFF, top_k=50, stability_n_boot=50)

Embeds the OVERALL_SELECTED_FEATURES_V0 list from data/splits/derived_8.3/dataset_metadata.py
as the global baseline feature list.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "configs" / "config_c1_baseline_bypass_off.yaml"
METADATA_PATH = PROJECT_ROOT / "data" / "splits" / "derived_8.3" / "dataset_metadata.py"
OUT_JSON = EXP_DIR / "selected_features.json"
ARTIFACTS_DIR = EXP_DIR / "artifacts" / "selection"

TARGET = "soil_moisture_5cm"
BINARY_THRESHOLD = 0.16
COLS_DYNAMIC = ["SMAP_sm_pm_interp_lag1", "G_API", "LST_modis"]
MIN_TRAIN_SAMPLES = 30


def load_v0_features() -> list[str]:
    spec = importlib.util.spec_from_file_location("dataset_metadata", METADATA_PATH)
    dm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dm)
    return list(dm.OVERALL_SELECTED_FEATURES_V0)


class QuantileBinner:
    def __init__(self, K: int):
        self.K = K
        self.thresholds: list[float] = []

    def fit(self, series: pd.Series):
        val = series.fillna(series.mean())
        self.thresholds = [float(val.quantile(i / self.K)) for i in range(1, self.K)]

    def predict(self, series: pd.Series) -> np.ndarray:
        val = series.fillna(series.mean())
        if self.K == 2:
            return np.where(val < self.thresholds[0], 0, 1)
        raise NotImplementedError("Only K=2 supported in eval-1.0")


class KMeansClusterer:
    def __init__(self, cols: list[str], K: int):
        self.cols = cols
        self.K = K
        self.means = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)

    def fit(self, df: pd.DataFrame):
        X = df[self.cols].copy()
        self.means = X.mean()
        X = X.fillna(self.means)
        self.kmeans.fit(self.scaler.fit_transform(X))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.cols].copy().fillna(self.means)
        return self.kmeans.predict(self.scaler.transform(X))


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["month"] = out["date"].dt.month.astype(int)
    out["year"] = out["date"].dt.year.astype(float)
    return out


def _seasonal_labels(df: pd.DataFrame) -> np.ndarray:
    cond = [
        df["month"].isin([5, 6, 7, 8, 9, 10]),
        df["month"].isin([11, 12, 1, 2, 3, 4]),
    ]
    return np.select(cond, [0, 1], default=0)


def _run_selection_on_subset(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
    y_te,
    cfg: dict,
    run_dir: Path,
    run_id: str,
    n_boot: int | None,
) -> list[str]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from Modeling.Src.soilmoist_fl.cli import select_features

    cfg_local = json.loads(json.dumps(cfg))
    cfg_local["models"] = []
    if n_boot is not None:
        cfg_local.setdefault("selection", {})["stability_n_boot"] = int(n_boot)
        for st in cfg_local["selection"].get("stages", []):
            if st.get("kind") == "stability":
                st["stability_n_boot"] = int(n_boot)

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    used_cfg = run_dir / "config_used.yaml"
    with open(used_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_local, f, sort_keys=False)

    try:
        res = select_features(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_va,
            y_val=y_va,
            X_test=X_te,
            y_test=y_te,
            config=cfg_local,
            run_dir=run_dir,
            run_id=run_id,
            verbose=True,
        )
        feats = list(res["selected_features"])
    except Exception as e:
        print(f"[ERROR] select_features failed for {run_id}: {type(e).__name__}: {e}")
        feats = []

    if len(feats) == 0:
        print(f"[WARNING] 0 features for {run_id}; retrying with min_freq=0.0")
        cfg_fb = json.loads(json.dumps(cfg_local))
        stages = []
        for st in cfg_fb.get("selection", {}).get("stages", []):
            if st.get("kind") == "stability":
                st = dict(st)
                st["min_freq"] = 0.0
            stages.append(st)
        cfg_fb.setdefault("selection", {})["stages"] = stages
        try:
            res = select_features(
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_va,
                y_val=y_va,
                X_test=X_te,
                y_test=y_te,
                config=cfg_fb,
                run_dir=run_dir,
                run_id=run_id + "_fallback",
                verbose=True,
            )
            feats = list(res["selected_features"])
        except Exception as e:
            print(f"[ERROR] fallback also failed for {run_id}: {type(e).__name__}: {e}")
            feats = []

    return feats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-boot",
        type=int,
        default=None,
        help="Override stability_n_boot (default: from c1 config, typically 50).",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=MIN_TRAIN_SAMPLES,
        help=f"Skip selection if train subset smaller than this (default {MIN_TRAIN_SAMPLES}).",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(PROJECT_ROOT))
    from Modeling.Src.soilmoist_fl.Features.preprocess import preprocess_split

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing c1 config: {CONFIG_PATH}")

    cfg = _load_config(CONFIG_PATH)
    global_v0_feats = load_v0_features()

    train = _add_temporal(pd.read_csv(PROJECT_ROOT / "data/splits/derived_8.3/train.csv"))
    val = _add_temporal(pd.read_csv(PROJECT_ROOT / "data/splits/derived_8.3/val.csv"))
    test = _add_temporal(pd.read_csv(PROJECT_ROOT / "data/splits/derived_8.3/test.csv"))

    drop_cols = list(cfg.get("data", {}).get("id_cols", ["station_id"])) + [
        cfg.get("data", {}).get("time_col", "date")
    ]
    for extra in ("month", "year"):
        if extra not in drop_cols:
            drop_cols.append(extra)

    X_tr, y_tr, _, _ = preprocess_split(train, TARGET, drop_cols=drop_cols)
    X_va, y_va, _, _ = preprocess_split(val, TARGET, drop_cols=drop_cols)
    X_te, y_te, _, _ = preprocess_split(test, TARGET, drop_cols=drop_cols)

    y_tr = pd.Series(np.asarray(y_tr).ravel(), index=X_tr.index)
    y_va = pd.Series(np.asarray(y_va).ravel(), index=X_va.index)
    y_te = pd.Series(np.asarray(y_te).ravel(), index=X_te.index)

    strategies: dict[str, dict] = {
        "binary_regime": {
            "kind": "binary_target",
            "K": 2,
            "labels": {
                "train": np.where(y_tr.values < BINARY_THRESHOLD, 0, 1),
                "val": np.where(y_va.values < BINARY_THRESHOLD, 0, 1),
                "test": np.where(y_te.values < BINARY_THRESHOLD, 0, 1),
            },
            "name_map": {0: "dry", 1: "wet"},
        },
        "Univariate_G_API_k2": {
            "kind": "quantile",
            "K": 2,
            "col": "G_API",
        },
        "Clustering_Dynamic_k2": {
            "kind": "kmeans",
            "K": 2,
            "cols": COLS_DYNAMIC,
        },
        "Seasonal_Binary_k2": {
            "kind": "seasonal",
            "K": 2,
        },
        "Clustering_V0_Full_k2": {
            "kind": "kmeans",
            "K": 2,
            "cols": global_v0_feats,
        },
    }

    binner = QuantileBinner(2)
    binner.fit(train["G_API"])
    strategies["Univariate_G_API_k2"]["labels"] = {
        "train": binner.predict(train.loc[X_tr.index, "G_API"]),
        "val": binner.predict(val.loc[X_va.index, "G_API"]),
        "test": binner.predict(test.loc[X_te.index, "G_API"]),
    }

    km_dyn = KMeansClusterer(COLS_DYNAMIC, 2)
    km_dyn.fit(train)
    strategies["Clustering_Dynamic_k2"]["labels"] = {
        "train": km_dyn.predict(train.loc[X_tr.index]),
        "val": km_dyn.predict(val.loc[X_va.index]),
        "test": km_dyn.predict(test.loc[X_te.index]),
    }

    strategies["Seasonal_Binary_k2"]["labels"] = {
        "train": _seasonal_labels(train.loc[X_tr.index]),
        "val": _seasonal_labels(val.loc[X_va.index]),
        "test": _seasonal_labels(test.loc[X_te.index]),
    }

    km_v0 = KMeansClusterer(global_v0_feats, 2)
    km_v0.fit(train)
    strategies["Clustering_V0_Full_k2"]["labels"] = {
        "train": km_v0.predict(train.loc[X_tr.index]),
        "val": km_v0.predict(val.loc[X_va.index]),
        "test": km_v0.predict(test.loc[X_te.index]),
    }

    binary_regime: dict = {}
    clusters: dict = {}
    selection_meta: dict = {}

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def _checkpoint():
        payload = _build_payload(
            global_v0_feats,
            binary_regime,
            clusters,
            selection_meta,
            partial=True,
        )
        OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for strat_name, strat in strategies.items():
        K = strat["K"]
        labels_tr = np.asarray(strat["labels"]["train"]).ravel()
        labels_va = np.asarray(strat["labels"]["val"]).ravel()
        labels_te = np.asarray(strat["labels"]["test"]).ravel()
        name_map = strat.get("name_map")

        print(f"\n{'=' * 60}")
        print(f"Strategy: {strat_name} (K={K})")
        print(f"{'=' * 60}")

        strat_out: dict[str, list[str]] = {}
        for c in range(K):
            key = name_map[c] if name_map else str(c)
            mask_tr = labels_tr == c
            mask_va = labels_va == c
            mask_te = labels_te == c
            n_tr = int(mask_tr.sum())
            n_va = int(mask_va.sum())
            n_te = int(mask_te.sum())
            print(f"\n--- {strat_name} / {key}: train={n_tr} val={n_va} test={n_te} ---")

            meta_key = f"{strat_name}/{key}"
            if n_tr < args.min_train:
                print(
                    f"[WARNING] train n={n_tr} < {args.min_train}; "
                    f"using global v0 as fallback for {meta_key}"
                )
                feats = list(global_v0_feats)
                selection_meta[meta_key] = {
                    "n_train": n_tr,
                    "n_val": n_va,
                    "n_test": n_te,
                    "status": "fallback_global_v0_small_n",
                    "n_features": len(feats),
                }
            else:
                run_dir = ARTIFACTS_DIR / strat_name / f"c{c}" / "run"
                run_id = f"eval83_{strat_name}_c{c}"
                feats = _run_selection_on_subset(
                    X_tr=X_tr.loc[mask_tr],
                    y_tr=y_tr.loc[mask_tr],
                    X_va=X_va.loc[mask_va] if n_va > 0 else None,
                    y_va=y_va.loc[mask_va] if n_va > 0 else None,
                    X_te=X_te.loc[mask_te] if n_te > 0 else None,
                    y_te=y_te.loc[mask_te] if n_te > 0 else None,
                    cfg=cfg,
                    run_dir=run_dir,
                    run_id=run_id,
                    n_boot=args.n_boot,
                )
                status = "ok"
                if len(feats) == 0:
                    print(f"[WARNING] empty selection for {meta_key}; fallback global v0")
                    feats = list(global_v0_feats)
                    status = "fallback_global_v0_empty"
                selection_meta[meta_key] = {
                    "n_train": n_tr,
                    "n_val": n_va,
                    "n_test": n_te,
                    "status": status,
                    "n_features": len(feats),
                    "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                }

            strat_out[key] = feats
            print(f"Selected {len(feats)} features for {meta_key}: {feats[:5]}...")

        if name_map:
            binary_regime = {
                k: {"n": len(v), "features": v} for k, v in strat_out.items()
            }
        else:
            clusters[strat_name] = {
                k: {"n": len(v), "features": v} for k, v in strat_out.items()
            }
        _checkpoint()

    payload = _build_payload(
        global_v0_feats,
        binary_regime,
        clusters,
        selection_meta,
        partial=False,
    )
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")


def _build_payload(
    global_v0_feats,
    binary_regime,
    clusters,
    selection_meta,
    partial: bool,
) -> dict:
    return {
        "version": "V6-c1-derived_8.3",
        "partial": partial,
        "pipeline": "c1_baseline_bypass_off",
        "dataset": "derived_8.3",
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "binary_regime_threshold": BINARY_THRESHOLD,
        "global_c1": {
            "n": len(global_v0_feats),
            "features": list(global_v0_feats),
            "source": "data/splits/derived_8.3/dataset_metadata.py::OVERALL_SELECTED_FEATURES_V0",
        },
        "global_v0": {
            "n": len(global_v0_feats),
            "features": list(global_v0_feats),
            "source": "data/splits/derived_8.3/dataset_metadata.py::OVERALL_SELECTED_FEATURES_V0",
        },
        "binary_regime": binary_regime,
        "clusters": clusters,
        "selection_meta": selection_meta,
    }


if __name__ == "__main__":
    main()
