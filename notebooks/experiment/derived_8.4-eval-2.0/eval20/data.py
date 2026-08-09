"""Fold-aware data preparation for derived_8.4-eval-2.0 (MLP LOSO).

Protocol (data_version 6): train on the official train split (2017-2020),
early-stop on the official val split (2021-2022), test on the test split
(2023-2025). For LOSO, every fold removes ONE station:

  fold_train = train rows with station != s   (2017-2020, 6 stations)
  fold_val   = val rows   with station != s   (2021-2022, 6 stations)
  fold_test  = all test rows of station s     (2023-2025)

The router (GlobalSingle for 1-regime, V0Full KMeans k=2 for 2-regime) is
refitted per fold on fold_trainval (train+val of the 6 remaining stations) —
no held-out-station leakage into routing. Specialists are preprocessed with the
mlp-1.3 pipeline (median impute + StandardScaler fit on the fold's train rows,
clip [-5, 5]; aux2020 = 2020 slice of fold_train, diagnostic only).

Feature sets are persisted per (family, station) under ``artifacts/`` so every
(config, station, seed) job trains from the same tensors (mirrors how
run_mlp_sweep.py prebuilds per-family tensors). A cluster whose fold-train mask
is empty is NOT persisted; the worker then falls back to the fold-train-mean
prediction (same fallback as eval-1.2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parents[2]
EVAL11_DIR = EXP_DIR.parent / "derived_8.4-eval-1.1"
EVAL12_DIR = EXP_DIR.parent / "derived_8.4-eval-1.2"
MLP13_DIR = EXP_DIR.parent / "derived_8.4-eval-mlp-1.3"
MLP11_DIR = EXP_DIR.parent / "derived_8.4-eval-mlp-1.1"
for _d in (EVAL11_DIR, EVAL12_DIR, MLP13_DIR, MLP11_DIR):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))

from eval11.data import load_experiment_data  # noqa: E402
from eval12.routers import get_router  # noqa: E402
from mlp13.data import build_feature_set, load_feature_set, save_feature_set  # noqa: E402

CLUSTERS = ("0", "1")


def load_cluster_deltas(config: dict) -> list[str]:
    """Cluster-1 delta additions: mlp-1.3's selected_features.json, else config."""
    sel_path = MLP13_DIR / "selected_features.json"
    if sel_path.exists():
        meta = json.loads(sel_path.read_text(encoding="utf-8"))
        for rec in meta.get("leaderboard", []):
            if rec.get("candidate_id") == "Clustering_V0_Full_k2_c0_0_c1_10":
                add = rec.get("cluster_1_additions", "")
                return [f for f in add.split(";") if f]
    return list(config["cluster_config"]["cluster_1_delta_features"])


def family_features(family_id: str, config: dict, data) -> dict[str, list[str]]:
    """{suffix: [feature cols]} for a family ('' = global; '_cluster0/1' = specialists)."""
    fam_cfg = next(f for f in config["families"] if f["id"] == family_id)
    if fam_cfg["features"] == "candidate_pool_96":
        base = list(data.candidate_pool)
    else:
        base = list(config["shared_backbone_54"])
    if fam_cfg["structure"] == "global":
        return {"": base}
    c1_deltas = load_cluster_deltas(config)
    c1 = [f for f in dict.fromkeys([*base, *c1_deltas]) if f in set(data.feature_columns)]
    return {"_cluster0": base, "_cluster1": c1}


def _fold_split(data, station: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(fold_train, fold_val, fold_test). station=None -> full (all 7 stations)."""
    if station is None:
        return data.train, data.val, data.test
    return (
        data.train[data.train["station_id"] != station].reset_index(drop=True),
        data.val[data.val["station_id"] != station].reset_index(drop=True),
        data.test[data.test["station_id"] == station].reset_index(drop=True),
    )


def _labels(router, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
            aux: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    out = {
        "train": np.asarray(router.predict(train)).ravel().astype(int),
        "val": np.asarray(router.predict(val)).ravel().astype(int),
        "test": np.asarray(router.predict(test)).ravel().astype(int),
    }
    if aux is not None and len(aux) > 0:
        out["aux"] = np.asarray(router.predict(aux)).ravel().astype(int)
    return out


def build_one_family_tensors(
    data,
    config: dict,
    family_id: str,
    artifacts: Path,
    *,
    station: str | None,
) -> None:
    """Build + persist the feature sets for one family on one fold (or full).

    station=None trains on ALL 7 stations (full-training baseline / temporal
    replication); station=<id> leaves that station out (LOSO fold).
    """
    fam_cfg = next(f for f in config["families"] if f["id"] == family_id)
    target = data.target
    aux_year = int(config["sweep"].get("aux_year", 2020))
    suffix = "__full" if station is None else f"__{station}"

    fold_train, fold_val, fold_test = _fold_split(data, station)
    fold_trainval = pd.concat([fold_train, fold_val], axis=0, ignore_index=True)
    aux_frame = fold_train[fold_train["year"] == aux_year].reset_index(drop=True)

    strategy = "Global_Single" if fam_cfg["structure"] == "global" else str(config["cluster_config"]["strategy"])
    router = get_router(strategy, data.v0_features, seed=int(config["model"]["seed"]))
    router.fit(fold_trainval)
    lab = _labels(router, fold_train, fold_val, fold_test, aux=aux_frame)

    featsets = family_features(family_id, config, data)
    for feats_suffix, feats in featsets.items():
        if fam_cfg["structure"] == "global":
            fs = build_feature_set(
                fold_train, fold_val, fold_test, feats, target,
                aux=aux_frame,
            )
            save_feature_set(artifacts / f"tensors_{family_id}{suffix}.npz", fs)
        else:
            cl = feats_suffix.replace("_cluster", "")
            tr_mask = lab["train"] == int(cl)
            va_mask = lab["val"] == int(cl)
            te_mask = lab["test"] == int(cl)
            if not tr_mask.any():
                # Empty fold-train cluster -> worker falls back to train mean.
                continue
            fs = build_feature_set(
                fold_train.loc[tr_mask].reset_index(drop=True),
                fold_val.loc[va_mask].reset_index(drop=True),
                fold_test.loc[te_mask].reset_index(drop=True),
                feats, target,
                test_positions=np.where(te_mask)[0],
                aux=aux_frame.loc[lab["aux"] == int(cl)].reset_index(drop=True)
                if "aux" in lab else None,
            )
            save_feature_set(artifacts / f"tensors_{family_id}{suffix}{feats_suffix}.npz", fs)

    # Labels (needed by the worker for per-regime masks / n_train / fallback).
    np.savez_compressed(
        artifacts / f"labels_{family_id}{suffix}.npz",
        train=lab["train"], val=lab["val"], test=lab["test"],
    )
    # Fold meta (worker uses n_test + train_target_mean for the mean fallback).
    with open(artifacts / f"fold_{family_id}{suffix}_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "station": station,
            "n_train_total": int(len(fold_train)),
            "n_val_total": int(len(fold_val)),
            "n_test": int(len(fold_test)),
            "train_target_mean": float(fold_train[target].mean()),
        }, f, indent=2)
    # Fold test targets / years / station ids (worker metric assembly).
    np.savez_compressed(
        artifacts / f"fold_{family_id}{suffix}_test.npz",
        y_test=fold_test[target].to_numpy(dtype=np.float64),
        year=fold_test["year"].to_numpy(dtype=np.int64),
        station=np.asarray(fold_test["station_id"], dtype=object),
    )
    print(f"[tensors] {family_id}{suffix}: train {len(fold_train)} val {len(fold_val)} "
          f"test {len(fold_test)} (router {strategy})", flush=True)


def build_all_fold_tensors(data, config: dict, artifacts: Path, stations: list[str]) -> None:
    """Persist per-fold feature sets for every family x station (LOSO)."""
    artifacts.mkdir(parents=True, exist_ok=True)
    for family in [f["id"] for f in config["families"]]:
        for station in stations:
            build_one_family_tensors(data, config, family, artifacts, station=station)
    np.savez_compressed(
        artifacts / "test_meta.npz",
        y_test=data.test[data.target].to_numpy(dtype=np.float64),
        year=data.test["year"].to_numpy(dtype=np.int64),
        station=np.asarray(data.test["station_id"], dtype=object),
    )


def build_all_full_tensors(data, config: dict, artifacts: Path) -> None:
    """Persist per-family feature sets trained on ALL 7 stations (full baseline)."""
    artifacts.mkdir(parents=True, exist_ok=True)
    for family in [f["id"] for f in config["families"]]:
        build_one_family_tensors(data, config, family, artifacts, station=None)


def load_fold_tensors(artifacts: Path, family: str, station: str) -> dict[str, dict]:
    """Load a fold's persisted feature sets ('' for global, or per cluster).

    A global family persists a single ``tensors_<family>__<station>.npz``; a
    cluster family persists per-cluster ``..._cluster{0,1}.npz`` files (a
    cluster with no fold-train rows is absent -> worker falls back to the mean).
    """
    suffix = f"__{station}"
    out: dict[str, dict] = {}
    global_path = artifacts / f"tensors_{family}{suffix}.npz"
    if global_path.exists():
        out[""] = load_feature_set(global_path)
        return out
    for cl in CLUSTERS:
        p = artifacts / f"tensors_{family}{suffix}_cluster{cl}.npz"
        if p.exists():
            out[cl] = load_feature_set(p)
    return out


def load_fold_labels(artifacts: Path, family: str, station: str) -> dict[str, np.ndarray]:
    suffix = f"__{station}"
    data = np.load(artifacts / f"labels_{family}{suffix}.npz", allow_pickle=True)
    return {k: data[k] for k in ("train", "val", "test")}


def build_sweep_configs(config: dict) -> dict[str, dict]:
    """config_id -> cfg dict (defaults + overrides), mirroring run_mlp_sweep."""
    defaults = dict(config["sweep"]["defaults"])
    defaults["seed"] = int(config["model"]["seed"])
    defaults["data_version"] = int(config["sweep"].get("data_version", 6))
    out: dict[str, dict] = {}
    for entry in config["sweep"]["configs"]:
        cfg = dict(defaults)
        cfg.update({k: v for k, v in entry.items() if k != "id"})
        cfg["id"] = entry["id"]
        if "hidden_sizes" in entry:
            cfg["hidden_sizes"] = [int(h) for h in entry["hidden_sizes"]]
        out[entry["id"]] = cfg
    return out


def family_config_ids(config: dict, family_id: str) -> list[str]:
    """Config ids to run in a family (honors sweep.family_configs filter)."""
    all_ids = [c["id"] for c in config["sweep"]["configs"]]
    fc = config["sweep"].get("family_configs", {})
    return list(fc.get(family_id, all_ids)) if family_id in fc else all_ids


def all_loso_configs(config: dict) -> list[tuple[str, str]]:
    """[(family, config_id)] for the whole LOSO scope (family_configs union)."""
    return [
        (family, cid)
        for family in [f["id"] for f in config["families"]]
        for cid in family_config_ids(config, family)
    ]


def seeds_for_family(config: dict, family: str) -> list[int]:
    """Seeds for a family (per-family override, else the default sweep seeds)."""
    fam_seeds = config["sweep"].get("family_seeds", {})
    if family in fam_seeds:
        return [int(s) for s in fam_seeds[family]]
    return [int(s) for s in config["sweep"].get("seeds", [42])]
