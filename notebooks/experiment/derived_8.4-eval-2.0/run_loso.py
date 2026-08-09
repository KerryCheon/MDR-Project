#!/usr/bin/env python3
"""Main execution script for derived_8.4-eval-2.0 — MLP LOSO spatial generalization.

Continuation of derived_8.4-eval-1.2 (same output format) but evaluating the
MLP models established in derived_8.4-eval-mlp-1.3 (2-regime) and
derived_8.4-eval-mlp-1.1 (1-regime) under leave-one-station-out across the 7 WA
stations of the derived_8.4 split. Scope: 1-regime (global single MLP) + the
best clustering strategy only (Clustering_V0_Full_k2, c0=0, c1=10).

Per (config, held-out station): the router is refitted on the fold's trainval
(6 remaining stations) and the MLP specialists are trained per regime cluster
on the fold's train split, early-stopped on the fold's val split (mlp-1.3
trainer, patience 60), then evaluated on all test rows of the held-out station.
Metrics (pooled / per-year / per-regime), per-fold predictions (.npy) and model
checkpoints are persisted; the XGBoost LOSO references from eval-1.2 are merged
into the leaderboard for the direct MLP-vs-XGBoost spatial comparison.

Usage:
    python run_loso.py                          # full run (6 configs x 7 stations)
    python run_loso.py --max-configs 1 --max-stations 1 --smoke   # smoke test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
CONFIG_PATH = EXP_DIR / "config.yaml"
MODELS_DIR = EXP_DIR / "models"
PREDICTIONS_DIR = EXP_DIR / "predictions"
sys.path.insert(0, str(EXP_DIR))

from eval20 import data as edata  # noqa: E402
from eval20.evaluator import compute_metrics  # noqa: E402
from eval20.references import (  # noqa: E402
    load_eval12_loso_reference_folds,
    load_eval12_loso_reference_summary,
    load_mlp_reference_map,
)
from eval20.plots import (  # noqa: E402
    plot_config_station_heatmap,
    plot_config_summary_bar,
    plot_station_boxplot,
    plot_station_difficulty,
)

FAMILY_ORDER = ["1regime_54", "1regime_96", "2regime_54", "2regime_96"]
FAMILY_LABELS = {
    "1regime_54": "MLP 1-Regime-54",
    "1regime_96": "MLP 1-Regime-96",
    "2regime_54": "MLP 2-Regime-54",
    "2regime_96": "MLP 2-Regime-96",
}
# Val-selected winners per family (honest selection from mlp-1.1 / mlp-1.3).
WINNER_CONFIGS = {
    "1regime_54": "w256x256_d0.3_tanh",
    "1regime_96": "res_w512x512_d0.2_wd1e-3",
    "2regime_54": "w512x512x512_d0.3_huber0.1",
    "2regime_96": "w512x512x512_d0.3_lr1e-3",
}
XGB_REF_LABELS = {
    "Global_Single_54": "XGBoost Global Single (54 Backbone)",
    "Clustering_V0_Full_k2_c0_0_c1_10": "XGBoost Clustering_V0_Full_k2 (Winner c0=0, c1=10)",
}

SMOKE = False
SMOKE_VERSION = -1


def build_configurations(data, config) -> list[dict]:
    """Pin the LOSO configurations (provenance / audit trail), eval-1.2-style."""
    ref_map = load_mlp_reference_map(config)
    configs: list[dict] = []
    for (family, cid) in edata.all_loso_configs(config):
        fam_cfg = next(f for f in config["families"] if f["id"] == family)
        feats = edata.family_features(family, config, data)
        configs.append({
            "config_id": f"{family}__{cid}",
            "config_id_short": cid,
            "family": family,
            "strategy_name": f"MLP_{family}",
            "config_label": f"{FAMILY_LABELS[family]} ({cid})",
            "structure": fam_cfg["structure"],
            "global_features": list(feats[""] if "" in feats else feats["_cluster0"]),
            "cluster_additions": {
                "0": [],
                "1": list(feats.get("_cluster1", [])[len(feats["_cluster0"]):]),
            } if fam_cfg["structure"] == "cluster" else {"0": [], "1": []},
            "is_baseline": False,
            "is_winner": WINNER_CONFIGS.get(family) == cid,
            "cluster_0_count": 0 if fam_cfg["structure"] == "cluster" else None,
            "cluster_1_count": int(config["cluster_config"]["c1_count"]) if fam_cfg["structure"] == "cluster" else None,
            "mlp13_test_r2": ref_map.get((family, cid), float("nan")),
        })
    return configs


def build_config_frame(configurations: list[dict]) -> pd.DataFrame:
    rows = []
    for cfg in configurations:
        rows.append({
            "config_id": cfg["config_id"],
            "config_id_short": cfg["config_id_short"],
            "config_label": cfg["config_label"],
            "strategy_name": cfg["strategy_name"],
            "strategy_order": FAMILY_ORDER.index(cfg["family"]),
            "family": cfg["family"],
            "is_baseline": cfg["is_baseline"],
            "is_winner": cfg["is_winner"],
            "cluster_0_count": cfg["cluster_0_count"],
            "cluster_1_count": cfg["cluster_1_count"],
            "mlp13_test_r2": cfg["mlp13_test_r2"],
            "n_global_features": len(cfg["global_features"]),
        })
    return pd.DataFrame(rows)


def compute_pooled_loso_metrics(df_pcs: pd.DataFrame, config, predictions_dir: Path) -> dict[str, dict[str, float]]:
    """Pooled LOSO R2/RMSE per config (concatenated 7-station folds, eval-1.2-style)."""
    if predictions_dir is None or not predictions_dir.exists():
        return {}
    data_cfg = config["data"]
    test = pd.read_csv(PROJECT_ROOT / Path(data_cfg["splits"]["test"]))
    test = test.sort_values("station_id", kind="mergesort").reset_index(drop=True)
    target = str(data_cfg["target"])
    y_true = test[target].to_numpy(dtype=float)

    results: dict[str, dict[str, float]] = {}
    for config_id in df_pcs["config_id"].unique():
        stations = sorted(df_pcs.loc[df_pcs["config_id"] == config_id, "station"].unique())
        if len(stations) < 7:
            continue  # partial run (smoke/resume): pooled LOSO needs all 7 folds
        try:
            preds = np.concatenate([
                np.load(predictions_dir / f"{config_id}__{s}_preds.npy") for s in stations
            ])
        except FileNotFoundError:
            continue
        if len(preds) != len(y_true):
            print(f"[warn] pooled length mismatch for {config_id}: {len(preds)} != {len(y_true)} — skipping",
                  flush=True)
            continue
        m = compute_metrics(y_true, preds)
        results[config_id] = {"loso_pooled_r2": m["r2"], "loso_pooled_rmse": m["rmse"]}
    return results


# ---------------------------------------------------------------------------
# Job management (mirrors run_mlp_sweep.py; workers = run_loso_worker.py)
# ---------------------------------------------------------------------------


def expected_version(config: dict) -> int:
    return SMOKE_VERSION if SMOKE else int(config["sweep"].get("data_version", 6))


def job_dir(out: Path, family: str, cid: str, station: str, seed: int) -> Path:
    return out / "models" / family / cid / station / f"seed_{seed}"


def seed_complete(jdir: Path, version: int) -> bool:
    meta = jdir / "meta.json"
    if not meta.exists():
        return False
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("status") == "completed" and payload.get("config", {}).get("data_version") == version


def _ckpt_version_ok(jdir: Path, version: int) -> bool:
    import torch

    for ck in [jdir / "checkpoint.pt", jdir / "spec_0" / "checkpoint.pt", jdir / "spec_1" / "checkpoint.pt"]:
        if ck.exists():
            try:
                ckpt = torch.load(ck, map_location="cpu", weights_only=False)
                return ckpt.get("config", {}).get("data_version") == version
            except Exception:
                return False
    return True


def seed_resumable(jdir: Path, version: int) -> bool:
    has_ckpt = (jdir / "checkpoint.pt").exists() or (jdir / "spec_0" / "checkpoint.pt").exists()
    return has_ckpt and _ckpt_version_ok(jdir, version)


def _clean_job_dir(jdir: Path) -> None:
    import shutil

    if jdir.exists():
        shutil.rmtree(jdir)


def make_jobs(out: Path, scope, seeds: list[int], config: dict) -> list[tuple[str, str, str, int, str]]:
    """Jobs = (family, cid, station, seed, mode) for the LOSO scope."""
    version = expected_version(config)
    jobs: list[tuple[str, str, str, int, str]] = []
    for family, cid, station in scope:
        for seed in seeds:
            jdir = job_dir(out, family, cid, station, seed)
            if seed_complete(jdir, version):
                mode = "skip"
            elif seed_resumable(jdir, version):
                mode = "resume"
            else:
                stale = jdir.exists() and (
                    (jdir / "meta.json").exists() or (jdir / "checkpoint.pt").exists()
                    or (jdir / "spec_0" / "checkpoint.pt").exists()
                )
                if stale:
                    print(f"[invalidate] {family}/{cid}/{station}/seed_{seed}: stale artifacts", flush=True)
                    _clean_job_dir(jdir)
                mode = "fresh"
            jobs.append((family, cid, station, seed, mode))
    return jobs


def run_jobs(out: Path, jobs, n_parallel: int, config: dict) -> None:
    artifacts = out / "artifacts"
    queue = list(jobs)
    active: list[tuple[subprocess.Popen, tuple, object, float]] = []
    while queue or active:
        while len(active) < n_parallel and queue:
            family, cid, station, seed, mode = queue.pop(0)
            log_path = artifacts / "logs" / f"{family}__{cid}__{station}__s{seed}.log"
            cmd = [
                sys.executable, str(EXP_DIR / "run_loso_worker.py"),
                "--family", family, "--config-id", cid, "--station", station,
                "--seed", str(seed), "--artifacts", str(artifacts), "--out", str(out),
            ]
            logf = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(EXP_DIR))
            active.append((proc, (family, cid, station, seed, mode), logf, time.perf_counter()))
            print(f"[launch] {family}/{cid}/{station}/seed_{seed} ({mode})", flush=True)

        still_active = []
        for proc, job, logf, started in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((proc, job, logf, started))
                continue
            logf.close()
            family, cid, station, seed, mode = job
            ok = seed_complete(job_dir(out, family, cid, station, seed), expected_version(config))
            status = "ok" if (ok and rc == 0) else f"FAILED(rc={rc})"
            print(f"[finish] {family}/{cid}/{station}/seed_{seed} {status} wall={time.perf_counter() - started:.1f}s",
                  flush=True)
            if not ok and rc != 0:
                print(f"         log: {artifacts / 'logs' / f'{family}__{cid}__{station}__s{seed}.log'}", flush=True)
        active = still_active
        if active:
            time.sleep(2.0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_fold(out: Path, family: str, cid: str, station: str, seeds: list[int],
                   config: dict) -> dict | None:
    """Seed-mean a (config, station) fold -> predictions file + fold metrics row."""
    version = expected_version(config)
    metas = {}
    for seed in seeds:
        jdir = job_dir(out, family, cid, station, seed)
        meta_path = jdir / "meta.json"
        if not meta_path.exists():
            continue
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        if m.get("status") == "completed" and m.get("config", {}).get("data_version") == version:
            metas[seed] = m
    if not metas:
        return None

    preds_list = [np.load(job_dir(out, family, cid, station, s) / "preds.npy") for s in sorted(metas)]
    mean_preds = np.mean(preds_list, axis=0)

    config_id = f"{family}__{cid}"
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PREDICTIONS_DIR / f"{config_id}__{station}_preds.npy", mean_preds)

    # Fold test targets / years / cluster labels.
    suffix = f"__{station}"
    ft = np.load(out / "artifacts" / f"fold_{family}{suffix}_test.npz", allow_pickle=True)
    y_test, years = ft["y_test"], ft["year"]
    labels_te = np.load(out / "artifacts" / f"labels_{family}{suffix}.npz", allow_pickle=True)["test"]
    np.save(PREDICTIONS_DIR / f"{config_id}__{station}_labels_te.npy", labels_te)

    pooled = compute_metrics(y_test, mean_preds)
    first = next(iter(metas.values()))
    row = {
        "config_id": config_id,
        "station": station,
        "strategy_name": f"MLP_{family}",
        "n_train_total": int(first.get("n_train_total", 0)),
        "n_test": int(len(y_test)),
        "r2": pooled["r2"],
        "rmse": pooled["rmse"],
        "ubrmse": pooled["ubrmse"],
        "bias": pooled["bias"],
        "mae": pooled["mae"],
        "pearson": pooled["pearson"],
        "train_time_s": float(np.mean([m.get("train_time_s", 0.0) for m in metas.values()])),
        "val_rmse": float(np.mean([m.get("val_rmse", float("nan")) for m in metas.values()])),
        "aux_rmse": float(np.mean([m.get("aux_rmse", float("nan")) for m in metas.values()])),
        "best_epoch": int(max(m.get("best_epoch", 0) for m in metas.values())),
        "epochs": int(max(m.get("epochs", 0) for m in metas.values())),
        "n_seeds": len(metas),
    }
    return row


def cluster_rows_for_fold(out: Path, family: str, cid: str, station: str) -> list[dict]:
    """Per-cluster metrics on the seed-mean fold predictions (eval-1.2 format)."""
    config_id = f"{family}__{cid}"
    pred_file = PREDICTIONS_DIR / f"{config_id}__{station}_preds.npy"
    if not pred_file.exists():
        return []
    mean_preds = np.load(pred_file)
    suffix = f"__{station}"
    ft = np.load(out / "artifacts" / f"fold_{family}{suffix}_test.npz", allow_pickle=True)
    y_test = ft["y_test"]
    labels_te = np.load(out / "artifacts" / f"labels_{family}{suffix}.npz", allow_pickle=True)["test"]

    rows = []
    for cl in sorted(set(labels_te.tolist())):
        mask = labels_te == int(cl)
        n_train = _cluster_n_train(out, family, cid, station, cl)
        rows.append({
            "config_id": config_id,
            "station": station,
            "strategy_name": f"MLP_{family}",
            "cluster": int(cl),
            "n_train": float(n_train),
            "n_test": float(mask.sum()),
            **compute_metrics(y_test[mask], mean_preds[mask]),
        })
    return rows


def _cluster_n_train(out: Path, family: str, cid: str, station: str, cl: int) -> int:
    """Sum n_train over completed seeds for a cluster (0 if the cluster was empty)."""
    jdir_root = out / "models" / family / cid / station
    total = 0
    for seed_dir in sorted(jdir_root.glob("seed_*")):
        meta_path = seed_dir / "meta.json"
        if not meta_path.exists():
            continue
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        pc = m.get("per_cluster", {}).get(str(cl))
        if pc:
            total += int(pc.get("n_train", 0))
    # mean over seeds
    n_seeds = len(list(jdir_root.glob("seed_*"))) or 1
    return max(0, total // n_seeds)


def yearly_rows_for_fold(out: Path, family: str, cid: str, station: str) -> list[dict]:
    config_id = f"{family}__{cid}"
    pred_file = PREDICTIONS_DIR / f"{config_id}__{station}_preds.npy"
    if not pred_file.exists():
        return []
    mean_preds = np.load(pred_file)
    suffix = f"__{station}"
    ft = np.load(out / "artifacts" / f"fold_{family}{suffix}_test.npz", allow_pickle=True)
    y_test, years = ft["y_test"], ft["year"]
    rows = []
    for year in sorted(set(int(y) for y in years.tolist())):
        mask = years == year
        rows.append({
            "config_id": config_id,
            "station": station,
            "strategy_name": f"MLP_{family}",
            "year": int(year),
            **compute_metrics(y_test[mask], mean_preds[mask]),
        })
    return rows


def merge_xgb_reference_folds(df_pcs: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach eval-1.2 XGBoost LOSO folds as reference rows (no retraining)."""
    ref = load_eval12_loso_reference_folds(config)
    if ref.empty:
        return df_pcs
    ref = ref.copy()
    ref["strategy_name"] = "XGBoost_Reference"
    ref["config_label"] = ref["config_id"].map(XGB_REF_LABELS)
    ref["strategy_order"] = 4
    ref["family"] = "xgboost"
    ref["is_baseline"] = True
    ref["is_winner"] = False
    ref["mlp13_test_r2"] = float("nan")
    keep = ["config_id", "config_label", "strategy_name", "strategy_order",
            "family", "is_baseline", "is_winner", "cluster_0_count", "cluster_1_count",
            "mlp13_test_r2", "eval11_test_r2", "station", "n_train_total", "n_test",
            "r2", "rmse", "ubrmse", "bias", "mae", "pearson", "train_time_s",
            "val_rmse", "aux_rmse", "best_epoch", "epochs", "n_seeds"]
    for col in keep:
        if col not in ref.columns:
            ref[col] = float("nan")
    df_pcs = pd.concat([df_pcs, ref[keep]], ignore_index=True)
    return df_pcs


def merge_xgb_reference_regime_year(df_regime: pd.DataFrame, df_year: pd.DataFrame,
                                    config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach eval-1.2 XGBoost reference rows to the per-regime / per-year CSVs."""
    src = PROJECT_ROOT / Path(config["eval12_reference_dir"])
    out_r, out_y = df_regime, df_year
    rpath = src / "loso_per_regime_metrics.csv"
    if rpath.exists() and not df_regime.empty:
        ref = pd.read_csv(rpath)
        ref = ref[ref["config_id"].isin(XGB_REF_LABELS)].copy()
        if not ref.empty:
            ref["strategy_name"] = "XGBoost_Reference"
            out_r = pd.concat([df_regime, ref], ignore_index=True)
    ypath = src / "loso_per_year_metrics.csv"
    if ypath.exists() and not df_year.empty:
        ref = pd.read_csv(ypath)
        ref = ref[ref["config_id"].isin(XGB_REF_LABELS)].copy()
        if not ref.empty:
            ref["strategy_name"] = "XGBoost_Reference"
            out_y = pd.concat([df_year, ref], ignore_index=True)
    return out_r, out_y


def _write_partial(per_config_station, per_regime_rows, per_year_rows) -> None:
    pd.DataFrame(per_config_station).to_csv(EXP_DIR / "loso_per_config_station.csv", index=False)
    pd.DataFrame(per_regime_rows).to_csv(EXP_DIR / "loso_per_regime_metrics.csv", index=False)
    pd.DataFrame(per_year_rows).to_csv(EXP_DIR / "loso_per_year_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-configs", type=int, default=None, help="Limit number of configs (smoke test).")
    parser.add_argument("--max-stations", type=int, default=None, help="Limit number of stations (smoke test).")
    parser.add_argument("--config-id", action="append", default=None, help="Only run these config ids (repeatable).")
    parser.add_argument("--station", action="append", default=None, help="Only run these stations (repeatable).")
    parser.add_argument("--n-parallel", type=int, default=None, help="Concurrent GPU workers.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing partial results.")
    parser.add_argument("--smoke", action="store_true", help="Cap all jobs at 3 epochs (data_version -1).")
    args = parser.parse_args()

    global SMOKE
    SMOKE = args.smoke

    print("=" * 70, flush=True)
    print("Starting derived_8.4-eval-2.0 MLP LOSO Spatial Generalization Evaluation", flush=True)
    print("=" * 70, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = edata.load_experiment_data(PROJECT_ROOT, config)
    print(f"[Data] Train={len(data.train)} Val={len(data.val)} Test={len(data.test)} "
          f"TrainVal={len(data.trainval)}.", flush=True)
    print(f"[Backbone] {len(data.shared_backbone_54)} shared features, "
          f"{len(data.v0_features)} V0 features, {len(data.candidate_pool)} candidate pool.", flush=True)

    all_configurations = build_configurations(data, config)
    stations = sorted(data.test["station_id"].unique())
    print(f"[Configs] Pinned {len(all_configurations)} MLP configurations.", flush=True)
    print(f"[Stations] {len(stations)} held-out stations: {stations}", flush=True)

    # The metadata frame always covers ALL pinned configs (so folds resumed from
    # previous runs of other configs keep their labels); the run filter below
    # only decides which configs get (re)trained.
    if args.config_id:
        configurations = [c for c in all_configurations if c["config_id"] in args.config_id
                          or c["config_id_short"] in args.config_id]
    else:
        configurations = list(all_configurations)
    if args.max_configs:
        configurations = configurations[: args.max_configs]
    if args.station:
        stations = [s for s in stations if s in args.station]
    if args.max_stations:
        stations = stations[: args.max_stations]
    print(f"[Run] Evaluating {len(configurations)} configs x {len(stations)} stations.", flush=True)

    # Pin the exact configurations (provenance / audit trail).
    with open(EXP_DIR / "loso_configurations.json", "w", encoding="utf-8") as f:
        json.dump(all_configurations, f, indent=2)
    cfg_frame = build_config_frame(all_configurations)
    cfg_frame.to_csv(EXP_DIR / "loso_configs.csv", index=False)
    print(f"[Artifacts] Pinned configurations to loso_configurations.json", flush=True)

    # Prebuild per-fold tensors (shared by every config/seed of the family).
    artifacts = EXP_DIR / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "logs").mkdir(parents=True, exist_ok=True)
    edata.build_all_fold_tensors(data, config, artifacts, stations)

    sweep_configs = edata.build_sweep_configs(config)
    if args.smoke:
        for cid, cfg in sweep_configs.items():
            cfg["max_epochs"] = 3
            cfg["patience"] = 2
            cfg["checkpoint_every"] = 1
            cfg["data_version"] = SMOKE_VERSION
    with open(artifacts / "sweep_configs.json", "w", encoding="utf-8") as f:
        json.dump(sweep_configs, f, indent=2)
    print(f"[Artifacts] Wrote sweep_configs.json ({len(sweep_configs)} configs)", flush=True)

    n_parallel = args.n_parallel or int(config["sweep"]["n_parallel"])

    # Scope: (family, cid, station) triples; seeds are per-family (mlp-1.3
    # 2-regime refs are 2-seed, mlp-1.1 1-regime refs are 1-seed).
    scope = [
        (c["family"], c["config_id_short"], station)
        for c in configurations for station in stations
    ]
    jobs: list[tuple[str, str, str, int, str]] = []
    for family in {c["family"] for c in configurations}:
        fam_seeds = edata.seeds_for_family(config, family)
        fam_scope = [(f, cid, s) for (f, cid, s) in scope if f == family]
        jobs += make_jobs(EXP_DIR, fam_scope, fam_seeds, config)
    todo = [j for j in jobs if j[4] != "skip"]
    print(f"[Jobs] {len(jobs)} total ({len([j for j in jobs if j[4]=='skip'])} done, "
          f"{len([j for j in jobs if j[4]=='resume'])} resume, "
          f"{len([j for j in jobs if j[4]=='fresh'])} fresh)", flush=True)
    run_jobs(EXP_DIR, todo, n_parallel, config)

    # ---- Aggregate folds into eval-1.2-format CSVs ----
    done: set[tuple[str, str]] = set()
    per_config_station: list[dict] = []
    per_regime_rows: list[dict] = []
    per_year_rows: list[dict] = []
    version = expected_version(config)
    if not args.no_resume and (EXP_DIR / "loso_per_config_station.csv").exists():
        prev = pd.read_csv(EXP_DIR / "loso_per_config_station.csv")
        # Only trust folds whose seed jobs carry the CURRENT data_version
        # (stale artifacts from --smoke / older data_versions are re-run).
        def _fold_valid(config_id: str, station: str) -> bool:
            if "__" not in config_id:
                return False
            family, cid = config_id.split("__", 1)
            seeds = edata.seeds_for_family(config, family)
            return all(
                seed_complete(job_dir(EXP_DIR, family, cid, station, s), version)
                for s in seeds
            )

        mask = [_fold_valid(cid, st) for cid, st in zip(prev["config_id"], prev["station"])]
        done = set(zip(prev["config_id"], prev["station"])) & set(
            (cid, st) for (cid, st), ok in zip(zip(prev["config_id"], prev["station"]), mask) if ok
        )
        kept = [r for r, ok in zip(prev.to_dict(orient="records"), mask) if ok]
        per_config_station = kept
        if (EXP_DIR / "loso_per_regime_metrics.csv").exists():
            rreg = pd.read_csv(EXP_DIR / "loso_per_regime_metrics.csv")
            per_regime_rows = rreg[rreg.apply(
                lambda r: (r["config_id"], r["station"]) in done, axis=1
            )].to_dict(orient="records")
        if (EXP_DIR / "loso_per_year_metrics.csv").exists():
            ryr = pd.read_csv(EXP_DIR / "loso_per_year_metrics.csv")
            per_year_rows = ryr[ryr.apply(
                lambda r: (r["config_id"], r["station"]) in done, axis=1
            )].to_dict(orient="records")
        print(f"[Resume] {len(done)} valid folds kept ({len(prev) - len(done)} stale dropped).", flush=True)

    for c in configurations:
        family, cid = c["family"], c["config_id_short"]
        config_id = c["config_id"]
        fam_seeds = edata.seeds_for_family(config, family)
        for station in stations:
            if (config_id, station) in done:
                continue
            row = aggregate_fold(EXP_DIR, family, cid, station, fam_seeds, config)
            if row is None:
                continue
            row.update({k: v for k, v in c.items() if k in (
                "config_label", "strategy_name", "strategy_order", "family", "is_baseline",
                "is_winner", "cluster_0_count", "cluster_1_count", "mlp13_test_r2",
            )})
            per_config_station.append(row)
            per_regime_rows.extend(cluster_rows_for_fold(EXP_DIR, family, cid, station))
            per_year_rows.extend(yearly_rows_for_fold(EXP_DIR, family, cid, station))
            _write_partial(per_config_station, per_regime_rows, per_year_rows)
        print(f"[Config] Finished {config_id} ({len(stations)} stations).", flush=True)

    df_pcs = pd.DataFrame(per_config_station)
    df_regime = pd.DataFrame(per_regime_rows)
    df_year = pd.DataFrame(per_year_rows)

    # Merge configuration metadata for plotting / summary (drop stale overlap cols).
    right_cols = cfg_frame.columns
    overlap = [c for c in right_cols if c in df_pcs.columns and c not in ("config_id", "strategy_name")]
    if overlap:
        df_pcs = df_pcs.drop(columns=overlap)
    df_pcs = df_pcs.merge(cfg_frame, on=["config_id", "strategy_name"], how="left")

    # XGBoost LOSO references (eval-1.2) as leaderboard/figure rows.
    df_pcs = merge_xgb_reference_folds(df_pcs, config)
    df_regime, df_year = merge_xgb_reference_regime_year(df_regime, df_year, config)

    # ---- Per-configuration summary ----
    summary = df_pcs.groupby("config_id").agg(
        n_stations=("station", "count"),
        total_test_n=("n_test", "sum"),
        loso_mean_r2=("r2", "mean"),
        loso_median_r2=("r2", "median"),
        loso_std_r2=("r2", "std"),
        loso_min_r2=("r2", "min"),
        loso_max_r2=("r2", "max"),
        loso_mean_rmse=("rmse", "mean"),
        loso_mean_ubrmse=("ubrmse", "mean"),
        loso_mean_bias=("bias", "mean"),
        loso_mean_mae=("mae", "mean"),
    ).reset_index()
    meta_cols = ["config_label", "strategy_name", "strategy_order", "family", "is_baseline",
                 "is_winner", "cluster_0_count", "cluster_1_count", "mlp13_test_r2",
                 "eval11_test_r2"]
    # Metadata source = df_pcs itself (MLP folds + XGBoost reference rows), so
    # the summary keeps the right labels for both.
    meta_lookup = df_pcs.drop_duplicates("config_id")[["config_id", *meta_cols]]
    summary = summary.merge(meta_lookup, on="config_id", how="left")
    for col in meta_cols:
        if col not in summary.columns:
            summary[col] = float("nan")
    best = df_pcs.loc[df_pcs.groupby("config_id")["r2"].idxmax(), ["config_id", "station"]]
    worst = df_pcs.loc[df_pcs.groupby("config_id")["r2"].idxmin(), ["config_id", "station"]]
    summary = summary.merge(best.rename(columns={"station": "best_station"}), on="config_id", how="left")
    summary = summary.merge(worst.rename(columns={"station": "worst_station"}), on="config_id", how="left")
    # Temporal reference for the LOSO gap: MLP rows use mlp-1.3/1.1 test R2;
    # XGBoost reference rows use eval-1.1 test R2 (via eval-1.2's summary).
    summary["ref_test_r2"] = summary["mlp13_test_r2"].fillna(summary["eval11_test_r2"])
    summary["loso_minus_test_r2"] = summary["loso_mean_r2"] - summary["ref_test_r2"]

    pooled = compute_pooled_loso_metrics(df_pcs, config, PREDICTIONS_DIR)
    if pooled:
        summary = summary.merge(
            pd.DataFrame(pooled).T.reset_index().rename(columns={"index": "config_id"}),
            on="config_id", how="left",
        )
        # XGBoost reference pooled LOSO backfilled from eval-1.2 (no retraining).
        ref_summary = load_eval12_loso_reference_summary(config)
        if not ref_summary.empty:
            backfill = ref_summary.set_index("config_id")["loso_pooled_r2"].to_dict()
            summary["loso_pooled_r2"] = summary["loso_pooled_r2"].fillna(
                summary["config_id"].map(backfill)
            )
        summary["pooled_loso_minus_test_r2"] = summary["loso_pooled_r2"] - summary["ref_test_r2"]

    summary = summary.sort_values("loso_mean_r2", ascending=False).reset_index(drop=True)
    summary.to_csv(EXP_DIR / "loso_config_summary.csv", index=False)

    # ---- Per-station summary (station difficulty byproduct) ----
    station_summary = df_pcs.groupby("station").agg(
        n_configs=("config_id", "count"),
        total_test_n=("n_test", "sum"),
        mean_r2=("r2", "mean"),
        median_r2=("r2", "median"),
        std_r2=("r2", "std"),
        min_r2=("r2", "min"),
        max_r2=("r2", "max"),
        mean_rmse=("rmse", "mean"),
        mean_mae=("mae", "mean"),
        mean_bias=("bias", "mean"),
    ).reset_index()
    neg = (
        df_pcs.groupby("station")["r2"]
        .apply(lambda r: int((r < 0).sum()))
        .rename("n_negative_r2")
        .reset_index()
    )
    station_summary = station_summary.merge(neg, on="station", how="left")
    station_summary = station_summary.sort_values("median_r2", ascending=False).reset_index(drop=True)
    station_summary.to_csv(EXP_DIR / "loso_station_summary.csv", index=False)

    df_pcs.to_csv(EXP_DIR / "loso_per_config_station.csv", index=False)
    df_regime.to_csv(EXP_DIR / "loso_per_regime_metrics.csv", index=False)
    df_year.to_csv(EXP_DIR / "loso_per_year_metrics.csv", index=False)
    print(f"[Artifacts] Wrote loso_per_config_station.csv / loso_config_summary.csv / "
          f"loso_station_summary.csv / loso_per_regime_metrics.csv / loso_per_year_metrics.csv", flush=True)

    if not args.skip_plots:
        plot_config_station_heatmap(df_pcs, EXP_DIR)
        plot_config_summary_bar(summary, EXP_DIR)
        plot_station_difficulty(station_summary, EXP_DIR)
        plot_station_boxplot(df_pcs, EXP_DIR)
        print(f"[Plots] Generated LOSO figures in {EXP_DIR}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("LOSO CONFIGURATION LEADERBOARD (derived_8.4-eval-2.0)", flush=True)
    print("=" * 70, flush=True)
    cols = ["config_id", "strategy_name", "loso_mean_r2", "loso_std_r2", "loso_min_r2",
            "loso_max_r2", "loso_mean_rmse", "loso_mean_bias", "mlp13_test_r2",
            "loso_minus_test_r2", "is_winner"]
    if "loso_pooled_r2" in summary.columns:
        cols = ["config_id", "strategy_name", "loso_mean_r2", "loso_pooled_r2", "loso_std_r2",
                "loso_min_r2", "loso_max_r2", "loso_mean_rmse", "loso_mean_bias",
                "mlp13_test_r2", "loso_minus_test_r2", "is_winner"]
    print(summary[cols].to_string(index=False), flush=True)

    print("\nSTATION DIFFICULTY (median LOSO R2 across configs)", flush=True)
    print(station_summary[["station", "median_r2", "mean_r2", "min_r2", "max_r2", "n_negative_r2"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
