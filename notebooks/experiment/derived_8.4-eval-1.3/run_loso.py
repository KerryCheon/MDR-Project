#!/usr/bin/env python3
"""Main execution script for derived_8.4-eval-1.3 LOSO spatial generalization evaluation.

Runs leave-one-station-out evaluation for all 56 configurations (the 47 pinned
from derived_8.4-eval-1.1 + the NEW Clustering_Backbone54_k2 strategy's 9
delta-grid points) across the 7 WA stations of the derived_8.4 split, using the
derived_8.4-eval-2.0 parallel worker format: the driver pins the configurations,
writes artifacts/runtime.json, and spawns n_parallel `run_loso_worker.py`
subprocesses — each trains one (config, station) fold on the GPU concurrently —
then aggregates the per-fold job meta.json files into the eval-1.2-format CSVs
(pooled / per-year / per-regime), predictions (.npy) and model weights (.json).

Resume: completed folds are detected via artifacts/jobs/<config_id>__<station>/
meta.json (status + data_version match) and skipped; interrupted folds are
re-run fresh (XGBoost folds are atomic). `--smoke` uses data_version=-1 and a
reduced n_estimators so smoke artifacts are never reused by the real run.

Usage:
    python run_loso.py                                   # full run (56 configs x 7 stations)
    python run_loso.py --max-configs 1 --max-stations 1 --skip-plots   # smoke test
    python run_loso.py --smoke --max-configs 1 --max-stations 1 --device cpu  # CPU smoke
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

from eval13.data import load_experiment_data
from eval13.evaluator import LosoEvaluator, compute_metrics
from eval13.plots import (
    plot_config_station_heatmap,
    plot_config_summary_bar,
    plot_station_boxplot,
    plot_station_difficulty,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.yaml"
MODELS_DIR = EXP_DIR / "models"
PREDICTIONS_DIR = EXP_DIR / "predictions"
ARTIFACTS_DIR = EXP_DIR / "artifacts"

STRATEGY_ORDER = [
    "Global_Single",
    "Clustering_V0_Full_k2",
    "Clustering_Backbone54_k2",  # NEW: KMeans k=2 on the 54 shared-backbone features
    "Clustering_Dynamic_k2",
    "Univariate_G_API_k2",
    "Seasonal_Binary_k2",
    "Trained_Gating_k2",
]
MOE_STRATEGIES = STRATEGY_ORDER[1:]

SMOKE = False


def parse_additions(value: object) -> list[str]:
    """Parse semicolon-joined cluster additions from eval-1.1 CSV cells."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [f for f in text.split(";") if f]


def load_configurations(data, config) -> list[dict]:
    """Build the 56 configurations: the 47 pinned from eval-1.1 + the NEW
    Clustering_Backbone54_k2 9-point delta grid (additions reused from eval-1.1's
    Clustering_V0_Full_k2 rows at the same (c0, c1) points, per config.yaml)."""
    src = PROJECT_ROOT / Path(config["loso"]["source_eval11_dir"])
    grid = pd.read_csv(src / "delta_grid_summary.csv")
    leaderboard = pd.read_csv(src / "metrics_summary.csv")

    # eval-1.1 test-set R2 by candidate_id (for LOSO vs temporal comparison).
    eval11_r2: dict[str, float] = {}
    for _, row in grid.iterrows():
        eval11_r2[str(row["candidate_id"])] = float(row["pooled_r2"])
    for _, row in leaderboard.iterrows():
        eval11_r2[str(row["candidate_id"])] = float(row["pooled_r2"])

    # eval-1.1 winning (c0, c1) per strategy, e.g. "Clustering_V0_Full_k2 (Winner c0=0, c1=10)".
    winner_by_strategy: dict[str, tuple[int, int]] = {}
    for _, row in leaderboard.iterrows():
        mname = str(row["model_name"])
        if "Winner" not in mname:
            continue
        strategy = str(row["strategy_name"])
        try:
            c0 = int(mname.split("c0=")[1].split(",")[0])
            c1 = int(mname.split("c1=")[1].split(")")[0])
        except (IndexError, ValueError):
            continue
        winner_by_strategy[strategy] = (c0, c1)

    configs: list[dict] = []

    def add_config(config_id, strategy, global_features, additions, c0=None, c1=None,
                   eval11_candidate_id=None, is_baseline=False, is_winner=None):
        if is_winner is None:
            is_winner = (
                not is_baseline and c0 is not None and c1 is not None
                and winner_by_strategy.get(strategy) == (c0, c1)
            )
        configs.append({
            "config_id": config_id,
            "strategy_name": strategy,
            "global_features": list(global_features),
            "cluster_additions": {str(k): list(v) for k, v in additions.items()},
            "is_baseline": bool(is_baseline),
            "cluster_0_count": c0,
            "cluster_1_count": c1,
            "eval11_candidate_id": eval11_candidate_id,
            "eval11_test_r2": eval11_r2.get(eval11_candidate_id, float("nan")),
            "is_winner": bool(is_winner),
        })

    # 1. Baseline V0 (50 Feats) — Global_Single with V0 features.
    add_config(
        config_id="Baseline_V0_50",
        strategy="Global_Single",
        global_features=data.v0_features,
        additions={},
        eval11_candidate_id="Baseline_V0_50",
        is_baseline=True,
    )
    # 2. Global Single Model (54 Backbone).
    add_config(
        config_id="Global_Single_54",
        strategy="Global_Single",
        global_features=data.shared_backbone_54,
        additions={},
        eval11_candidate_id="Global_Single_backbone_0_0",
        is_baseline=True,
    )
    # 3. MoE strategies x 9 delta-grid points (fixed additions from eval-1.1).
    for strat in MOE_STRATEGIES:
        sub = grid[grid["strategy_name"] == strat].sort_values(
            ["cluster_0_count", "cluster_1_count"]
        )
        for _, row in sub.iterrows():
            c0 = int(row["cluster_0_count"])
            c1 = int(row["cluster_1_count"])
            add_config(
                config_id=f"{strat}_c0_{c0}_c1_{c1}",
                strategy=strat,
                global_features=data.shared_backbone_54,
                additions={
                    "0": parse_additions(row["cluster_0_additions"]),
                    "1": parse_additions(row["cluster_1_additions"]),
                },
                c0=c0,
                c1=c1,
                eval11_candidate_id=str(row["candidate_id"]),
            )
    # 4. NEW strategy Clustering_Backbone54_k2: same 54 backbone features as the
    #    single-regime global model, 9 delta-grid points. Per-(c0, c1) additions
    #    are pinned from eval-1.1's Clustering_V0_Full_k2 rows (the additions stay
    #    fixed from eval-1.1, exactly like the other MoE strategies — only the
    #    routing features differ). Winner pinned in config.yaml (mirrors the
    #    eval-1.1 V0-Full winner c0=0, c1=10), decided before running.
    new_cfg = config["new_clustering_strategy"]
    new_strat = str(new_cfg["strategy"])
    winner_pt = (int(new_cfg["winner"]["c0"]), int(new_cfg["winner"]["c1"]))
    additions_src = str(new_cfg.get("additions_source", "Clustering_V0_Full_k2"))
    sub = grid[grid["strategy_name"] == additions_src].sort_values(
        ["cluster_0_count", "cluster_1_count"]
    )
    for _, row in sub.iterrows():
        c0 = int(row["cluster_0_count"])
        c1 = int(row["cluster_1_count"])
        add_config(
            config_id=f"{new_strat}_c0_{c0}_c1_{c1}",
            strategy=new_strat,
            global_features=data.shared_backbone_54,
            additions={
                "0": parse_additions(row["cluster_0_additions"]),
                "1": parse_additions(row["cluster_1_additions"]),
            },
            c0=c0,
            c1=c1,
            eval11_candidate_id=None,
            is_winner=(c0, c1) == winner_pt,
        )
    return configs


def build_config_frame(configurations: list[dict]) -> pd.DataFrame:
    """DataFrame describing each configuration (used for plots and summary)."""
    rows = []
    for cfg in configurations:
        label = cfg["config_id"] if cfg["is_baseline"] else (
            f"{cfg['strategy_name']}  c0={cfg['cluster_0_count']}, c1={cfg['cluster_1_count']}"
        )
        rows.append({
            "config_id": cfg["config_id"],
            "config_label": label,
            "strategy_name": cfg["strategy_name"],
            "strategy_order": STRATEGY_ORDER.index(cfg["strategy_name"]),
            "is_baseline": cfg["is_baseline"],
            "is_winner": cfg["is_winner"],
            "cluster_0_count": cfg["cluster_0_count"],
            "cluster_1_count": cfg["cluster_1_count"],
            "eval11_test_r2": cfg["eval11_test_r2"],
            "n_global_features": len(cfg["global_features"]),
            "n_add0": len(cfg["cluster_additions"].get("0", [])),
            "n_add1": len(cfg["cluster_additions"].get("1", [])),
        })
    return pd.DataFrame(rows)


def compute_pooled_loso_metrics(df_pcs: pd.DataFrame, config, predictions_dir: Path) -> dict[str, dict[str, float]]:
    """Pooled LOSO metrics per config.

    Concatenates per-fold predictions across the 7 held-out stations (every test sample is
    predicted by a model trained without its own station) and computes sample-count-weighted
    R2/RMSE — directly comparable to eval-1.1's pooled test-set R2.
    """
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
            raise ValueError(f"Pooled prediction length mismatch for {config_id}: {len(preds)} != {len(y_true)}")
        m = compute_metrics(y_true, preds)
        results[config_id] = {"loso_pooled_r2": m["r2"], "loso_pooled_rmse": m["rmse"]}
    return results


# ---------------------------------------------------------------------------
# Job management (derived_8.4-eval-2.0 parallel worker format; workers =
# run_loso_worker.py, one (config, station) XGBoost fold per job)
# ---------------------------------------------------------------------------


def expected_version(config: dict) -> int:
    return -1 if SMOKE else int(config["loso"].get("data_version", 1))


def runtime_path() -> Path:
    return ARTIFACTS_DIR / "runtime.json"


def job_dir(config_id: str, station: str) -> Path:
    return ARTIFACTS_DIR / "jobs" / f"{config_id}__{station}"


def fold_preds_file(config_id: str, station: str) -> Path:
    if station == "full":
        return EXP_DIR / "predictions_full" / f"{config_id}__full_preds.npy"
    return PREDICTIONS_DIR / f"{config_id}__{station}_preds.npy"


def load_job_meta(config_id: str, station: str) -> dict | None:
    meta_path = job_dir(config_id, station) / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fold_complete(config_id: str, station: str, version: int) -> bool:
    meta = load_job_meta(config_id, station)
    if meta is None:
        return False
    if meta.get("status") != "completed" or meta.get("data_version") != version:
        return False
    return fold_preds_file(config_id, station).exists()


def make_jobs(config_ids: list[str], stations: list[str], config: dict) -> list[tuple[str, str, str]]:
    """Jobs = (config_id, station, mode) with mode in {skip, fresh}."""
    version = expected_version(config)
    jobs: list[tuple[str, str, str]] = []
    for config_id in config_ids:
        for station in stations:
            if fold_complete(config_id, station, version):
                mode = "skip"
            else:
                mode = "fresh"
            jobs.append((config_id, station, mode))
    return jobs


def run_jobs(jobs: list[tuple[str, str, str]], n_parallel: int, config: dict) -> None:
    """Spawn up to n_parallel worker subprocesses; each trains one fold."""
    (ARTIFACTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    queue = list(jobs)
    active: list[tuple[subprocess.Popen, tuple, object, float]] = []
    while queue or active:
        while len(active) < n_parallel and queue:
            config_id, station, mode = queue.pop(0)
            log_path = ARTIFACTS_DIR / "logs" / f"{config_id}__{station}.log"
            cmd = [
                sys.executable, str(EXP_DIR / "run_loso_worker.py"),
                "--config-id", config_id, "--station", station,
                "--artifacts", str(ARTIFACTS_DIR), "--out", str(EXP_DIR),
            ]
            logf = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(EXP_DIR))
            active.append((proc, (config_id, station, mode), logf, time.perf_counter()))
            print(f"[launch] {config_id} @ {station} ({mode})", flush=True)

        still_active = []
        for proc, job, logf, started in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((proc, job, logf, started))
                continue
            logf.close()
            config_id, station, mode = job
            ok = fold_complete(config_id, station, expected_version(config))
            status = "ok" if (ok and rc == 0) else f"FAILED(rc={rc})"
            print(f"[finish] {config_id} @ {station} {status} wall={time.perf_counter() - started:.1f}s",
                  flush=True)
            if not ok and rc != 0:
                print(f"         log: {ARTIFACTS_DIR / 'logs' / f'{config_id}__{station}.log'}", flush=True)
        active = still_active
        if active:
            time.sleep(2.0)


def write_runtime(config: dict, device: str | None, n_estimators: int | None = None) -> None:
    """Per-run worker overrides: data_version (resume invalidation), device, n_estimators."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_version": expected_version(config),
        "device": device or str(config["model"]["exact_params"].get("device", "cuda")),
    }
    if n_estimators is not None:
        payload["n_estimators"] = int(n_estimators)
    with open(runtime_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Aggregation (reads per-fold worker meta.json; eval-1.2-format CSVs)
# ---------------------------------------------------------------------------


def aggregate_fold(config_id: str, station: str, config: dict) -> dict | None:
    """One (config, station) fold -> per-config-station row (None if incomplete)."""
    meta = load_job_meta(config_id, station)
    if meta is None or meta.get("status") != "completed":
        return None
    if meta.get("data_version") != expected_version(config):
        return None
    pooled = meta.get("pooled", {})
    return {
        "config_id": config_id,
        "station": station,
        "strategy_name": meta.get("strategy_name", ""),
        "n_train_total": int(meta.get("n_train_total", 0)),
        "n_test": int(meta.get("n_test", 0)),
        "r2": pooled.get("r2", float("nan")),
        "rmse": pooled.get("rmse", float("nan")),
        "ubrmse": pooled.get("ubrmse", float("nan")),
        "bias": pooled.get("bias", float("nan")),
        "mae": pooled.get("mae", float("nan")),
        "pearson": pooled.get("pearson", float("nan")),
        "train_time_s": float(meta.get("train_time_s", 0.0)),
    }


def cluster_rows_for_fold(config_id: str, station: str, config: dict) -> list[dict]:
    meta = load_job_meta(config_id, station)
    if meta is None or meta.get("status") != "completed":
        return []
    rows = []
    for cl, m in (meta.get("per_cluster") or {}).items():
        rows.append({
            "config_id": config_id,
            "station": station,
            "strategy_name": meta.get("strategy_name", ""),
            "cluster": int(cl),
            **{k: m.get(k) for k in ("n_train", "n_test", "r2", "rmse", "ubrmse", "bias", "mae", "pearson")},
        })
    return rows


def yearly_rows_for_fold(config_id: str, station: str, config: dict) -> list[dict]:
    meta = load_job_meta(config_id, station)
    if meta is None or meta.get("status") != "completed":
        return []
    rows = []
    for year, m in (meta.get("yearly") or {}).items():
        rows.append({
            "config_id": config_id,
            "station": station,
            "strategy_name": meta.get("strategy_name", ""),
            "year": int(year),
            **{k: m.get(k) for k in ("r2", "rmse", "ubrmse", "bias", "mae", "pearson")},
        })
    return rows


# ---------------------------------------------------------------------------
# eval-1.2 reference rows (deterministic same-protocol results)
# ---------------------------------------------------------------------------
# The 47 eval-1.1 configs were already evaluated under the IDENTICAL LOSO
# protocol in derived_8.4-eval-1.2 (same seed 42, same xgboost 3.2.0 env — its
# full baseline replicated eval-1.1 to 0.000000). Rather than re-computing them
# (~2 h of GPU time, no new information), eval-1.3 merges those recorded rows
# as references for any pinned config NOT computed in this run (all-or-nothing
# per config) and computes only the NEW Clustering_Backbone54_k2 configs — the
# same reference pattern derived_8.4-eval-2.0 used for its XGBoost baselines.


def load_eval12_loso_references(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """eval-1.2 LOSO rows: (per-config-station, per-regime, per-year, pooled R2 map)."""
    src = PROJECT_ROOT / Path(config["loso"].get(
        "reference_eval12_dir", "notebooks/experiment/derived_8.4-eval-1.2"))
    pcs_path = src / "loso_per_config_station.csv"
    if not pcs_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    pcs = pd.read_csv(pcs_path)
    regime = pd.read_csv(src / "loso_per_regime_metrics.csv")
    year = pd.read_csv(src / "loso_per_year_metrics.csv")
    summary = pd.read_csv(src / "loso_config_summary.csv")
    pooled_r2 = dict(zip(summary["config_id"], summary.get("loso_pooled_r2", float("nan"))))
    pooled_rmse = dict(zip(summary["config_id"], summary.get("loso_pooled_rmse", float("nan"))))
    pooled = {k: {"loso_pooled_r2": pooled_r2.get(k, float("nan")),
                  "loso_pooled_rmse": pooled_rmse.get(k, float("nan"))}
              for k in pooled_r2}
    return pcs, regime, year, pooled


def merge_loso_references(df_pcs: pd.DataFrame, df_regime: pd.DataFrame, df_year: pd.DataFrame,
                          configurations: list[dict], config: dict
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    """Append eval-1.2 reference rows for pinned configs with no computed folds.

    Returns (df_pcs, df_regime, df_year, referenced_config_ids, eval12_pooled_map).
    All-or-nothing per config: a config is referenced only if NONE of its folds
    were computed in this run (computed folds win).
    """
    computed_ids = set(df_pcs["config_id"].unique()) if not df_pcs.empty else set()
    pinned_ids = {c["config_id"] for c in configurations}
    need_ref = sorted(pinned_ids - computed_ids)
    if not need_ref:
        return df_pcs, df_regime, df_year, set(), {}
    ref_pcs, ref_regime, ref_year, pooled = load_eval12_loso_references(config)
    if ref_pcs.empty:
        print(f"[Refs] eval-1.2 reference file not found — {len(need_ref)} configs missing "
              f"(computed-only aggregation).", flush=True)
        return df_pcs, df_regime, df_year, set(), {}

    mask = ref_pcs["config_id"].isin(need_ref)
    ref_pcs = ref_pcs[mask].copy()
    ref_pcs["is_reference"] = True
    ref_regime = ref_regime[ref_regime["config_id"].isin(need_ref)].copy()
    ref_regime["is_reference"] = True
    ref_year = ref_year[ref_year["config_id"].isin(need_ref)].copy()
    ref_year["is_reference"] = True

    df_pcs = pd.concat([df_pcs, ref_pcs], ignore_index=True)
    df_regime = pd.concat([df_regime, ref_regime], ignore_index=True)
    df_year = pd.concat([df_year, ref_year], ignore_index=True)
    print(f"[Refs] Merged eval-1.2 LOSO rows for {len(need_ref)} configurations "
          f"({len(ref_pcs)} folds, {len(ref_regime)} regime rows, {len(ref_year)} year rows).", flush=True)
    return df_pcs, df_regime, df_year, set(need_ref), pooled


def _write_partial(per_config_station, per_regime_rows, per_year_rows) -> None:
    """Crash-safety: flush accumulated rows to CSVs."""
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
    parser.add_argument("--device", default=None, help="Override model device (e.g. cpu for login-node smoke).")
    parser.add_argument("--new-strategy-only", action="store_true",
                        help="Compute only the NEW Clustering_Backbone54_k2 configs; merge the "
                             "other 47 configs' LOSO results as eval-1.2 references (deterministic "
                             "same-protocol results — fits the 1h GPU wall).")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing partial results.")
    parser.add_argument("--smoke", action="store_true", help="data_version -1 + n_estimators 100 (never reused).")
    args = parser.parse_args()

    global SMOKE
    SMOKE = args.smoke

    print("=" * 70, flush=True)
    print("Starting derived_8.4-eval-1.3 LOSO Spatial Generalization Evaluation", flush=True)
    print("=" * 70, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = load_experiment_data(PROJECT_ROOT, config)
    print(f"[Data] TrainVal={len(data.trainval)} samples, Test={len(data.test)} samples.", flush=True)
    print(f"[Backbone] Shared global backbone: {len(data.shared_backbone_54)} features.", flush=True)
    print(f"[V0] Baseline V0: {len(data.v0_features)} features.", flush=True)

    all_configurations = load_configurations(data, config)
    stations = sorted(data.test["station_id"].unique())
    print(f"[Configs] Pinned {len(all_configurations)} configurations "
          f"({len([c for c in all_configurations if c['is_winner']])} winners).", flush=True)
    print(f"[Stations] {len(stations)} held-out stations: {stations}", flush=True)

    # Pin ALL configurations (provenance / audit trail) before any run filter.
    pin_path = EXP_DIR / "loso_configurations.json"
    with open(pin_path, "w", encoding="utf-8") as f:
        json.dump(all_configurations, f, indent=2)
    cfg_frame = build_config_frame(all_configurations)
    cfg_frame.to_csv(EXP_DIR / "loso_configs.csv", index=False)
    print(f"[Artifacts] Pinned configurations to {pin_path.name}", flush=True)

    # Run filter (execution scope only — the pin always covers all 56).
    if args.new_strategy_only:
        new_strat = str(config["new_clustering_strategy"]["strategy"])
        configurations = [c for c in all_configurations if c["strategy_name"] == new_strat]
        print(f"[Scope] Computing only the NEW strategy ({new_strat}, "
              f"{len(configurations)} configs); other pinned configs merge as eval-1.2 references.", flush=True)
    elif args.config_id:
        configurations = [c for c in all_configurations if c["config_id"] in args.config_id]
    else:
        configurations = list(all_configurations)
    if args.max_configs:
        configurations = configurations[: args.max_configs]
    if args.station:
        stations = [s for s in stations if s in args.station]
    if args.max_stations:
        stations = stations[: args.max_stations]
    print(f"[Run] Evaluating {len(configurations)} configs x {len(stations)} stations.", flush=True)

    n_parallel = args.n_parallel or int(config["loso"].get("n_parallel", 8))
    write_runtime(config, device=args.device, n_estimators=100 if args.smoke else None)

    jobs = make_jobs([c["config_id"] for c in configurations], stations, config)
    todo = [j for j in jobs if j[2] != "skip"]
    print(f"[Jobs] {len(jobs)} total ({len(jobs) - len(todo)} done, {len(todo)} to run) "
          f"with {n_parallel} parallel workers.", flush=True)
    run_jobs(todo, n_parallel, config)

    # ---- Aggregate folds into eval-1.2-format CSVs ----
    done: set[tuple[str, str]] = set()
    per_config_station: list[dict] = []
    per_regime_rows: list[dict] = []
    per_year_rows: list[dict] = []
    version = expected_version(config)
    if not args.no_resume and (EXP_DIR / "loso_per_config_station.csv").exists():
        prev = pd.read_csv(EXP_DIR / "loso_per_config_station.csv")
        mask = [
            fold_complete(cid, st, version)
            for cid, st in zip(prev["config_id"], prev["station"])
        ]
        done = set(zip(prev["config_id"], prev["station"])) & set(
            (cid, st) for (cid, st), ok in zip(zip(prev["config_id"], prev["station"]), mask) if ok
        )
        per_config_station = [r for r, ok in zip(prev.to_dict(orient="records"), mask) if ok]
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

    total = len(configurations) * len(stations)
    missing = 0
    for cfg in configurations:
        config_id = cfg["config_id"]
        for station in stations:
            if (config_id, station) in done:
                continue
            row = aggregate_fold(config_id, station, config)
            if row is None:
                missing += 1
                continue
            row["is_reference"] = False
            row.update({k: v for k, v in cfg.items() if k in (
                "config_label", "strategy_name", "strategy_order", "is_baseline", "is_winner",
                "cluster_0_count", "cluster_1_count", "eval11_test_r2",
            )})
            per_config_station.append(row)
            per_regime_rows.extend(cluster_rows_for_fold(config_id, station, config))
            per_year_rows.extend(yearly_rows_for_fold(config_id, station, config))
            if len(per_config_station) % 20 == 0:
                _write_partial(per_config_station, per_regime_rows, per_year_rows)
        _write_partial(per_config_station, per_regime_rows, per_year_rows)
        print(f"[Config] Finished {config_id} ({len(stations)} stations).", flush=True)
    if missing:
        print(f"[Warn] {missing} folds missing/incomplete — aggregation is partial.", flush=True)

    df_pcs = pd.DataFrame(per_config_station)
    df_regime = pd.DataFrame(per_regime_rows)
    df_year = pd.DataFrame(per_year_rows)

    # Merge eval-1.2 references for pinned configs not computed in this run
    # (all pinned configs, not just the run scope).
    df_pcs, df_regime, df_year, ref_ids, eval12_pooled = merge_loso_references(
        df_pcs, df_regime, df_year, all_configurations, config)
    if df_pcs.empty:
        print("[Error] No folds aggregated — check artifacts/logs/*.log for worker failures.", flush=True)
        raise SystemExit(1)

    # Merge configuration metadata for plotting / summary. When resuming, previously
    # written CSVs already carry these columns, so drop them before re-merging.
    right_cols = cfg_frame.drop(columns=["n_global_features", "n_add0", "n_add1"]).columns
    overlap = [c for c in right_cols if c in df_pcs.columns and c not in ("config_id", "strategy_name")]
    if overlap:
        df_pcs = df_pcs.drop(columns=overlap)
    df_pcs = df_pcs.merge(
        cfg_frame.drop(columns=["n_global_features", "n_add0", "n_add1"]),
        on=["config_id", "strategy_name"], how="left",
    )

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
    summary = summary.merge(
        cfg_frame[["config_id", "config_label", "strategy_name", "is_baseline", "is_winner",
                   "cluster_0_count", "cluster_1_count", "eval11_test_r2"]],
        on="config_id", how="left",
    )
    # Best / worst held-out station per configuration.
    best = df_pcs.loc[df_pcs.groupby("config_id")["r2"].idxmax(), ["config_id", "station"]]
    worst = df_pcs.loc[df_pcs.groupby("config_id")["r2"].idxmin(), ["config_id", "station"]]
    summary = summary.merge(best.rename(columns={"station": "best_station"}), on="config_id")
    summary = summary.merge(worst.rename(columns={"station": "worst_station"}), on="config_id")
    summary["is_reference"] = summary["config_id"].isin(ref_ids)

    # Temporal reference: eval-1.1 test R2 for the 47 pinned configs; the NEW
    # Clustering_Backbone54_k2 configs have no eval-1.1 row, so their temporal
    # reference is the full-training baseline's pooled test R2
    # (full_config_summary.csv, written by run_full_baseline.py — run it first
    # for the backfill, or re-run this script afterwards).
    summary["temporal_test_r2"] = summary["eval11_test_r2"]
    full_path = EXP_DIR / "full_config_summary.csv"
    if full_path.exists():
        fb = pd.read_csv(full_path)[["config_id", "full_pooled_r2"]]
        summary = summary.merge(fb, on="config_id", how="left")
        summary["temporal_test_r2"] = summary["temporal_test_r2"].fillna(summary["full_pooled_r2"])
        summary = summary.drop(columns=["full_pooled_r2"])
    summary["loso_minus_test_r2"] = summary["loso_mean_r2"] - summary["temporal_test_r2"]

    # Pooled LOSO R2/RMSE (concatenated folds) for direct comparison with eval-1.1.
    pooled = compute_pooled_loso_metrics(df_pcs, config, PREDICTIONS_DIR)
    if pooled:
        summary = summary.merge(
            pd.DataFrame(pooled).T.reset_index().rename(columns={"index": "config_id"}),
            on="config_id", how="left",
        )
    # Reference configs have no per-fold predictions in this run — backfill their
    # pooled LOSO from eval-1.2's summary (identical protocol). The merge suffix
    # only applies when the computed column already exists, so handle both.
    if eval12_pooled:
        ref_pooled = pd.DataFrame(eval12_pooled).T.reset_index().rename(columns={"index": "config_id"})
        summary = summary.merge(ref_pooled, on="config_id", how="left", suffixes=("", "_ref"))
        if "loso_pooled_r2" not in summary.columns:
            summary["loso_pooled_r2"] = np.nan
        if "loso_pooled_r2_ref" in summary.columns:
            summary["loso_pooled_r2"] = summary["loso_pooled_r2"].fillna(summary["loso_pooled_r2_ref"])
            summary = summary.drop(columns=["loso_pooled_r2_ref"])
        if "loso_pooled_rmse" not in summary.columns:
            summary["loso_pooled_rmse"] = np.nan
        if "loso_pooled_rmse_ref" in summary.columns:
            summary["loso_pooled_rmse"] = summary["loso_pooled_rmse"].fillna(summary["loso_pooled_rmse_ref"])
            summary = summary.drop(columns=["loso_pooled_rmse_ref"])
    summary["pooled_loso_minus_test_r2"] = summary["loso_pooled_r2"] - summary["temporal_test_r2"]

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
    print(f"[Artifacts] Wrote {EXP_DIR / 'loso_per_config_station.csv'}", flush=True)
    print(f"[Artifacts] Wrote {EXP_DIR / 'loso_config_summary.csv'}", flush=True)
    print(f"[Artifacts] Wrote {EXP_DIR / 'loso_station_summary.csv'}", flush=True)
    print(f"[Artifacts] Wrote {EXP_DIR / 'loso_per_regime_metrics.csv'}", flush=True)
    print(f"[Artifacts] Wrote {EXP_DIR / 'loso_per_year_metrics.csv'}", flush=True)

    if not args.skip_plots:
        plot_config_station_heatmap(df_pcs, EXP_DIR)
        plot_config_summary_bar(summary, EXP_DIR)
        plot_station_difficulty(station_summary, EXP_DIR)
        plot_station_boxplot(df_pcs, EXP_DIR)
        print(f"[Plots] Generated LOSO figures in {EXP_DIR}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("LOSO CONFIGURATION LEADERBOARD (derived_8.4-eval-1.3)", flush=True)
    print("=" * 70, flush=True)
    cols = ["config_id", "strategy_name", "loso_mean_r2", "loso_std_r2", "loso_min_r2",
            "loso_max_r2", "loso_mean_rmse", "loso_mean_bias", "temporal_test_r2",
            "loso_minus_test_r2", "is_winner"]
    if "loso_pooled_r2" in summary.columns:
        cols = ["config_id", "strategy_name", "loso_mean_r2", "loso_pooled_r2", "loso_std_r2",
                "loso_min_r2", "loso_max_r2", "loso_mean_rmse", "loso_mean_bias",
                "temporal_test_r2", "loso_minus_test_r2", "is_winner"]
    print(summary[cols].to_string(index=False), flush=True)

    print("\nSTATION DIFFICULTY (median LOSO R2 across configs)", flush=True)
    print(station_summary[["station", "median_r2", "mean_r2", "min_r2", "max_r2", "n_negative_r2"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
