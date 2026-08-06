#!/usr/bin/env python3
"""Driver for the derived_8.4-eval-mlp-1.0 hyperparameter sweep.

Runs every sweep config in BOTH families (1-regime global MLP and 2-regime
Clustering_V0_Full_k2 specialists) with a pool of concurrent GPU worker
subprocesses. Resumable: completed jobs (meta.json present) are skipped,
in-progress jobs (checkpoint.pt present) resume from their checkpoint.

Usage:
    python run_mlp_sweep.py [--config config.yaml] [--out .] [--resume]
                            [--n-parallel N] [--families 1regime,2regime]

Outputs:
    artifacts/tensors_{global,cluster0,cluster1}.npz   preprocessed tensors
    artifacts/test_meta.npz                            full test y / year / station
    artifacts/sweep_configs.json                       config id -> cfg
    artifacts/logs/<family>__<config>.log              per-job stdout
    models/<family>/<config_id>/{meta.json, ...}       per-job artifacts
    sweep_results.csv                                  ranking per family
    timing_log.json                                    per-job + total wall time
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
EVAL11_DIR = EXP_DIR.parent / "derived_8.4-eval-1.1"
sys.path.insert(0, str(EVAL11_DIR))
sys.path.insert(0, str(EXP_DIR))

from eval11.data import load_experiment_data  # noqa: E402
from eval11.routers import get_router  # noqa: E402
from mlp10.data import build_feature_set, save_feature_set  # noqa: E402

FAMILIES = ("1regime", "2regime")


def load_cluster_deltas(config: dict, exp_dir: Path) -> list[str]:
    """Load the eval-1.1 winner's cluster-1 additions (fallback: config list)."""
    sel_path = exp_dir / "selected_features.json"
    if sel_path.exists():
        meta = json.loads(sel_path.read_text(encoding="utf-8"))
        for rec in meta.get("leaderboard", []):
            if rec.get("candidate_id") == "Clustering_V0_Full_k2_c0_0_c1_10":
                add = rec.get("cluster_1_additions", "")
                return [f for f in add.split(";") if f]
    return list(config["cluster_config"]["cluster_1_delta_features"])


def build_all_tensors(data, config: dict, artifacts: Path) -> None:
    target = data.target
    cluster_cfg = config["cluster_config"]
    backbone = list(config["shared_backbone_54"])
    c1_deltas = load_cluster_deltas(config, EXP_DIR)

    router = get_router(cluster_cfg["strategy"], data.v0_features, seed=int(config["model"]["seed"]))
    router.fit(data.trainval)
    labels_tv = router.predict(data.trainval)
    labels_te = router.predict(data.test)
    np.save(artifacts / "labels_trainval.npy", labels_tv)
    np.save(artifacts / "labels_test.npy", labels_te)

    hold_frac = float(config["sweep"]["holdout_frac"])
    seed = int(config["model"]["seed"])

    # Global (1-regime): all trainval rows, 54 backbone feats.
    fs_global = build_feature_set(
        data.trainval, data.test, backbone, target,
        holdout_frac=hold_frac, seed=seed, salt="global",
    )
    save_feature_set(artifacts / "tensors_global.npz", fs_global)

    # Cluster feature sets (2-regime specialists).
    for cl, extra in (("0", []), ("1", c1_deltas)):
        feats = [*backbone, *extra]
        tv_mask = labels_tv == int(cl)
        te_mask = labels_te == int(cl)
        fs_cl = build_feature_set(
            data.trainval.loc[tv_mask].reset_index(drop=True),
            data.test.loc[te_mask].reset_index(drop=True),
            feats, target,
            holdout_frac=hold_frac, seed=seed, salt=f"cluster{cl}",
            test_positions=np.where(te_mask)[0],
        )
        save_feature_set(artifacts / f"tensors_cluster{cl}.npz", fs_cl)

    np.savez_compressed(
        artifacts / "test_meta.npz",
        y_test=data.test[target].to_numpy(dtype=np.float64),
        year=data.test["year"].to_numpy(dtype=np.int64),
        station=np.asarray(data.test["station_id"], dtype=object),
    )
    print(f"[tensors] global {fs_global['X_train'].shape} | "
          f"cluster0 {np.load(artifacts/'tensors_cluster0.npz')['X_train'].shape} | "
          f"cluster1 {np.load(artifacts/'tensors_cluster1.npz')['X_train'].shape}", flush=True)


def build_sweep_configs(config: dict) -> dict[str, dict]:
    defaults = dict(config["sweep"]["defaults"])
    defaults["seed"] = int(config["model"]["seed"])
    defaults["data_version"] = int(config["sweep"].get("data_version", 1))
    out: dict[str, dict] = {}
    for entry in config["sweep"]["configs"]:
        cfg = dict(defaults)
        cfg.update({k: v for k, v in entry.items() if k != "id"})
        cfg["id"] = entry["id"]
        cfg["hidden_sizes"] = [int(h) for h in entry["hidden_sizes"]]
        out[entry["id"]] = cfg
    return out


def job_dir(out: Path, family: str, config_id: str) -> Path:
    return out / "models" / family / config_id


def current_data_version(config: dict) -> int:
    return int(config["sweep"].get("data_version", 1))


def job_complete(jdir: Path, config: dict) -> bool:
    meta = jdir / "meta.json"
    if not meta.exists():
        return False
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("status") != "completed":
        return False
    # Stale artifacts from an older data version must be invalidated.
    return payload.get("config", {}).get("data_version") == current_data_version(config)


def _ckpt_version_ok(jdir: Path, version: int) -> bool:
    for ck in [jdir / "checkpoint.pt", jdir / "spec_0" / "checkpoint.pt", jdir / "spec_1" / "checkpoint.pt"]:
        if ck.exists():
            try:
                import torch

                ckpt = torch.load(ck, map_location="cpu", weights_only=False)
                return ckpt.get("config", {}).get("data_version") == version
            except Exception:
                return False
    return True


def job_resumable(jdir: Path, config: dict) -> bool:
    # 2-regime jobs checkpoint under spec_0/spec_1; 1-regime at the top level.
    has_ckpt = (jdir / "checkpoint.pt").exists() or (jdir / "spec_0" / "checkpoint.pt").exists()
    return has_ckpt and _ckpt_version_ok(jdir, current_data_version(config))


def _clean_job_dir(jdir: Path) -> None:
    import shutil

    if jdir.exists():
        shutil.rmtree(jdir)


def make_jobs(out: Path, families, config_ids, config: dict) -> list[tuple[str, str, str]]:
    """Return (family, config_id, mode) with mode in {skip, resume, fresh}.

    Jobs whose artifacts were produced under an older data_version are
    invalidated (removed) so they re-train with the current holdout split.
    """
    version = current_data_version(config)
    jobs = []
    for family in families:
        for cid in config_ids:
            jdir = job_dir(out, family, cid)
            if job_complete(jdir, config):
                mode = "skip"
            elif job_resumable(jdir, config):
                mode = "resume"
            else:
                if (jdir / "meta.json").exists() or (jdir / "checkpoint.pt").exists() \
                        or (jdir / "spec_0" / "checkpoint.pt").exists():
                    print(f"[invalidate] {family}/{cid}: stale artifacts (data_version < {version})", flush=True)
                    _clean_job_dir(jdir)
                mode = "fresh"
            jobs.append((family, cid, mode))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-parallel", type=int, default=None)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--only", default=None, help="comma list of config ids (debug)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    artifacts = args.out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "logs").mkdir(parents=True, exist_ok=True)

    n_parallel = args.n_parallel or int(config["sweep"]["n_parallel"])
    families = tuple(f for f in FAMILIES if f in args.families.split(","))

    # ---- prepare tensors + sweep configs (idempotent; cheap to redo) ----
    t0 = time.perf_counter()
    data = load_experiment_data(PROJECT_ROOT, config)
    build_all_tensors(data, config, artifacts)
    sweep_configs = build_sweep_configs(config)
    with open(artifacts / "sweep_configs.json", "w", encoding="utf-8") as f:
        json.dump(sweep_configs, f, indent=2)

    config_ids = [c["id"] for c in config["sweep"]["configs"]]
    if args.only:
        config_ids = [c for c in config_ids if c in args.only.split(",")]

    jobs = make_jobs(args.out, families, config_ids, config)
    todo = [j for j in jobs if j[2] != "skip"]
    print(f"[sweep] {len(jobs)} jobs total, {len([j for j in jobs if j[2]=='skip'])} already done, "
          f"{len([j for j in jobs if j[2]=='resume'])} to resume, {len([j for j in jobs if j[2]=='fresh'])} fresh",
          flush=True)

    # ---- run job pool ----
    t_start = time.perf_counter()
    queue = list(todo)
    active: list[tuple[subprocess.Popen, tuple[str, str, str], Path, float]] = []
    per_job_time: dict[str, float] = {}

    while queue or active:
        while len(active) < n_parallel and queue:
            family, cid, mode = queue.pop(0)
            jdir = job_dir(args.out, family, cid)
            log_path = artifacts / "logs" / f"{family}__{cid}.log"
            cmd = [
                sys.executable, str(EXP_DIR / "run_mlp_worker.py"),
                "--family", family, "--config-id", cid,
                "--artifacts", str(artifacts), "--out", str(args.out),
            ]
            if args.resume and mode == "resume":
                cmd.append("--resume")
            logf = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(EXP_DIR))
            active.append((proc, (family, cid, mode), logf, time.perf_counter()))
            print(f"[launch] {family}/{cid} ({mode})", flush=True)

        # reap finished workers
        still_active = []
        for proc, (family, cid, mode), logf, started in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((proc, (family, cid, mode), logf, started))
                continue
            logf.close()
            elapsed = time.perf_counter() - started
            jdir = job_dir(args.out, family, cid)
            ok = job_complete(jdir, config)
            status = "ok" if (ok and rc == 0) else f"FAILED(rc={rc})"
            print(f"[finish] {family}/{cid} {status} wall={elapsed:.1f}s", flush=True)
            per_job_time[f"{family}/{cid}"] = elapsed
            if not ok and rc != 0:
                print(f"         log: {artifacts / 'logs' / f'{family}__{cid}.log'}", flush=True)
        active = still_active

        if active:
            time.sleep(2.0)

    sweep_wall = time.perf_counter() - t_start
    print(f"\n[sweep] all jobs done in {sweep_wall:.1f}s wall", flush=True)

    # ---- collect results ----
    collect_sweep_results(args.out, config, families, config_ids, sweep_wall, per_job_time)


def collect_sweep_results(out: Path, config: dict, families, config_ids, sweep_wall: float, per_job_time: dict) -> None:
    rows = []
    timing: dict[str, object] = {"sweep_wall_s": sweep_wall, "jobs": {}}
    for family in families:
        for cid in config_ids:
            jdir = job_dir(out, family, cid)
            meta_path = jdir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            test = meta["test"]
            rows.append({
                "family": family,
                "config_id": cid,
                "hidden_sizes": str(meta["config"]["hidden_sizes"]),
                "activation": meta["config"]["activation"],
                "dropout": meta["config"]["dropout"],
                "lr": meta["config"]["lr"],
                "weight_decay": meta["config"]["weight_decay"],
                "batch_size": meta["config"]["batch_size"],
                "loss": meta["config"]["loss"],
                "holdout_rmse": meta["holdout_rmse"],
                "test_r2": test["r2"],
                "test_rmse": test["rmse"],
                "test_bias": test["bias"],
                "test_mae": test["mae"],
                "epochs": meta["epochs"],
                "best_epoch": meta["best_epoch"],
                "train_time_s": meta["train_time_s"],
                "wall_time_s": per_job_time.get(f"{family}/{cid}", float("nan")),
                "n_params": meta["n_params"],
            })
            timing["jobs"][f"{family}/{cid}"] = {
                "train_time_s": meta["train_time_s"],
                "wall_time_s": per_job_time.get(f"{family}/{cid}", None),
                "holdout_rmse": meta["holdout_rmse"],
                "test_r2": test["r2"],
            }
    df = pd.DataFrame(rows)
    if df.empty:
        print("[sweep] no completed jobs yet — run again with --resume to finish", flush=True)
        df.to_csv(out / "sweep_results.csv", index=False)
        with open(out / "timing_log.json", "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)
        return
    df = df.sort_values(["family", "holdout_rmse"]).reset_index(drop=True)
    df.to_csv(out / "sweep_results.csv", index=False)
    with open(out / "timing_log.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(f"[sweep] wrote sweep_results.csv ({len(df)} rows) + timing_log.json", flush=True)
    for family in families:
        sub = df[df["family"] == family].head(5)
        print(f"\n[top5 {family} by holdout RMSE]", flush=True)
        print(sub[["config_id", "holdout_rmse", "test_r2", "test_rmse", "train_time_s"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
