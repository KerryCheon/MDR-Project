#!/usr/bin/env python3
"""Driver for the derived_8.4-eval-mlp-2.3 sweep (3 families, 3-phase seeds).

Protocol is data_version 10: train on train (2017-2020), early-stop on val
(2021-2022), test on test (2023-2025). Every job is ALSO evaluated at its
best-val epoch on the AUX2020 holdout (the 2020 slice of train, n=2519) —
DIAGNOSTIC ONLY (mlp-1.2 documented it measures train fit, not generalization).

Families (unchanged from 2.0): 2regime_96 (c0/c1 on the 96-pool), 2regime_54
(c0 = 54 backbone, c1 = 54 + eval-1.1 winner's 10 delta features), and
2regime_mixed (c0 = 96-pool, c1 = 54 + deltas) — the per-cluster-optimal
feature allocation that broke the ceiling in 2.0 (val top-5 ensemble 0.8003).

Three-phase, 3-seed sweep (NEW in 2.3: the FULL 3-seed pool — phase-2/3
top-Ns are capped at the deduped family sizes, so EVERY config trains seeds
{42, 7, 123}; 2.2 gave only the phase-3 top-M a 3rd seed, which limited the
3-seed val-year/selection diagnostics to a subset and left the 54 winner
flip under-documented):
  phase 1 (default): every (family x config) trains seed[0] (42).
  phase 2 (--phase2-top-n N): the top-N configs per family by val RMSE
    (2-seed-able, honest signal) train seed[1] (7) — N = family size.
  phase 3 (--phase3-top-n M): the top-M per family by the now-2-seed val RMSE
    train seed[2] (123) -> 3-seed mean val RMSE winner selection — M = family
    size.
  Per-family N/M via int-or-dict sweep.phase2_top_n / phase3_top_n. Final
  per-config metrics are the mean over completed seeds; test predictions are
  the seed-mean.

NEW in 2.2 (kept): `build_all_tensors` also saves `artifacts/val_meta.npz`
(y_val/year/station of the full official val set) and every job saves
best-val predictions (`val_preds.npy`, from mlp23/trainer.py) — together they
feed the offline val-year (2021 vs 2022) selection-reliability diagnostic
(`analyze_val_years.py`). No selection-rule change.

Architectures: ONLY `mlp` configs run in 2.3 — `fg`/`plr` are documented
negatives from 2.0 (best 0.782 / 0.720 vs plain-MLP 0.790), `swa` is a
documented negative from 2.1 (0/152 deployments with the RNG guard in place),
and 54-family 3-layer is a documented 2.2 negative (the val-overfit trap:
the 2.2 54 val top-10 was 3-layer-dominated, test 0.7596-0.7790 vs the
2-layer 320^2 frontier 0.79+); none get GPU except the 3 bit-identity
anchors and 2 re-check probes. The winner pool filter in the phase-2/3
selection stays mlp/fg/plr so the protocol text is unchanged (effectively
mlp-only).

Per-config model layout:
    models/<family>/<config_id>/seed_<s>/{checkpoint.pt, best_model.pt,
        curves.npy, preds.npy, meta.json}          (spec_0/ spec_1/ for cluster)
    models/<family>/<config_id>/meta.json          aggregated (mean over seeds)
    models/<family>/<config_id>/preds.npy          seed-mean test predictions

Usage:
    python run_mlp_sweep.py [--config config.yaml] [--out .] [--resume]
                            [--n-parallel N] [--families 2regime_96,...]
                            [--only id1,id2] [--phase2-top-n N]
                            [--phase3-top-n M] [--phase2-only] [--phase3-only]
                            [--smoke]

Outputs:
    artifacts/tensors_<family>[_cluster{0,1}].npz   preprocessed tensors (+aux)
    artifacts/test_meta.npz                          full test y / year / station
    artifacts/val_meta.npz                           full val y / year / station
    artifacts/labels_{train,val,test,aux}.npy        cluster labels
    artifacts/sweep_configs.json                     config id -> cfg
    artifacts/logs/<family>__<config>__s<seed>.log   per-job stdout
    models/<family>/<config_id>/{meta.json, preds.npy, ...}
    sweep_results.csv                                ranking per family
    timing_log.json                                  per-job + total wall time
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
from mlp23.data import build_feature_set, save_feature_set  # noqa: E402


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


def _resolve_feature_key(key: str, config: dict, data) -> list[str]:
    """Resolve a family feature-set key to a concrete feature list.

    Keys: "candidate_pool_96" (the 96-feature selection lab pool),
    "backbone_54" (the shared 54-feature global backbone), and
    "backbone_54_delta" (backbone + the eval-1.1 winner's 10 cluster-1 delta
    features — the XGBoost winner's feature structure).
    """
    if key == "candidate_pool_96":
        return list(data.candidate_pool)
    if key == "backbone_54":
        return list(config["shared_backbone_54"])
    if key == "backbone_54_delta":
        base = list(config["shared_backbone_54"])
        c1_deltas = load_cluster_deltas(config, EXP_DIR)
        c1 = list(dict.fromkeys([*base, *c1_deltas]))
        return [f for f in c1 if f in set(data.feature_columns)]
    raise ValueError(f"Unknown feature-set key: {key}")


def family_features(family_id: str, config: dict, data) -> dict[str, list[str]]:
    """Return {feature_set_key: [feature cols]} for a family.

    New in 2.0: a family may declare `feature_sets` (cluster id -> feature-set
    key) for an explicit per-cluster feature allocation — e.g. 2regime_mixed
    uses c0 = candidate_pool_96 and c1 = backbone_54_delta. Without
    `feature_sets` the mlp-1.3 schema is used: 'global' families get the single
    key ""; 'cluster' families get base features for c0 and base+delta for c1.
    """
    fam_cfg = next(f for f in config["families"] if f["id"] == family_id)
    if "feature_sets" in fam_cfg:
        return {f"_cluster{cl}": _resolve_feature_key(key, config, data)
                for cl, key in fam_cfg["feature_sets"].items()}
    feats_cfg = fam_cfg["features"]
    if feats_cfg == "candidate_pool_96":
        base = list(data.candidate_pool)
    else:
        base = list(config["shared_backbone_54"])
    if fam_cfg["structure"] == "global":
        return {"": base}
    c1_deltas = load_cluster_deltas(config, EXP_DIR)
    c1 = list(dict.fromkeys([*base, *c1_deltas]))
    c1 = [f for f in c1 if f in set(data.feature_columns)]
    return {"_cluster0": base, "_cluster1": c1}


def build_all_tensors(data, config: dict, artifacts: Path) -> None:
    target = data.target
    cluster_cfg = config["cluster_config"]
    aux_year = int(config["sweep"].get("aux_year", 2020))

    router = get_router(cluster_cfg["strategy"], data.v0_features, seed=int(config["model"]["seed"]))
    router.fit(data.trainval)
    labels_train = router.predict(data.train)
    labels_val = router.predict(data.val)
    labels_te = router.predict(data.test)
    np.save(artifacts / "labels_train.npy", labels_train)
    np.save(artifacts / "labels_val.npy", labels_val)
    np.save(artifacts / "labels_test.npy", labels_te)

    aux_frame = data.train[data.train["year"] == aux_year].reset_index(drop=True)
    labels_aux = router.predict(aux_frame)
    np.save(artifacts / "labels_aux.npy", labels_aux)
    print(f"[tensors] aux2020 holdout: {len(aux_frame)} rows", flush=True)

    for fam_cfg in config["families"]:
        fid = fam_cfg["id"]
        featsets = family_features(fid, config, data)
        for suffix, feats in featsets.items():
            if fam_cfg["structure"] == "global":
                fs = build_feature_set(
                    data.train, data.val, data.test, feats, target,
                    aux=aux_frame,
                )
                save_feature_set(artifacts / f"tensors_{fid}{suffix}.npz", fs)
            else:
                cl = suffix.replace("_cluster", "")
                tr_mask = labels_train == int(cl)
                va_mask = labels_val == int(cl)
                te_mask = labels_te == int(cl)
                au_mask = labels_aux == int(cl)
                fs = build_feature_set(
                    data.train.loc[tr_mask].reset_index(drop=True),
                    data.val.loc[va_mask].reset_index(drop=True),
                    data.test.loc[te_mask].reset_index(drop=True),
                    feats, target,
                    test_positions=np.where(te_mask)[0],
                    aux=aux_frame.loc[au_mask].reset_index(drop=True),
                )
                save_feature_set(artifacts / f"tensors_{fid}{suffix}.npz", fs)
            print(f"[tensors] {fid}{suffix}: train {fs['X_train'].shape} "
                  f"val {fs['X_val'].shape} aux {fs['X_aux'].shape} test {fs['X_test'].shape}", flush=True)

    np.savez_compressed(
        artifacts / "test_meta.npz",
        y_test=data.test[target].to_numpy(dtype=np.float64),
        year=data.test["year"].to_numpy(dtype=np.int64),
        station=np.asarray(data.test["station_id"], dtype=object),
    )
    # NEW in 2.2: full official-val metadata for the val-year (2021 vs 2022)
    # selection-reliability diagnostic (analyze_val_years.py); val_preds.npy
    # per job + labels_val.npy reconstruct the per-year val RMSE offline.
    np.savez_compressed(
        artifacts / "val_meta.npz",
        y_val=data.val[target].to_numpy(dtype=np.float64),
        year=data.val["year"].to_numpy(dtype=np.int64),
        station=np.asarray(data.val["station_id"], dtype=object),
    )
    print("[tensors] done", flush=True)


def build_sweep_configs(config: dict) -> dict[str, dict]:
    defaults = dict(config["sweep"]["defaults"])
    defaults["seed"] = int(config["model"]["seed"])
    defaults["data_version"] = int(config["sweep"].get("data_version", 4))
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


def job_dir(out: Path, family: str, config_id: str, seed: int) -> Path:
    return out / "models" / family / config_id / f"seed_{seed}"


def current_data_version(config: dict) -> int:
    return int(config["sweep"].get("data_version", 4))


SMOKE_VERSION = -1  # smoke jobs use this so real runs never reuse them
SMOKE = False       # set by --smoke; completion checks then use SMOKE_VERSION


def expected_version(config: dict) -> int:
    return SMOKE_VERSION if SMOKE else current_data_version(config)


def seed_complete(jdir: Path, version: int) -> bool:
    meta = jdir / "meta.json"
    if not meta.exists():
        return False
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("status") != "completed":
        return False
    return payload.get("config", {}).get("data_version") == version


def _ckpt_version_ok(jdir: Path, version: int) -> bool:
    cks = [jdir / "checkpoint.pt", jdir / "spec_0" / "checkpoint.pt", jdir / "spec_1" / "checkpoint.pt"]
    for ck in cks:
        if ck.exists():
            try:
                import torch

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


def read_seed_meta(out: Path, family: str, cid: str, seed: int) -> dict | None:
    p = job_dir(out, family, cid, seed) / "meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def aggregate_config(out: Path, family: str, cid: str, config: dict) -> dict | None:
    """Aggregate completed seeds of one config into models/<family>/<cid>/.

    Writes meta.json (mean metrics over completed seeds) + preds.npy (seed-mean
    test predictions). Idempotent: re-called whenever another seed completes.
    """
    from eval11.evaluator import compute_metrics  # noqa: F401

    version = expected_version(config)
    seeds = [int(s) for s in config["sweep"].get("seeds", [42])]
    metas = {s: read_seed_meta(out, family, cid, s) for s in seeds}
    metas = {s: m for s, m in metas.items() if m is not None and m.get("status") == "completed"
             and m.get("config", {}).get("data_version") == version}
    if not metas:
        return None

    cdir = out / "models" / family / cid
    cdir.mkdir(parents=True, exist_ok=True)

    preds_list = []
    for s, m in sorted(metas.items()):
        p = job_dir(out, family, cid, s) / "preds.npy"
        if p.exists():
            preds_list.append(np.load(p))
    if not preds_list:
        return None
    seed_mean_preds = np.mean(preds_list, axis=0)
    np.save(cdir / "preds.npy", seed_mean_preds)

    test_meta = np.load(out / "artifacts" / "test_meta.npz")
    test = compute_metrics(test_meta["y_test"], seed_mean_preds)

    val_rmses = [m["val_rmse"] for m in metas.values() if m.get("val_rmse") is not None]
    aux_rmses = [m["aux_rmse"] for m in metas.values() if m.get("aux_rmse") is not None]
    agg = {
        "family": family,
        "config_id": cid,
        "config": next(iter(metas.values()))["config"],
        "seeds": {str(s): {k: m[k] for k in ("val_rmse", "aux_rmse", "test", "best_epoch", "epochs", "train_time_s", "n_params")}
                  for s, m in sorted(metas.items())},
        "val_rmse": float(np.mean(val_rmses)) if val_rmses else float("nan"),
        "aux_rmse": float(np.mean(aux_rmses)) if aux_rmses else float("nan"),
        "robust_score": float(np.mean([0.5 * (v + a) for v, a in zip(val_rmses, aux_rmses)])) if (val_rmses and aux_rmses and len(val_rmses) == len(aux_rmses)) else float("nan"),
        "test": test,
        "epochs": max(int(m.get("epochs", 0)) for m in metas.values()),
        "best_epoch": max(int(m.get("best_epoch", 0)) for m in metas.values()),
        "train_time_s": float(sum(m.get("train_time_s", 0.0) for m in metas.values())),
        "n_params": int(next(iter(metas.values())).get("n_params", 0)),
        "n_seeds": len(metas),
        "status": "completed",
    }

    # per-cluster test metrics (2-regime families) from the seed-mean preds
    labels_te_path = out / "artifacts" / "labels_test.npy"
    first_meta = next(iter(metas.values()))
    if "per_cluster" in first_meta and labels_te_path.exists():
        labels_te = np.load(labels_te_path)
        per_cluster = {}
        for cl in ["0", "1"]:
            mask = labels_te == int(cl)
            cm = compute_metrics(test_meta["y_test"][mask], seed_mean_preds[mask])
            info = first_meta["per_cluster"][cl]
            per_cluster[cl] = {
                "n_train": int(info["n_train"]),
                "n_val": int(info["n_val"]),
                "n_aux": int(info["n_aux"]),
                "n_test": int(mask.sum()),
                "test": cm,
            }
        agg["per_cluster"] = per_cluster

    # SWA deployment + pooled live/SWA val RMSE — the aggregated meta cannot
    # otherwise report them (2.0 left val_rmse_live/val_rmse_swa/deployed NaN in
    # the leaderboard). Pooled = n_val-weighted over clusters, mean over seeds;
    # `deployed` = "swa" iff ANY (seed, cluster) specialist deployed the SWA
    # snapshot (honest within-val rule per specialist), n_deployed_swa counts them.
    if "per_cluster" in agg:
        live_vals: list[tuple[int, float]] = []
        swa_vals: list[tuple[int, float]] = []
        n_deployed_swa = 0
        n_specs = 0
        for m in metas.values():
            pc = m.get("per_cluster", {})
            for cl in ("0", "1"):
                cm = pc.get(cl, {})
                n_specs += 1
                if cm.get("deployed") == "swa":
                    n_deployed_swa += 1
                if cm.get("val_rmse_live") is not None:
                    live_vals.append((int(cm.get("n_val", 0)), float(cm["val_rmse_live"])))
                if cm.get("val_rmse_swa") is not None:
                    swa_vals.append((int(cm.get("n_val", 0)), float(cm["val_rmse_swa"])))

        def _pool(vals: list[tuple[int, float]]) -> float:
            # never-started SWA evals keep inf in the per-seed metas; treat
            # them as missing so the pooled value is NaN, not inf (reporting).
            finite = [(n, r) for n, r in vals if np.isfinite(r)]
            if not finite:
                return float("nan")
            tot = sum(n for n, _ in finite)
            return float(np.sqrt(sum((n / tot) * r**2 for n, r in finite)))

        agg["val_rmse_live"] = _pool(live_vals) if live_vals else float("nan")
        agg["val_rmse_swa"] = _pool(swa_vals) if swa_vals else float("nan")
        agg["deployed"] = "swa" if n_deployed_swa else "live"
        agg["n_deployed_swa"] = n_deployed_swa
        agg["n_specs"] = n_specs

    with open(cdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)
    return agg


def robust_ranking(out: Path, family: str, config: dict, metric: str = "robust_score") -> pd.DataFrame:
    """Rank a family's completed configs by a selection metric (MLP-selectable only)."""
    rows = []
    for cid in family_config_ids(config, family):
        meta_path = out / "models" / family / cid / "meta.json"
        if not meta_path.exists():
            continue
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "config_id": cid,
            "architecture": m.get("config", {}).get("architecture", "mlp"),
            "val_rmse": m.get("val_rmse"),
            "aux_rmse": m.get("aux_rmse"),
            "robust_score": m.get("robust_score"),
            "test_r2": m.get("test", {}).get("r2"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(metric, na_position="last").reset_index(drop=True)
    return df


def make_jobs(out: Path, families, config_ids_by_family, seeds, config: dict) -> list[tuple[str, str, int, str]]:
    version = expected_version(config)
    jobs = []
    for family in families:
        for cid in config_ids_by_family[family]:
            for seed in seeds:
                jdir = job_dir(out, family, cid, seed)
                if seed_complete(jdir, version):
                    mode = "skip"
                elif seed_resumable(jdir, version):
                    mode = "resume"
                else:
                    if jdir.exists() and ((jdir / "meta.json").exists() or (jdir / "checkpoint.pt").exists()
                                          or (jdir / "spec_0" / "checkpoint.pt").exists()):
                        print(f"[invalidate] {family}/{cid}/seed_{seed}: stale artifacts "
                              f"(data_version < {version})", flush=True)
                        _clean_job_dir(jdir)
                    mode = "fresh"
                jobs.append((family, cid, seed, mode))
    return jobs


def run_jobs(out: Path, jobs, n_parallel: int, config: dict) -> dict[str, float]:
    """Run a list of (family, cid, seed, mode) jobs; returns per-job wall times."""
    artifacts = out / "artifacts"
    per_job_time: dict[str, float] = {}
    queue = list(jobs)
    active: list[tuple[subprocess.Popen, tuple[str, str, int, str], object, float]] = []

    while queue or active:
        while len(active) < n_parallel and queue:
            family, cid, seed, mode = queue.pop(0)
            log_path = artifacts / "logs" / f"{family}__{cid}__s{seed}.log"
            cmd = [
                sys.executable, str(EXP_DIR / "run_mlp_worker.py"),
                "--family", family, "--config-id", cid, "--seed", str(seed),
                "--artifacts", str(artifacts), "--out", str(out),
            ]
            if mode == "resume":
                cmd.append("--resume")
            logf = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(EXP_DIR))
            active.append((proc, (family, cid, seed, mode), logf, time.perf_counter()))
            print(f"[launch] {family}/{cid}/seed_{seed} ({mode})", flush=True)

        still_active = []
        for proc, (family, cid, seed, mode), logf, started in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((proc, (family, cid, seed, mode), logf, started))
                continue
            logf.close()
            elapsed = time.perf_counter() - started
            ok = seed_complete(job_dir(out, family, cid, seed), expected_version(config))
            status = "ok" if (ok and rc == 0) else f"FAILED(rc={rc})"
            print(f"[finish] {family}/{cid}/seed_{seed} {status} wall={elapsed:.1f}s", flush=True)
            per_job_time[f"{family}/{cid}/seed_{seed}"] = elapsed
            if not ok and rc != 0:
                print(f"         log: {artifacts / 'logs' / f'{family}__{cid}__s{seed}.log'}", flush=True)
            aggregate_config(out, family, cid, config)
        active = still_active
        if active:
            time.sleep(2.0)
    return per_job_time


def collect_sweep_results(out: Path, config: dict, families, per_job_time: dict) -> None:
    rows = []
    timing: dict[str, object] = {"jobs": {}}
    for family in families:
        for cid in family_config_ids(config, family):
            meta_path = out / "models" / family / cid / "meta.json"
            if not meta_path.exists():
                continue
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            test = m["test"]
            row = {
                "family": family,
                "config_id": cid,
                "architecture": m["config"].get("architecture", "mlp"),
                "hidden_sizes": str(m["config"].get("hidden_sizes", "")),
                "ft_d": m["config"].get("ft_d", ""),
                "ft_layers": m["config"].get("ft_layers", ""),
                "activation": m["config"]["activation"],
                "norm": m["config"].get("norm", "bn"),
                "dropout": m["config"]["dropout"],
                "lr": m["config"]["lr"],
                "weight_decay": m["config"]["weight_decay"],
                "batch_size": m["config"]["batch_size"],
                "loss": m["config"]["loss"],
                "n_seeds": m.get("n_seeds", 1),
                "val_rmse": m.get("val_rmse"),
                "aux_rmse": m.get("aux_rmse"),
                "robust_score": m.get("robust_score"),
                "test_r2": test["r2"],
                "test_rmse": test["rmse"],
                "test_bias": test["bias"],
                "test_mae": test["mae"],
                "epochs": m.get("epochs"),
                "best_epoch": m.get("best_epoch"),
                "train_time_s": m.get("train_time_s"),
                "n_params": m.get("n_params"),
            }
            # per-seed details for the report
            for s, sm in sorted(m.get("seeds", {}).items(), key=lambda kv: int(kv[0])):
                row[f"seed{s}_val_rmse"] = sm.get("val_rmse")
                row[f"seed{s}_aux_rmse"] = sm.get("aux_rmse")
                row[f"seed{s}_test_r2"] = sm.get("test", {}).get("r2")
            rows.append(row)
            timing["jobs"][f"{family}/{cid}"] = {
                "train_time_s": m.get("train_time_s"),
                "wall_time_s": sum(per_job_time.get(f"{family}/{cid}/seed_{s}", 0.0)
                                   for s in [int(s) for s in config["sweep"].get("seeds", [42])]),
                "val_rmse": m.get("val_rmse"),
                "aux_rmse": m.get("aux_rmse"),
                "robust_score": m.get("robust_score"),
                "test_r2": test["r2"],
                "n_seeds": m.get("n_seeds", 1),
            }
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["family", "robust_score"], na_position="last").reset_index(drop=True)
    df.to_csv(out / "sweep_results.csv", index=False)
    with open(out / "timing_log.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(f"[sweep] wrote sweep_results.csv ({len(df)} rows) + timing_log.json", flush=True)
    if df.empty:
        return
    for family in families:
        sub = df[df["family"] == family].head(6)
        if sub.empty:
            continue
        cols = [c for c in ["config_id", "architecture", "n_seeds", "robust_score", "val_rmse", "aux_rmse", "test_r2", "train_time_s"] if c in sub.columns]
        print(f"\n[top6 {family} by robust score]", flush=True)
        print(sub[cols].to_string(index=False), flush=True)


def top_n_for(spec, family: str, default: int) -> int:
    """Resolve a phase top-N spec (int, or {family: int}) for one family."""
    if spec is None:
        return default
    if isinstance(spec, dict):
        return int(spec.get(family, default))
    return int(spec)


def _phase_jobs(out: Path, families, seeds_for_phase, config: dict, metric: str,
                top_n_spec, phase_name: str) -> list[tuple[str, str, int, str]]:
    """Build the (family, cid, seed, mode) jobs for a later-seed phase.

    Phase 2 (seed 7) and phase 3 (seed 123) both add one seed to the top-N
    honest-architecture configs per family by the phase metric (val_rmse).
    """
    jobs: list[tuple[str, str, int, str]] = []
    for family in families:
        rank = robust_ranking(out, family, config, metric=metric)
        # winner pool = plain MLP + FeatureGroupedMLP + PLR (residual/ft are
        # reference-only, documented failures in mlp-1.1/1.2).
        honest = rank[rank["architecture"].isin(("mlp", "fg", "plr"))].head(top_n_for(top_n_spec, family, 10))
        for _, r in honest.iterrows():
            cid = r["config_id"]
            jdir = job_dir(out, family, cid, seeds_for_phase)
            if seed_complete(jdir, expected_version(config)):
                mode = "skip"
            elif seed_resumable(jdir, expected_version(config)):
                mode = "resume"
            else:
                mode = "fresh"
            jobs.append((family, cid, seeds_for_phase, mode))
    n_done = len([j for j in jobs if j[3] == "skip"])
    n_resume = len([j for j in jobs if j[3] == "resume"])
    print(f"[{phase_name}] top-{top_n_spec} MLP configs/family by {metric} get seed {seeds_for_phase}: "
          f"{len(jobs)} jobs ({n_done} done, {n_resume} resume)", flush=True)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-parallel", type=int, default=None)
    parser.add_argument("--families", default=None, help="comma list of family ids")
    parser.add_argument("--only", default=None, help="comma list of config ids (debug)")
    parser.add_argument("--phase2-top-n", type=int, default=None,
                        help="after phase 1, add the 2nd seed to the top-N plain-MLP configs "
                             "per family by the phase-2 metric (default: sweep.phase2_top_n)")
    parser.add_argument("--phase2-metric", default=None,
                        help="ranking metric for phase-2 2nd-seed selection: "
                             "robust_score (default) | val_rmse")
    parser.add_argument("--phase2-only", action="store_true",
                        help="skip phase-1 jobs; run phases 2 and 3 only")
    parser.add_argument("--phase3-top-n", type=int, default=None,
                        help="after phase 2, add the 3rd seed to the top-N MLP configs per "
                             "family by the phase-3 metric (default: sweep.phase3_top_n)")
    parser.add_argument("--phase3-metric", default=None,
                        help="ranking metric for phase-3 3rd-seed selection: val_rmse (default)")
    parser.add_argument("--phase3-only", action="store_true",
                        help="skip phases 1-2; run the phase-3 3rd-seed jobs only")
    parser.add_argument("--smoke", action="store_true",
                        help="train 3 epochs max (data_version -1: never reused by real runs)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    global SMOKE
    SMOKE = args.smoke

    artifacts = args.out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "logs").mkdir(parents=True, exist_ok=True)

    n_parallel = args.n_parallel or int(config["sweep"]["n_parallel"])
    families = [f["id"] for f in config["families"]]
    if args.families:
        families = [f for f in families if f in args.families.split(",")]
    seeds = [int(s) for s in config["sweep"].get("seeds", [42])]
    phase2_top_n = args.phase2_top_n if args.phase2_top_n is not None else config["sweep"].get("phase2_top_n", 10)
    phase2_metric = args.phase2_metric or str(config["sweep"].get("phase2_metric", "robust_score"))
    if phase2_metric not in ("robust_score", "val_rmse"):
        raise SystemExit(f"--phase2-metric must be robust_score or val_rmse, got {phase2_metric}")
    phase3_top_n = args.phase3_top_n if args.phase3_top_n is not None else config["sweep"].get("phase3_top_n", None)
    phase3_metric = args.phase3_metric or str(config["sweep"].get("phase3_metric", "val_rmse"))
    if phase3_metric not in ("robust_score", "val_rmse"):
        raise SystemExit(f"--phase3-metric must be robust_score or val_rmse, got {phase3_metric}")

    t0 = time.perf_counter()
    with open(artifacts / "families.json", "w", encoding="utf-8") as f:
        json.dump({f["id"]: f for f in config["families"]}, f, indent=2)
    data = load_experiment_data(PROJECT_ROOT, config)
    build_all_tensors(data, config, artifacts)
    sweep_configs = build_sweep_configs(config)
    if args.smoke:
        for cid, cfg in sweep_configs.items():
            cfg["max_epochs"] = 3
            cfg["patience"] = 2
            cfg["checkpoint_every"] = 1
            cfg["data_version"] = -1  # never considered complete/resumable by real runs
        print("[smoke] capped all jobs at 3 epochs (data_version -1)", flush=True)
    with open(artifacts / "sweep_configs.json", "w", encoding="utf-8") as f:
        json.dump(sweep_configs, f, indent=2)

    config_ids_by_family = {f: family_config_ids(config, f) for f in families}
    if args.only:
        only = set(args.only.split(","))
        config_ids_by_family = {f: [c for c in cids if c in only] for f, cids in config_ids_by_family.items()}

    t_start = time.perf_counter()
    all_wall: dict[str, float] = {}

    if not args.phase2_only and not args.phase3_only:
        # ---- phase 1: seed[0] for every (family, config) ----
        p1 = make_jobs(args.out, families, config_ids_by_family, [seeds[0]], config)
        todo = [j for j in p1 if j[3] != "skip"]
        print(f"[phase1] {len(p1)} jobs, {len([j for j in p1 if j[3]=='skip'])} done, "
              f"{len([j for j in p1 if j[3]=='resume'])} resume, {len([j for j in p1 if j[3]=='fresh'])} fresh", flush=True)
        all_wall.update(run_jobs(args.out, todo, n_parallel, config))
        collect_sweep_results(args.out, config, families, all_wall)

    if not args.phase3_only and len(seeds) > 1:
        # ---- phase 2: 2nd seed (seeds[1]) for the top-N honest configs per family ----
        p2 = _phase_jobs(args.out, families, seeds[1], config, phase2_metric, phase2_top_n, "phase2")
        all_wall.update(run_jobs(args.out, [j for j in p2 if j[3] != "skip"], n_parallel, config))
        collect_sweep_results(args.out, config, families, all_wall)

    if len(seeds) > 2:
        # ---- phase 3: 3rd seed (seeds[2]) for the top-M per family -> 3-seed winners ----
        p3 = _phase_jobs(args.out, families, seeds[2], config, phase3_metric, phase3_top_n, "phase3")
        all_wall.update(run_jobs(args.out, [j for j in p3 if j[3] != "skip"], n_parallel, config))
        collect_sweep_results(args.out, config, families, all_wall)

    sweep_wall = time.perf_counter() - t_start
    collect_sweep_results(args.out, config, families, all_wall)
    timing = json.loads((args.out / "timing_log.json").read_text(encoding="utf-8"))
    timing["sweep_wall_s"] = sweep_wall
    with open(args.out / "timing_log.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(f"\n[sweep] all done in {sweep_wall:.1f}s wall (total train_time "
          f"{sum(j.get('train_time_s', 0.0) for j in timing['jobs'].values()):.1f}s)", flush=True)


if __name__ == "__main__":
    main()
