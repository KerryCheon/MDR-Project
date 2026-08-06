#!/usr/bin/env python3
"""Retrain the winning configs on the FULL trainval (train+val) + 5-seed ensembles.

The sweep (run_mlp_sweep.py) trains on `train` (2017-2020) with early stopping
on `val` (2021-2022), and selects winners by robust score (= mean over seeds of
mean(val_rmse, aux2020_rmse)). The XGBoost baselines train on the full
trainval, so this script retrains each family's winner on trainval for its
best-epoch count (standard fixed-epoch retrain; no val early-stopping signal
left). Multiple seeds are averaged into a final ensemble — the neural analog of
XGBoost's 2500-tree ensemble.

Imputer/scaler are re-fit on trainval for the retrain (no leakage: test is
untouched). Outputs go under models/retrain_<family>_<config_id>/seed_<s>/ and
ensemble predictions/metrics into retrain_results.csv.

Usage:
    python run_mlp_retrain.py [--config config.yaml] [--out .] [--top-n 1]
                              [--seeds 42,7,123,2024,999] [--resume] [--families ...]
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
from mlp12.data import build_feature_set  # noqa: E402
from mlp12.trainer import train_one_config  # noqa: E402
from run_mlp_sweep import family_features  # noqa: E402

CLUSTERS = ("0", "1")


def load_job_configs(artifacts: Path) -> dict[str, dict]:
    with open(artifacts / "sweep_configs.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_retrain_tensors(data, config: dict, family_id: str, artifacts: Path):
    """Rebuild per-feature-set tensors with imputer/scaler fit on trainval.

    Returns {suffix: fs}; val arrays are empty (retrain trains fixed epochs).
    """
    target = data.target
    cluster_cfg = config["cluster_config"]
    router = get_router(cluster_cfg["strategy"], data.v0_features, seed=int(config["model"]["seed"]))
    router.fit(data.trainval)
    labels_tv = router.predict(data.trainval)
    labels_te = router.predict(data.test)

    featsets = family_features(family_id, config, data)
    out: dict[str, dict] = {}
    for suffix, feats in featsets.items():
        if suffix == "":
            train_frame = data.trainval
            test_frame = data.test
            test_positions = None
        else:
            cl = suffix.replace("_cluster", "")
            tr_mask = labels_tv == int(cl)
            te_mask = labels_te == int(cl)
            train_frame = data.trainval.loc[tr_mask].reset_index(drop=True)
            test_frame = data.test.loc[te_mask].reset_index(drop=True)
            test_positions = np.where(te_mask)[0]
        empty_val = train_frame.iloc[0:0]
        fs = build_feature_set(train_frame, empty_val, test_frame, feats, target,
                               test_positions=test_positions)
        out[suffix] = fs
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--metric", default="robust_score",
                        help="selection metric for the winner: robust_score (default) | val_rmse")
    parser.add_argument("--configs", default=None,
                        help="explicit retrain set as 'family:id1,id2;family2:id3' "
                             "(overrides --metric/--top-n selection)")
    parser.add_argument("--seeds", default=None, help="comma list (default: config stability_seeds)")
    parser.add_argument("--families", default=None, help="comma list of family ids")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    artifacts = args.out / "artifacts"

    sweep = pd.read_csv(args.out / "sweep_results.csv")
    if args.metric not in ("robust_score", "val_rmse"):
        raise SystemExit(f"--metric must be robust_score or val_rmse, got {args.metric}")
    job_cfgs = load_job_configs(artifacts)
    seeds = [int(s) for s in (args.seeds or ",".join(str(x) for x in config["sweep"]["stability_seeds"])).split(",")]
    families = [f["id"] for f in config["families"]]
    if args.families:
        families = [f for f in families if f in args.families.split(",")]

    test_meta = np.load(artifacts / "test_meta.npz")
    y_test_full = test_meta["y_test"]
    data = load_experiment_data(PROJECT_ROOT, config)
    family_cfgs = {f["id"]: f for f in config["families"]}

    t0 = time.perf_counter()
    rows = []

    # per-family config lists: explicit --configs overrides metric-based selection
    explicit: dict[str, list[str]] = {}
    if args.configs:
        for part in args.configs.split(";"):
            fam, ids = part.split(":")
            explicit[fam] = [c for c in ids.split(",") if c]

    for family in families:
        if args.configs and family in explicit:
            sub = pd.DataFrame({"config_id": explicit[family], "family": [family] * len(explicit[family])})
        else:
            sub = sweep[sweep["family"] == family].sort_values(args.metric, na_position="last").head(args.top_n)
        family_ens_preds: list[np.ndarray] = []
        epochs_used: list[int] = []
        for _, r in sub.iterrows():
            cid = r["config_id"]
            cfg = dict(job_cfgs[cid])
            # best epoch from the robust-selected sweep run (max over seeds)
            sweep_meta = json.loads((args.out / "models" / family / cid / "meta.json").read_text(encoding="utf-8"))
            best_epoch = max(1, int(sweep_meta["best_epoch"]))
            cfg["retrain_epochs"] = best_epoch
            structure = family_cfgs[family]["structure"]
            tensors = build_retrain_tensors(data, config, family, artifacts)

            seed_preds: list[np.ndarray] = []
            for seed in seeds:
                seed_dir = args.out / "models" / "retrain" / f"{family}__{cid}" / f"seed_{seed}"
                meta_path = seed_dir / "meta.json"
                if args.resume and meta_path.exists():
                    m = json.loads(meta_path.read_text(encoding="utf-8"))
                    if m.get("status") == "completed":
                        seed_preds.append(np.load(seed_dir / "preds.npy"))
                        print(f"[skip] retrain {family}/{cid} seed {seed}", flush=True)
                        continue

                seed_dir.mkdir(parents=True, exist_ok=True)
                cfg_s = dict(cfg)
                cfg_s["seed"] = seed
                cfg_s["id"] = f"{cid}_retrain"
                cfg_s["data_version"] = int(config["sweep"].get("data_version", 4))

                full_preds = np.zeros(len(y_test_full), dtype=np.float64)
                per_cluster: dict[str, dict] = {}
                if structure == "global":
                    fs = tensors[""]
                    res = train_one_config(cfg_s, fs, seed_dir, resume=args.resume)
                    full_preds = res.test_preds
                else:
                    for cl in CLUSTERS:
                        fs = tensors[f"_cluster{cl}"]
                        spec_out = seed_dir / f"spec_{cl}"
                        res = train_one_config(cfg_s, fs, spec_out, resume=args.resume)
                        full_preds[fs["test_idx"]] = res.test_preds
                        per_cluster[cl] = {
                            "n_train": int(len(fs["train_idx"])),
                            "n_test": int(len(fs["test_idx"])),
                            "test": compute_metrics(y_test_full[fs["test_idx"]], res.test_preds),
                        }
                np.save(seed_dir / "preds.npy", full_preds)
                tm = compute_metrics(y_test_full, full_preds)
                payload = {
                    "family": family, "config_id": cid, "seed": seed,
                    "mode": "retrain", "retrain_epochs": best_epoch,
                    "structure": structure,
                    "test": tm, "per_cluster": per_cluster,
                    "data_version": cfg_s["data_version"],
                    "status": "completed",
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                print(f"[retrain] {family}/{cid} seed {seed}: test_r2={tm['r2']:.4f} "
                      f"epochs={best_epoch} time={res.train_time_s:.1f}s", flush=True)
                seed_preds.append(full_preds)

            # --- seed ensemble ---
            ens = np.mean(seed_preds, axis=0)
            ens_metrics = compute_metrics(y_test_full, ens)
            ens_dir = args.out / "models" / "retrain" / f"{family}__{cid}"
            np.save(ens_dir / "ens_preds.npy", ens)
            with open(ens_dir / "ens_meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "family": family, "config_id": cid, "seeds": seeds,
                    "retrain_epochs": best_epoch, "test": ens_metrics,
                }, f, indent=2)
            rows.append({
                "family": family, "config_id": cid, "structure": structure,
                "retrain_epochs": best_epoch, "n_seeds": len(seeds),
                "test_r2": ens_metrics["r2"], "test_rmse": ens_metrics["rmse"],
                "test_bias": ens_metrics["bias"], "test_mae": ens_metrics["mae"],
            })
            print(f"[ensemble] {family}/{cid}: {len(seeds)}-seed mean test_r2={ens_metrics['r2']:.4f}", flush=True)
            family_ens_preds.append(ens)
            epochs_used.append(int(best_epoch))

        max_retrain_epochs = max(epochs_used) if epochs_used else 0

        # --- cross-config ensemble (robust to selection noise) ---
        if len(family_ens_preds) > 1:
            ens3 = np.mean(family_ens_preds, axis=0)
            ens3_metrics = compute_metrics(y_test_full, ens3)
            top_dir = args.out / "models" / "retrain" / f"{family}__top{len(family_ens_preds)}"
            top_dir.mkdir(parents=True, exist_ok=True)
            np.save(top_dir / "ens_preds.npy", ens3)
            with open(top_dir / "ens_meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "family": family, "configs": sub["config_id"].tolist(), "seeds": seeds,
                    "test": ens3_metrics,
                }, f, indent=2)
            rows.append({
                "family": family,
                "config_id": "top" + str(len(family_ens_preds)) + "_configs",
                "structure": structure,
                "retrain_epochs": max_retrain_epochs,
                "n_seeds": len(seeds) * len(family_ens_preds),
                "test_r2": ens3_metrics["r2"], "test_rmse": ens3_metrics["rmse"],
                "test_bias": ens3_metrics["bias"], "test_mae": ens3_metrics["mae"],
            })
            print(f"[ensemble] {family}: top-{len(family_ens_preds)}-config seed-ensemble test_r2={ens3_metrics['r2']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "retrain_results.csv", index=False)
    print(f"[retrain] wrote retrain_results.csv ({len(df)} rows) in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
