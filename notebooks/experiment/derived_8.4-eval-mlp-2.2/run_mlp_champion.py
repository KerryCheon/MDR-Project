#!/usr/bin/env python3
"""Champion step for derived_8.4-eval-mlp-2.2: 5-seed ensembles of the winners.

After the 3-phase sweep, the val-selected winner config per family (top-N by
3-seed mean val RMSE among mlp/fg/plr) is trained on the remaining stability
seeds {2024, 999} (seeds 42, 7, 123 already exist from the sweep) and the
5-seed seed-mean test predictions are aggregated into:

    models/champion/<family>__<config_id>/{ens_preds.npy, ens_meta.json}

NEW in 2.2: `--top-n` may be an int (2.1 parity) or the per-family dict
`sweep.champion_top_n` from config.yaml — this fixes 2.1's documented
limitation ("top-2-mixed not expressible" because the CLI applied uniformly).

No trainval retrain (documented negative in mlp-1.2) — the ensemble is a pure
seed average of train-only models, exactly the neural analog of the XGBoost
tree ensemble. The sweep's own per-config aggregation (models/<family>/<cid>/
meta.json, mean over completed sweep seeds) is left untouched.

Usage:
    python run_mlp_champion.py [--config config.yaml] [--out .] [--top-n 1]
                               [--n-parallel 8] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
EVAL11_DIR = EXP_DIR.parent / "derived_8.4-eval-1.1"
sys.path.insert(0, str(EVAL11_DIR))
sys.path.insert(0, str(EXP_DIR))

from eval11.data import load_experiment_data  # noqa: E402
from eval11.evaluator import compute_metrics  # noqa: E402
from run_mlp_sweep import (  # noqa: E402
    build_all_tensors,
    build_sweep_configs,
    current_data_version,
    expected_version,
    job_dir,
    load_cluster_deltas,
    seed_complete,
    seed_resumable,
    _clean_job_dir,
)

HONEST_ARCHS = ("mlp", "fg", "plr")


def top_configs_per_family(out: Path, families: list[str], top_n) -> dict[str, list[str]]:
    """Top-N val-selected configs per family; `top_n` is an int or {family: int}."""
    import pandas as pd

    sweep = pd.read_csv(out / "sweep_results.csv")
    out_map: dict[str, list[str]] = {}
    for fam in families:
        n = top_n.get(fam, 1) if isinstance(top_n, dict) else int(top_n)
        sub = sweep[(sweep["family"] == fam) & sweep["architecture"].isin(HONEST_ARCHS)] \
            .dropna(subset=["val_rmse"]).sort_values("val_rmse")
        out_map[fam] = sub.head(n)["config_id"].tolist()
    return out_map


def make_champion_jobs(out: Path, config: dict, fam_cids: dict[str, list[str]],
                       seeds: list[int]) -> list[tuple[str, str, int, str]]:
    version = expected_version(config)
    jobs = []
    for fam, cids in fam_cids.items():
        for cid in cids:
            for s in seeds:
                jdir = job_dir(out, fam, cid, s)
                if seed_complete(jdir, version):
                    jobs.append((fam, cid, s, "skip"))
                elif seed_resumable(jdir, version):
                    jobs.append((fam, cid, s, "resume"))
                else:
                    if jdir.exists():
                        print(f"[invalidate] {fam}/{cid}/seed_{s}: stale artifacts "
                              f"(data_version < {version})", flush=True)
                        _clean_job_dir(jdir)
                    jobs.append((fam, cid, s, "fresh"))
    return jobs


def run_champion_jobs(out: Path, jobs, n_parallel: int, config: dict) -> None:
    artifacts = out / "artifacts"
    queue = list(jobs)
    active: list[tuple[subprocess.Popen, object, object, float]] = []

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
            print(f"[champion launch] {family}/{cid}/seed_{seed} ({mode})", flush=True)

        still_active = []
        for proc, job, logf, started in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((proc, job, logf, started))
                continue
            logf.close()
            family, cid, seed, mode = job
            ok = seed_complete(job_dir(out, family, cid, seed), expected_version(config))
            status = "ok" if (ok and rc == 0) else f"FAILED(rc={rc})"
            print(f"[champion finish] {family}/{cid}/seed_{seed} {status} wall={time.perf_counter() - started:.1f}s",
                  flush=True)
        active = still_active
        if active:
            time.sleep(2.0)


def aggregate_champion(out: Path, config: dict, fam_cids: dict[str, list[str]], seeds: list[int]) -> None:
    version = expected_version(config)
    test_meta = np.load(out / "artifacts" / "test_meta.npz")
    champ_root = out / "models" / "champion"
    champ_root.mkdir(parents=True, exist_ok=True)
    for fam, cids in fam_cids.items():
        for cid in cids:
            preds_list = []
            for s in seeds:
                p = job_dir(out, fam, cid, s) / "preds.npy"
                m = job_dir(out, fam, cid, s) / "meta.json"
                if not (p.exists() and m.exists()):
                    continue
                meta = json.loads(m.read_text(encoding="utf-8"))
                if meta.get("status") != "completed" or meta.get("config", {}).get("data_version") != version:
                    continue
                preds_list.append(np.load(p))
            if not preds_list:
                print(f"[champion] WARNING: no completed seeds for {fam}/{cid}", flush=True)
                continue
            ens_preds = np.mean(preds_list, axis=0)
            test = compute_metrics(test_meta["y_test"], ens_preds)
            ens_dir = champ_root / f"{fam}__{cid}"
            ens_dir.mkdir(parents=True, exist_ok=True)
            np.save(ens_dir / "ens_preds.npy", ens_preds)
            (ens_dir / "ens_meta.json").write_text(json.dumps({
                "family": fam,
                "config_id": cid,
                "n_seeds": len(preds_list),
                "seeds": [int(s) for s in seeds],
                "test": test,
                "data_version": version,
            }, indent=2))
            print(f"[champion] {fam}/{cid}: {len(preds_list)}-seed ensemble test_r2={test['r2']:.4f} "
                  f"bias={test['bias']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.yaml")
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    parser.add_argument("--top-n", type=int, default=None,
                        help="top-N configs per family (uniform); default: "
                             "sweep.champion_top_n from config.yaml (int or {family: n})")
    parser.add_argument("--n-parallel", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from run_mlp_sweep import SMOKE_VERSION, SMOKE  # noqa: F401  (module-level flag)

    if args.smoke:
        import run_mlp_sweep as rms

        rms.SMOKE = True
        for cid, cfg in build_sweep_configs(config).items():
            cfg["max_epochs"] = 3
            cfg["patience"] = 2
            cfg["checkpoint_every"] = 1
            cfg["data_version"] = -1

    artifacts = args.out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "logs").mkdir(parents=True, exist_ok=True)

    n_parallel = args.n_parallel or int(config["sweep"]["n_parallel"])
    families = [f["id"] for f in config["families"]]
    # NEW in 2.2: per-family champion depth from config (int or {family: n});
    # the CLI --top-n overrides (2.1 parity).
    top_n = args.top_n if args.top_n is not None else config["sweep"].get("champion_top_n", 1)
    fam_cids = top_configs_per_family(args.out, families, top_n)
    if not any(fam_cids.values()):
        print("[champion] no sweep results found; run the sweep first", flush=True)
        return

    # tensor artifacts must exist for the worker; rebuild idempotently
    data = load_experiment_data(PROJECT_ROOT, config)
    build_all_tensors(data, config, artifacts)
    sweep_configs = build_sweep_configs(config)
    if args.smoke:
        for cfg in sweep_configs.values():
            cfg["max_epochs"] = 3
            cfg["patience"] = 2
            cfg["checkpoint_every"] = 1
            cfg["data_version"] = -1
    with open(artifacts / "sweep_configs.json", "w", encoding="utf-8") as f:
        json.dump(sweep_configs, f, indent=2)

    seeds = [int(s) for s in config["sweep"].get("stability_seeds", [42, 7, 123, 2024, 999])]
    jobs = make_champion_jobs(args.out, config, fam_cids, seeds)
    todo = [j for j in jobs if j[3] != "skip"]
    print(f"[champion] {len(jobs)} jobs ({len([j for j in jobs if j[3]=='skip'])} done, "
          f"{len([j for j in jobs if j[3]=='resume'])} resume, {len([j for j in jobs if j[3]=='fresh'])} fresh)",
          flush=True)
    if todo:
        run_champion_jobs(args.out, todo, n_parallel, config)
    aggregate_champion(args.out, config, fam_cids, seeds)
    print("[champion] done", flush=True)


if __name__ == "__main__":
    main()
