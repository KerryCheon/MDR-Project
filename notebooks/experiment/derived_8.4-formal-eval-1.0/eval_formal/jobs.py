"""Shared parallel-job machinery for derived_8.4-formal-eval-1.0.

Adapted from derived_8.4-eval-1.3/run_loso.py (eval-2.0 worker format): the driver
pins configurations, writes artifacts/runtime.json, and spawns n_parallel worker
subprocesses — each trains one (config, seed[, station]) job on the GPU — then
aggregates per-job meta.json files. Completed jobs are resumed/skipped via
meta.json (status + data_version match) plus prediction/weight file presence, so a
partially-written seed never counts as done. XGBoost GPU folds serialize on one
H100, so workers buy resilience/resume, not aggregate throughput.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
EXP_DIR = ARTIFACTS_DIR.parent

SMOKE = False


def set_smoke(smoke: bool) -> None:
    global SMOKE
    SMOKE = bool(smoke)


def expected_version(config: dict, section: str) -> int:
    return -1 if SMOKE else int(config[section].get("data_version", 1))


def runtime_path() -> Path:
    return ARTIFACTS_DIR / "runtime.json"


def write_runtime(config: dict, device: str | None, n_estimators: int | None = None,
                  version: int | None = None) -> None:
    """Per-run worker overrides: data_version (resume invalidation), device, n_estimators."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_version": -1 if SMOKE else (
            version if version is not None else int(config["temporal"].get("data_version", 1))),
        "device": device or str(config["model"]["exact_params"].get("device", "cuda")),
    }
    if n_estimators is not None:
        payload["n_estimators"] = int(n_estimators)
    with open(runtime_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def job_dir(config_id: str, seed: int, station: str) -> Path:
    return ARTIFACTS_DIR / "jobs" / f"{config_id}__s{seed}__{station}"


def load_job_meta(config_id: str, seed: int, station: str) -> dict | None:
    meta_path = job_dir(config_id, seed, station) / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _booster_present(models_dir: Path, stem: str) -> bool:
    if not models_dir.exists():
        return False
    return (models_dir / f"{stem}_reg.json").exists() or \
        (models_dir / f"{stem}_spec_0.json").exists()


def job_complete(config_id: str, seed: int, station: str, version: int, *,
                 predictions_dir: Path, models_dir: Path) -> bool:
    meta = load_job_meta(config_id, seed, station)
    if meta is None:
        return False
    if meta.get("status") != "completed" or meta.get("data_version") != version:
        return False
    stem = f"{config_id}__s{seed}__{station}"
    if not (predictions_dir / f"{stem}_preds.npy").exists():
        return False
    if not _booster_present(models_dir, stem):
        return False
    return True


def make_jobs(config_ids: list[str], seeds: list[int], stations: list[str], config: dict,
              section: str, predictions_dir: Path, models_dir: Path, *,
              no_resume: bool = False) -> list[tuple[str, int, str, str]]:
    """Jobs = (config_id, seed, station, mode) with mode in {skip, fresh}."""
    version = expected_version(config, section)
    jobs: list[tuple[str, int, str, str]] = []
    for config_id in config_ids:
        for seed in seeds:
            for station in stations:
                if (not no_resume and job_complete(config_id, seed, station, version,
                                                   predictions_dir=predictions_dir,
                                                   models_dir=models_dir)):
                    mode = "skip"
                else:
                    mode = "fresh"
                jobs.append((config_id, seed, station, mode))
    return jobs


def run_jobs(jobs: list[tuple[str, int, str, str]], n_parallel: int) -> None:
    """Spawn up to n_parallel worker subprocesses; each trains one job."""
    (ARTIFACTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    queue = list(jobs)
    active: list[tuple[subprocess.Popen, tuple, object, float]] = []
    while queue or active:
        while len(active) < n_parallel and queue:
            config_id, seed, station, mode = queue.pop(0)
            log_path = ARTIFACTS_DIR / "logs" / f"{config_id}__s{seed}__{station}.log"
            cmd = [
                sys.executable, str(EXP_DIR / "run_worker.py"),
                "--config-id", config_id, "--seed", str(seed), "--station", station,
                "--artifacts", str(ARTIFACTS_DIR), "--out", str(EXP_DIR),
            ]
            logf = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(EXP_DIR))
            active.append((proc, (config_id, seed, station, mode), logf, time.perf_counter()))
            print(f"[launch] {config_id} s{seed} @ {station} ({mode})", flush=True)

        still_active = []
        for proc, job, logf, started in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((proc, job, logf, started))
                continue
            logf.close()
            config_id, seed, station, mode = job
            ok = load_job_meta(config_id, seed, station) is not None and rc == 0
            status = "ok" if ok else f"FAILED(rc={rc})"
            print(f"[finish] {config_id} s{seed} @ {station} {status} "
                  f"wall={time.perf_counter() - started:.1f}s", flush=True)
            if not ok:
                print(f"         log: {ARTIFACTS_DIR / 'logs' / f'{config_id}__s{seed}__{station}.log'}",
                      flush=True)
        active = still_active
        if active:
            time.sleep(2.0)
