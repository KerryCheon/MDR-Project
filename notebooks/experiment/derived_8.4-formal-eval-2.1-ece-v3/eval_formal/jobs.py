"""Shared parallel-job machinery for derived_8.4-formal-eval-2.1-ece-v3.

Driver pins configurations, writes artifacts/runtime.json, and spawns n_parallel worker
subprocesses — each trains or evaluates one (config, seed, target_tag) job — then
aggregates per-job meta.json files. Completed jobs are resumed/skipped via
meta.json (status + data_version match) plus prediction file presence.
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


def job_dir(config_id: str, seed: int, target_tag: str) -> Path:
    return ARTIFACTS_DIR / "jobs" / f"{config_id}__s{seed}__{target_tag}"


def load_job_meta(config_id: str, seed: int, target_tag: str) -> dict | None:
    meta_path = job_dir(config_id, seed, target_tag) / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def job_complete(config_id: str, seed: int, target_tag: str, version: int, *,
                 predictions_dir: Path, models_dir: Path) -> bool:
    meta = load_job_meta(config_id, seed, target_tag)
    if meta is None:
        return False
    if meta.get("status") != "completed" or meta.get("data_version") != version:
        return False
    stem = f"{config_id}__s{seed}__{target_tag}"
    if not (predictions_dir / f"{stem}_preds.npy").exists():
        return False
    return True


def make_jobs(config_ids: list[str], seeds: list[int], targets: list[str], config: dict,
              section: str, predictions_dir: Path, models_dir: Path, *,
              no_resume: bool = False) -> list[tuple[str, int, str, str]]:
    """Jobs = (config_id, seed, target_tag, mode) with mode in {skip, fresh}."""
    version = expected_version(config, section)
    jobs: list[tuple[str, int, str, str]] = []
    for config_id in config_ids:
        for seed in seeds:
            for target in targets:
                if (not no_resume and job_complete(config_id, seed, target, version,
                                                   predictions_dir=predictions_dir,
                                                   models_dir=models_dir)):
                    mode = "skip"
                else:
                    mode = "fresh"
                jobs.append((config_id, seed, target, mode))
    return jobs


def run_jobs(jobs: list[tuple[str, int, str, str]], n_parallel: int) -> None:
    """Spawn up to n_parallel worker subprocesses; each runs one job.

    Raises RuntimeError if any worker exits nonzero or leaves no meta.json,
    so callers never silently aggregate a partial seed grid.
    """
    (ARTIFACTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    queue = list(jobs)
    active: list[tuple[subprocess.Popen, tuple, object, float]] = []
    failures: list[str] = []
    while queue or active:
        while len(active) < n_parallel and queue:
            config_id, seed, target, mode = queue.pop(0)
            log_path = ARTIFACTS_DIR / "logs" / f"{config_id}__s{seed}__{target}.log"
            cmd = [
                sys.executable, str(EXP_DIR / "run_worker.py"),
                "--config-id", config_id, "--seed", str(seed), "--target", target,
                "--artifacts", str(ARTIFACTS_DIR), "--out", str(EXP_DIR),
            ]
            logf = open(log_path, "w", encoding="utf-8")
            try:
                proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(EXP_DIR))
            except Exception:
                logf.close()
                raise
            active.append((proc, (config_id, seed, target, mode), logf, time.perf_counter()))
            print(f"[launch] {config_id} s{seed} @ {target} ({mode})", flush=True)

        still_active = []
        for proc, job, logf, started in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((proc, job, logf, started))
                continue
            try:
                config_id, seed, target, mode = job
                ok = load_job_meta(config_id, seed, target) is not None and rc == 0
                status = "ok" if ok else f"FAILED(rc={rc})"
                print(f"[finish] {config_id} s{seed} @ {target} {status} "
                      f"wall={time.perf_counter() - started:.1f}s", flush=True)
                if not ok:
                    print(f"         log: {ARTIFACTS_DIR / 'logs' / f'{config_id}__s{seed}__{target}.log'}",
                          flush=True)
                    failures.append(f"{config_id}__s{seed}__{target} (rc={rc})")
            finally:
                logf.close()
        active = still_active
        if active:
            time.sleep(1.0)
    if failures:
        raise RuntimeError(
            f"{len(failures)} worker(s) failed: {failures[:10]}"
            f"{' ...' if len(failures) > 10 else ''}. "
            "See artifacts/logs/*.log. Aggregation aborted to avoid "
            "silent partial-seed results."
        )
