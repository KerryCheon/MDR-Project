# Jakob Balkovec
# Run Registry

import json
from pathlib import Path
from datetime import datetime

from Modeling.Utils.logging import get_logger


REGISTRY_FILE = "registry.json"


def _load_registry(path):
    path = Path(path)
    if not path.exists():
        return {"runs": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"runs": []}


def _save_registry(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def register_run(base_runs_dir, run_id, meta=None):
    log = get_logger("tracking.registry")

    base = Path(base_runs_dir)
    base.mkdir(parents=True, exist_ok=True)

    reg_path = base / REGISTRY_FILE
    reg = _load_registry(reg_path)

    entry = {
        "run_id": str(run_id),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "path": str((base / str(run_id)).resolve()),
        "meta": meta or {},
    }

    # Replace if run_id already exists
    runs = reg.get("runs", [])
    runs = [r for r in runs if r.get("run_id") != str(run_id)]
    runs.append(entry)
    reg["runs"] = runs

    _save_registry(reg_path, reg)
    log.info("register_run: %s", entry["run_id"])
    return str(reg_path)


def list_runs(base_runs_dir):
    base = Path(base_runs_dir)
    reg_path = base / REGISTRY_FILE
    reg = _load_registry(reg_path)
    return reg.get("runs", [])
