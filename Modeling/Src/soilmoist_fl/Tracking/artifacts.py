# Jakob Balkovec
# Artifacts

import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from Modeling.Utils.logging import get_logger


def ensure_run_dir(base_runs_dir, run_id=None):
    log = get_logger("tracking.artifacts")

    base = Path(base_runs_dir)
    base.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    run_dir = base / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "stage_features").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    log.info("ensure_run_dir: %s", run_dir)
    return run_dir, str(run_id)


def save_json(path, obj):
    log = get_logger("tracking.artifacts")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    log.info("save_json: %s", path)
    return str(path)


def save_text(path, text):
    log = get_logger("tracking.artifacts")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(str(text), encoding="utf-8")
    log.info("save_text: %s", path)
    return str(path)


def save_metrics_csv(path, metric_rows):
    log = get_logger("tracking.artifacts")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metric_rows)
    df.to_csv(path, index=False)

    log.info("save_metrics_csv: %s rows=%d", path, len(df))
    return str(path)


def save_stage_features(run_dir, stage_name, selected, ranked=None, scores=None, extra=None):
    log = get_logger("tracking.artifacts")

    run_dir = Path(run_dir)
    out = {
        "stage": stage_name,
        "selected": list(selected) if selected is not None else [],
        "ranked": list(ranked) if ranked is not None else None,
        "scores": scores if scores is not None else None,
        "extra": extra if extra is not None else None,
    }

    path = run_dir / "stage_features" / f"{stage_name}.json"
    save_json(path, out)
    log.info("save_stage_features: stage=%s selected=%d", stage_name, len(out["selected"]))
    return str(path)


def save_selected_features(run_dir, selected_features):
    run_dir = Path(run_dir)
    path = run_dir / "selected_features.json"
    return save_json(path, list(selected_features) if selected_features is not None else [])
