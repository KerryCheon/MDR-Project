# Jakob Balkovec
# Config Loader

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os
import yaml

from Modeling.Utils.logging import get_logger


def load_config(path: str | Path) -> Dict[str, Any]:
    # pre: path needs to be valid
    # post: config is loaded globally
    # desc: load the config from a YAML file

    path = Path(os.path.expandvars(str(path))).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg: Dict[str, Any]
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    logger = get_logger("config")
    logger.info("Loaded config from %s", path)

    cfg.setdefault("_meta", {})
    cfg["_meta"]["config_path"] = str(path)

    return cfg
