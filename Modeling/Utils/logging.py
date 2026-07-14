# Jakob Balkovec
# Logger

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict

_ROOT_LOGGER = "soilmoist_feat"
_INITIALIZED = False


def setup_logger(config: Dict[str, Any], run_dir: str | Path | None = None) -> logging.Logger:
    # pre: config is a dict, run_dir is a valid path or None
    # post: instantiates a logger globally
    # desc: sets up logging; other modules should use get_logger() or .getChild()

    global _INITIALIZED

    root = logging.getLogger(_ROOT_LOGGER)
    if _INITIALIZED:
        return root

    log_cfg = (config or {}).get("logging", {})
    level = str(log_cfg.get("level", "INFO")).upper()

    root.setLevel(level)
    root.propagate = False

    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root.handlers.clear()

    if log_cfg.get("console", True):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    if run_dir is not None and log_cfg.get("log_to_file", True):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir = run_dir / str(log_cfg.get("runs_subdir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / str(log_cfg.get("file_name", "pipeline.log"))
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=int(log_cfg.get("max_bytes", 10_485_760)),
            backupCount=int(log_cfg.get("backup_count", 5)),
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    _INITIALIZED = True
    root.info("Logger initialized (level=%s)", level)
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    # desc: follows the singleton pattern, @see some web forum about the "Singleton Design Pattern"

    root = logging.getLogger(_ROOT_LOGGER)
    if name is None:
        return root
    return root.getChild(name)
