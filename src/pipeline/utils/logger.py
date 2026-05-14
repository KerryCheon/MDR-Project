# Jakob Balkovec
# Logger

# This module defines a simple logger utility to log messages with different
# severity levels (INFO, WARNING, ERROR).

import logging
import logging.handlers
from pathlib import Path

_ROOT_LOGGER = "pipeline"
_INITIALIZED = False

def setup_logger(config):
    global _INITIALIZED

    if _INITIALIZED:
        return logging.getLogger(_ROOT_LOGGER)

    log_cfg = config.get("logging", {})

    root = logging.getLogger(_ROOT_LOGGER)
    root.setLevel(log_cfg.get("level", "INFO").upper())

    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)

    if log_cfg.get("log_to_file", True):
        pipeline_file = Path(log_cfg.get("file_path", "logs/pipeline.log"))
        pipeline_file.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.handlers.RotatingFileHandler(
            pipeline_file,
            maxBytes=log_cfg.get("max_bytes", 10_485_760),
            backupCount=log_cfg.get("backup_count", 5)
        )
        fh.setFormatter(formatter)
        root.addHandler(fh)

    imputer_log = logging.getLogger(f"{_ROOT_LOGGER}.imputer")
    imputer_log.setLevel(log_cfg.get("imputer_level", "DEBUG").upper())
    imputer_log.propagate = False  # prevent sending logs to pipeline.log

    imputer_file = Path(log_cfg.get("imputer_file_path", "logs/imputer.log"))
    imputer_file.parent.mkdir(parents=True, exist_ok=True)
    fh_imp = logging.FileHandler(imputer_file, mode="a")
    fh_imp.setFormatter(formatter)
    imputer_log.addHandler(fh_imp)

    val_log = logging.getLogger(f"{_ROOT_LOGGER}.validator")
    val_log.setLevel(log_cfg.get("validator_level", "INFO").upper())
    val_log.propagate = False

    val_file = Path(log_cfg.get("validator_file_path", "logs/validator.log"))
    val_file.parent.mkdir(parents=True, exist_ok=True)

    fh_val = logging.FileHandler(val_file, mode="a", encoding="utf-8")
    fh_val.setFormatter(formatter)
    val_log.addHandler(fh_val)

    _INITIALIZED = True
    root.info("Logger initialized.")
    return root

def get_logger(name=None):
    # pre: None
    # post: logging.Logger
    # desc: Retrieves the logger instance, this is here so that other modules
    #       can easily get the logger without needing to pass it around.
    # *** [SINGLETON PATTERN] ***

    if name is None:
        return logging.getLogger(_ROOT_LOGGER)
    return logging.getLogger(f"{_ROOT_LOGGER}.{name}")
