# Jakob Balkovec
# Logger

# This module defines a simple logger utility to log messages with different
# severity levels (INFO, WARNING, ERROR).

import logging
from pathlib import Path

_LOGGER_NAME = "pipeline"

def setup_logger(config):
    # pre: config: dict (YAML loaded configuration)
    # post: logging.Logger
    # desc: Sets up the logger based on the provided configuration.

    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    log_to_file = log_cfg.get("log_to_file", True)
    log_file = Path(log_cfg.get("file_path", "logs/pipeline.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(_LOGGER_NAME)

    # prevent duplicate handlers if setup_logger() is called multiple times
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_to_file:
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False
        logger.info("Logger initialized.")

    return logger

def get_logger():
    # pre: None
    # post: logging.Logger
    # desc: Retrieves the logger instance, this is here so that other modules
    #       can easily get the logger without needing to pass it around.
    # *** [SINGLETON PATTERN] ***

    return logging.getLogger(_LOGGER_NAME)
