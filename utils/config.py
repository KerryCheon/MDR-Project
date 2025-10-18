# Jakob Balkovec
# Config Loader

# This module defines the conifg loader utility to load configuration settings
# from a YAML file.

import yaml
from pathlib import Path

from utils.logger import get_logger

CONFIG_FILE = Path("config.yaml").resolve()

def load_config(path=CONFIG_FILE):
  # pre: path is valid
  # post: returns dict of config settings
  # desc: loads configuration from a YAML file.

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger = get_logger()
    if logger.handlers:  # only log if logger is set up
        logger.debug(f"Loaded configuration from {path}")

    return config

# [TEST]
if __name__ == "__main__":
    config = load_config()
    print(config)
