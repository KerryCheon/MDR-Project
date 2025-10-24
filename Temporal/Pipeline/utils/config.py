# Jakob Balkovec
# Config Loader

# This module defines the conifg loader utility to load configuration settings
# from a YAML file.

import yaml
from pathlib import Path
from tqdm import tqdm
import time

from utils.logger import get_logger

CONFIG_FILE = (Path(__file__).resolve().parent.parent / "config.yaml").resolve()


def load_config(path=CONFIG_FILE):
    # pre: path is valid
    # post: returns dict of config settings
    # desc: loads configuration from a YAML file with a visible tqdm progress indicator.

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    logger = get_logger()
    logger.info(f"Loading configuration from {path}...")

    # Simulate a quick, clean progress bar for visibility
    with tqdm(total=3, desc="Loading config", ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        pbar.update(1)
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        time.sleep(0.2)  # small delay for visual feedback
        pbar.update(2)

    if logger.handlers:
        logger.info(f"Configuration successfully loaded from {path}")

    return config

# [TEST]
if __name__ == "__main__":
    config = load_config()
    print(yaml.dump(config, sort_keys=False))
