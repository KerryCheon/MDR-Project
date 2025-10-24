# Jakob Balkovec
# Merge Pipe

# This module defines the MergePipe class, which is responsible for merging
# multiple DataFrames or datasets into a unified DataFrame.

import pandas as pd
from utils.logger import get_logger
from utils.config import load_config

class MergePipe:
    def __init__(self, config=None):
        # pre:  config is a dictionary loaded from config.yaml or None
        # post: initializes MergePipe with merging strategy parameters
        # desc: Sets up merging behavior for multiple DataFrames or datasets.

        self.config = config or load_config()
        merge_cfg = self.config["merge"]

        self.on_columns = merge_cfg.get("on_columns", [])
        self.how = merge_cfg.get("how", "outer")
        self.logger = get_logger().getChild("merge")

    def run(self, data):
        # pre:  receives list of DataFrames to merge (or one DataFrame)
        # post: returns unified DataFrame based on merge config
        # desc: Merges multiple data sources according to configured join strategy.

        if isinstance(data, list) and len(data) > 1:
            self.logger.info(f"Merging {len(data)} DataFrames using '{self.how}' join.")
            merged = pd.concat(data, ignore_index=True)
        elif isinstance(data, list):
            self.logger.info("Received a single DataFrame in list — no merge needed.")
            merged = data[0]
        else:
            self.logger.info("Received single DataFrame — returning unchanged.")
            merged = data

        # Drop duplicates based on merge keys
        if self.on_columns:
            before = len(merged)
            merged = merged.drop_duplicates(subset=self.on_columns)
            after = len(merged)
            removed = before - after
            self.logger.debug(f"Deduplicated on {self.on_columns} ({removed} rows removed).")

        self.logger.info(f"MergePipe complete — {len(merged)} rows total.")
        return merged
