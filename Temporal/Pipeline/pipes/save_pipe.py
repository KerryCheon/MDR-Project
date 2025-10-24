# Jakob Balkovec
# Save Pipe

# This module defines the SavePipe class, which is responsible for saving
# the final processed DataFrame to disk in a specified format.

from pathlib import Path
from utils.logger import get_logger
from utils.config import load_config


class SavePipe:
    SUPPORTED_FORMATS = {"csv", "parquet", "json", "pkl", "excel"}
    OUT_PATH = r"data/processed/final.csv"

    def __init__(self, config=None, station_name=None):
        # pre:  config is a dictionary loaded from config.yaml or None
        # post: initializes SavePipe with output directory and file format
        # desc: Loads save parameters from config and ensures output directory exists.

        self.config = config or load_config()
        self.station_name = station_name or "unknown_station"
        save_cfg = self.config

        self.out_path = Path(save_cfg.get("out_path", self.OUT_PATH))
        self.format = save_cfg.get("format", "csv").lower()
        self.index = save_cfg.get("index", False)

        # station-scoped logger for clean multi-station output
        self.logger = get_logger().getChild(f"save.{self.station_name}")

        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported save format: {self.format}")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, df):
        # pre:  cleaned and merged DataFrame provided
        # post: saves DataFrame to disk in configured format
        # desc: Writes processed data to disk (CSV, Parquet, or JSON).
        if df is None or df.empty:
            self.logger.warning(f"[{self.station_name}] Skipping SavePipe — no data to save.")
            return None

        self.logger.info(
            f"[{self.station_name}] Saving DataFrame ({len(df)} rows) to {self.out_path} as {self.format.upper()}."
        )

        if self.format == "csv":
            df.to_csv(self.out_path, index=self.index)
        elif self.format == "parquet":
            df.to_parquet(self.out_path, index=self.index)
        elif self.format == "json":
            df.to_json(self.out_path, orient="records", indent=2)
        elif self.format == "pkl":
            df.to_pickle(self.out_path)
        elif self.format == "excel":
            df.to_excel(self.out_path, index=self.index)
        else:
            msg = f"[{self.station_name}] Unsupported save format encountered: {self.format}"
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info(f"[{self.station_name}] SavePipe complete — wrote {self.out_path.resolve()}")
        return self.out_path
