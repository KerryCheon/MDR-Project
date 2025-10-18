# Jakob Balkovec
# Save Pipe

# This modules defines the SavePipe class, which is responsible for saving
# the final processed DataFrame to disk in a specified format.

from pathlib import Path
from utils.logger import get_logger
from utils.config import load_config


class SavePipe:
    SUPPORTED_FORMATS = {"csv", "parquet", "json", "pkl", "excel"}
    OUT_PATH = r"data/processed/final.csv"

    def __init__(self, config=None):
        # pre:  config is a dictionary loaded from config.yaml or None
        # post: initializes SavePipe with output directory and file format
        # desc: Loads save parameters from config and ensures output directory exists.

        self.config = config or load_config()
        save_cfg = self.config["pipeline"]["save"]

        self.out_path = Path(save_cfg.get("out_path", self.OUT_PATH))
        self.format = save_cfg.get("format", "csv").lower()
        self.index = save_cfg.get("index", False)
        self.logger = get_logger().getChild("save")

        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"unsupported save format: {self.format}")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, df):
        # pre:  cleaned and merged DataFrame provided
        # post: saves DataFrame to disk in configured format
        # desc: Writes processed data to disk (CSV, Parquet, or JSON).
        if df is None or df.empty:
            self.logger.warning("[SKIPPING] SavePipe -- No data to save :(")
            return None

        self.logger.info(f"Saving DataFrame ({len(df)} rows) to {self.out_path} as {self.format.upper()}.")

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
            msg = f"Unsupported save format encountered: {self.format}"
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info(f"SavePipe complete — wrote {self.out_path}")
        return self.out_path
