# Jakob Balkovec
# Parse Pipe

# This module defines the ParsePipe class, which parses the data retrieved by the
# RequestPipe into a structured format for further processing.

import pandas as pd
from pathlib import Path
from utils.logger import get_logger
from utils.config import load_config

class ParsePipe:
    FILE_GLOB = "uscrn_*.txt"
    NA_VALUE = -9999.0
    DATE_FORMAT = "%Y%m%d"
    DELIM = r"\s+"

    def __init__(self, config=None):
        # pre:  config is a dictionary loaded from config.yaml or None
        # post: initializes ParsePipe with configuration parameters
        # desc: Sets up parsing behavior, column indices, and output configuration.

        self.config = config or load_config()
        parse_cfg = self.config

        self.in_dir = Path(parse_cfg.get("in_dir", "data/raw"))
        self.out_dir = Path(parse_cfg.get("out_dir", "data/processed"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.col_indices = parse_cfg.get("col_indices", {
            "station_id": 0,
            "date": 1,
            "longitude": 2,
            "latitude": 3,
            "soil_moisture_5cm": 17,
        })

        self.drop_duplicates = parse_cfg.get("drop_duplicates", True)
        self.station_name = self.config.get("request", {}).get("station", "unknown_station")
        self.logger = get_logger().getChild(f"parse.{self.station_name}")

    def run(self, _=None):
        # pre:  raw yearly files exist in in_dir and follow CRND0103 format
        # post: returns unified DataFrame of all parsed files
        # desc: Reads all downloaded USCRN files, maps columns, replaces missing values,
        #       and concatenates them into a clean unified DataFrame.

        parsed_dfs = []
        files = sorted(self.in_dir.glob(self.FILE_GLOB))

        if not files:
            self.logger.warning(f"No USCRN files found in {self.in_dir}")
            return pd.DataFrame()

        self.logger.info(f"[{self.station_name}] Found {len(files)} USCRN files — parsing all years.")

        for file_path in files:
            try:
                df = pd.read_csv(
                    file_path,
                    sep=self.DELIM,
                    comment="#",
                    header=None,
                    engine="python",
                )

                # rename known indices for clarity
                rename_map = {idx: name for name, idx in self.col_indices.items() if idx in df.columns}
                df = df.rename(columns=rename_map)

                # ensure all expected columns are present
                df.columns = [
                    c if isinstance(c, str) else rename_map.get(c, f"col_{c}")
                    for c in df.columns
                ]

                # safely convert date
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], format=self.DATE_FORMAT, errors="coerce")

                # replace -9999 placeholders with NaN for all numeric columns
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].replace(self.NA_VALUE, pd.NA)

                # tag file source
                df["source_file"] = file_path.name
                parsed_dfs.append(df)

                self.logger.debug(f"[{self.station_name}] Parsed {file_path.name}: {len(df)} rows, {len(df.columns)} cols")

            except Exception as e:
                self.logger.error(f"[{self.station_name}] Failed to parse {file_path.name}: {e}")

        if not parsed_dfs:
            self.logger.warning(f"[{self.station_name}] No valid data parsed from any files.")
            return pd.DataFrame()

        combined_df = pd.concat(parsed_dfs, ignore_index=True)
        self.logger.info(f"[{self.station_name}] Combined {len(parsed_dfs)} files into {len(combined_df)} total rows.")

        # [optional] deduplication
        if self.drop_duplicates and {"station_id", "date"} <= set(combined_df.columns):
            before = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=["station_id", "date"])
            removed = before - len(combined_df)
            self.logger.info(f"[{self.station_name}] Removed {removed} duplicate rows.")

        return combined_df
