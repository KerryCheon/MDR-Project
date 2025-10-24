# Jakob Balkovec
# Request Pipe

# This module defines the RequestPipe class, which handles
# HTTP requests to fetch data from the NOAA USCRN dataset.

import requests
from pathlib import Path
from utils.logger import get_logger
from utils.config import load_config


class RequestPipe:
    FILE_PREFIX = "CRND0103"
    FILE_SUFFIX = ".txt"
    OUTPUT_PREFIX = "uscrn_"

    def __init__(self, config=None):
        # pre:  config is a dictionary loaded from config.yaml or None
        # post: initializes the RequestPipe with configuration settings
        # desc: reads request parameters from the config file and sets up logging/output paths.

        self.config = config or load_config()
        req_cfg = self.config["request"]

        # Core parameters
        self.base_url = req_cfg["base_url"]
        self.station = req_cfg["station"]
        self.start_year = req_cfg["start_year"]
        self.end_year = req_cfg["end_year"]
        self.timeout = req_cfg.get("timeout", 20)
        self.min_bytes = req_cfg.get("min_bytes", 500)

        # Output directory
        self.out_dir = Path(req_cfg.get("out_dir", f"data/{self.station}/raw"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Logger (namespaced per station)
        self.logger = get_logger().getChild(f"request.{self.station}")

    def run(self, _=None):
        # pre:  configuration loaded and output directory exists
        # post: all valid yearly files downloaded and saved to out_dir
        # desc: executes HTTP requests for each configured year and station, logging results and errors.

        saved_files = []
        self.logger.info(
            f"[{self.station}] Starting RequestPipe for {self.start_year}-{self.end_year}"
        )

        for year in range(self.start_year, self.end_year + 1):
            file_name = f"{self.FILE_PREFIX}-{year}-{self.station}{self.FILE_SUFFIX}"
            url = f"{self.base_url}/{year}/{file_name}"
            self.logger.debug(f"[{self.station}] GET {url}")

            try:
                response = requests.get(url, timeout=self.timeout)

                if response.status_code == 200 and len(response.content) > self.min_bytes:
                    out_file = self.out_dir / f"{self.OUTPUT_PREFIX}{self.station}_{year}{self.FILE_SUFFIX}"
                    out_file.write_text(response.text, encoding="utf-8")
                    self.logger.info(f"[{self.station}] Saved {out_file}")
                    saved_files.append(out_file)
                else:
                    self.logger.warning(
                        f"[{self.station}] Skipped {year}: HTTP {response.status_code} "
                        f"({len(response.content)} bytes)"
                    )

            except Exception as e:
                self.logger.error(f"[{self.station}] Failed {year}: {e}")

        self.logger.info(f"[{self.station}] RequestPipe complete — {len(saved_files)} files saved.")
        return saved_files
