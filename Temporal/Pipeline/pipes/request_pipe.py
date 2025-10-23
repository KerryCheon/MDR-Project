# Jakob Balkovec
# Request Pipe

# This modules defines the RequestPipe class, which is responsible for handling
# HTTP requests to fetch data from a specified URL.

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
        req_cfg = self.config["pipeline"]["request"]

        # strictly required config settings
        self.base_url = req_cfg["base_url"]
        self.station = req_cfg["station"]
        self.start_year = req_cfg["start_year"]
        self.end_year = req_cfg["end_year"]
        self.timeout = req_cfg.get("timeout", 20)
        self.min_bytes = req_cfg.get("min_bytes", 500)
        self.out_dir = Path(req_cfg.get("out_dir", "data/raw"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger().getChild("request")


    def run(self, _=None):
        # pre:  configuration loaded and output directory exists
        # post: all valid yearly files downloaded and saved to out_dir
        # desc: executes HTTP requests for each configured year and station, logging results and errors.

        saved_files = []
        self.logger.info(
            f"starting request_pipe for {self.station} ({self.start_year}-{self.end_year})"
        )

        for year in range(self.start_year, self.end_year + 1):
            file_name = f"{self.FILE_PREFIX}-{year}-{self.station}{self.FILE_SUFFIX}"
            url = f"{self.base_url}/{year}/{file_name}"
            self.logger.debug(f"GET {url}")

            try:
                response = requests.get(url, timeout=self.timeout)

                if response.status_code == 200 and len(response.content) > self.min_bytes:
                    out_file = (
                        self.out_dir
                        / f"{self.OUTPUT_PREFIX}{self.station}_{year}{self.FILE_SUFFIX}"
                    )
                    out_file.write_text(response.text, encoding="utf-8")
                    self.logger.info(f"saved {out_file}")
                    saved_files.append(out_file)
                else:
                    self.logger.warning(
                        f"skipped {year}: HTTP {response.status_code} "
                        f"({len(response.content)} bytes)"
                    )

            except Exception as e:
                self.logger.error(f"failed {year}: {e}")

        self.logger.info(f"request_pipe complete, saved {len(saved_files)} files.")
        return saved_files
