# Jakob Balkovec
# Request Pipe

# This module defines the RequestPipe class, which handles
# HTTP requests to fetch data from the NOAA USCRN dataset.

import socket
import requests
import urllib3.util.connection as urllib3_cn
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from ..utils.logger import get_logger
from ..utils.config import load_config

# Patch urllib3 to force IPv4 and prevent slow connection/DNS resolution timeouts (e.g., IPv6 fallbacks)
def allowed_gai_family():
    return socket.AF_INET

urllib3_cn.allowed_gai_family = allowed_gai_family


class RequestPipe:
    FILE_PREFIX = "CRND0103"
    FILE_SUFFIX = ".txt"
    OUTPUT_PREFIX = "uscrn_"

    def __init__(self, config=None):
        # pre: config is a dictionary loaded from config.yaml or None
        # post: initializes the RequestPipe with configuration settings
        # desc: reads request parameters from the config file and sets up logging/output paths.

        self.config = config or load_config()
        req_cfg = self.config

        self.base_url = req_cfg["base_url"]
        self.station = req_cfg["station"]
        self.start_year = req_cfg["start_year"]
        self.end_year = req_cfg["end_year"]
        self.timeout = req_cfg.get("timeout", 20)
        self.min_bytes = req_cfg.get("min_bytes", 500)

        self.out_dir = Path(req_cfg.get("out_dir", f"data/{self.station}/raw"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger().getChild(f"request.{self.station}")

    def _is_valid_content(self, text: str) -> bool:
        # pre:  text is a string
        # post: returns True if text is a valid USCRN data format, False otherwise
        # desc: validates length, checks against HTML markers, and verifies daily format column prefix
        if not text or len(text.encode("utf-8")) <= self.min_bytes:
            return False

        text_strip = text.strip()
        if text_strip.startswith("<") or "<html>" in text_strip.lower():
            return False

        lines = text_strip.splitlines()
        if not lines:
            return False

        first_line = lines[0].strip()
        parts = first_line.split()
        if len(parts) < 5:
            return False

        # WBANNO should be 5 digits and LST_DATE should be 8 digits
        if not (parts[0].isdigit() and len(parts[0]) == 5):
            return False
        if not (parts[1].isdigit() and len(parts[1]) == 8):
            return False

        return True

    def _is_valid_file(self, path: Path) -> bool:
        # pre:  path is a Path object representing a local file
        # post: returns True if the file exists and has valid content, False otherwise
        # desc: opens local file to check its validity.
        if not path.exists():
            return False
        try:
            text = path.read_text(encoding="utf-8")
            return self._is_valid_content(text)
        except Exception:
            return False

    def _download_year(self, session: requests.Session, year: int) -> Path | None:
        # pre:  session is a requests.Session object, year is an integer
        # post: downloads, validates, and saves USCRN daily file for year to out_dir
        # desc: performs HTTP GET to download data, validates content, and writes atomically.
        out_file = self.out_dir / f"{self.OUTPUT_PREFIX}{self.station}_{year}{self.FILE_SUFFIX}"

        if self._is_valid_file(out_file):
            self.logger.info(f"[{self.station}] Valid local file already exists: {out_file}. Skipping download.")
            return out_file

        file_name = f"{self.FILE_PREFIX}-{year}-{self.station}{self.FILE_SUFFIX}"
        url = f"{self.base_url}/{year}/{file_name}"
        self.logger.debug(f"[{self.station}] GET {url}")

        try:
            response = session.get(url, timeout=self.timeout)

            if response.status_code == 200:
                text_content = response.text
                if self._is_valid_content(text_content):
                    # Write atomically using a temporary file
                    temp_file = out_file.with_suffix(".tmp")
                    try:
                        temp_file.write_text(text_content, encoding="utf-8")
                        temp_file.replace(out_file)
                        self.logger.info(f"[{self.station}] Saved {out_file}")
                        return out_file
                    except Exception as write_err:
                        self.logger.error(f"[{self.station}] Failed to write/rename {out_file}: {write_err}")
                        if temp_file.exists():
                            temp_file.unlink()
                else:
                    self.logger.warning(
                        f"[{self.station}] Skipped {year}: HTTP {response.status_code} "
                        f"but content was invalid or did not match USCRN schema ({len(response.content)} bytes)"
                    )
            else:
                self.logger.warning(
                    f"[{self.station}] Skipped {year}: HTTP {response.status_code}"
                )

        except Exception as e:
            self.logger.error(f"[{self.station}] Failed {year}: {e}")

        return None

    def run(self, _=None):
        # pre:  configuration loaded and output directory exists
        # post: all valid yearly files downloaded and saved to out_dir
        # desc: executes HTTP requests for each configured year and station, logging results and errors.

        self.logger.info(
            f"[{self.station}] Starting RequestPipe for {self.start_year}-{self.end_year}"
        )

        saved_files = []
        years = list(range(self.start_year, self.end_year + 1))
        max_workers = self.config.get("max_workers", 5)

        with requests.Session() as session:
            # Configure connection pool size to match max_workers to avoid pool connection reuse issues
            adapter = HTTPAdapter(
                pool_connections=max_workers,
                pool_maxsize=max_workers
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._download_year, session, year): year
                    for year in years
                }
                for future in futures:
                    result = future.result()
                    if result:
                        saved_files.append(result)

        saved_files.sort()
        self.logger.info(f"[{self.station}] RequestPipe complete — {len(saved_files)} valid files ready/saved.")
        return saved_files
