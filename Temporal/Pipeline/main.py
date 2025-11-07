# Jakob Balkovec & Kerry Cheon
# Main Pipeline Execution

# This script orchestrates the execution of the full data processing pipeline,
# chaining together the request, parse, clean, merge, and save pipes.

# MUTE THE ANNOYING INSECURE REQUESTS WARNING
import warnings
from requests import packages

warnings.filterwarnings("ignore", category=UserWarning, module="requests")
# MUTE THE ANNOYING INSECURE REQUESTS WARNING

# Jakob Balkovec & Kerry Cheon
# MDR Project - Main Pipeline Runner

from utils.config import load_config
from utils.logger import get_logger

from pipes.request_pipe import RequestPipe
from pipes.parse_pipe import ParsePipe
from pipes.clean_pipe import CleanPipe
from pipes.merge_pipe import MergePipe
from pipes.temporal_fill_pipe import TemporalFillPipe
from pipes.satellite_pipe import SatellitePipe
from pipes.feature_pipe import FeaturePipe
from pipes.save_pipe import SavePipe

def run_pipeline_for_station(station_name, station_cfg, global_cfg):
    logger = get_logger().getChild(f"main.{station_name}")
    logger.info(f"=== Starting pipeline for {station_name} ===")

    # Skip RequestPipe for SNOTEL stations (they use local .stm files, not HTTP downloads)
    if station_cfg.get("parse", {}).get("snotel_mode", False):
        logger.info(f"[{station_name}] SNOTEL mode - skipping RequestPipe, will parse .stm files directly")
    else:
        request_pipe = RequestPipe(config=station_cfg["request"])
        request_pipe.run()

    # ParsePipe handles both USCRN and SNOTEL formats automatically
    parsed = ParsePipe(config=station_cfg["parse"]).run()
    cleaned = CleanPipe(config=station_cfg["clean"]).run(parsed)
    merged = MergePipe(config=station_cfg["merge"]).run(cleaned)
    filled = TemporalFillPipe(config=global_cfg["temporal_fill"]).run(merged)
    with_sat = SatellitePipe(config=global_cfg, station_name=station_name).run(filled)
    featured = FeaturePipe(config=station_cfg.get("feature", {})).run(with_sat)
    SavePipe(config=station_cfg["save"]).run(featured)

    logger.info(f"=== Pipeline complete for {station_name} ===\n")


if __name__ == "__main__":
    config = load_config()
    logger = get_logger()

    stations_cfg = config.get("stations", {})
    if not stations_cfg:
        logger.error("No stations found. Check the 'config' file!")
        exit(1)

    for station_name, station_cfg in stations_cfg.items():
        run_pipeline_for_station(station_name, station_cfg, config)

    logger.info("All station pipelines completed successfully.")