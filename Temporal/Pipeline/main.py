# Jakob Balkovec & Kerry Cheon
# Main Pipeline Execution

# This script orchestrates the execution of the full data processing pipeline,
# chaining together the request, parse, clean, merge, and save pipes.

# MUTE THE ANNOYING REQUESTS WARNING
import warnings
from requests import packages

warnings.filterwarnings("ignore", category=UserWarning, module="requests")
# MUTE THE ANNOYING REQUESTS WARNING

from utils.config import load_config
from utils.logger import get_logger, setup_logger
import argparse
import sys

from pipes.request_pipe import RequestPipe
from pipes.parse_pipe import ParsePipe
from pipes.clean_pipe import CleanPipe
from pipes.merge_pipe import MergePipe
from pipes.satellite_pipe import SatellitePipe
from pipes.temporal_fill_pipe import TemporalFillPipe
from pipes.whittaker_pipe import WhittakerPipe
from pipes.feature_pipe import FeaturePipe
from pipes.weather_pipe import WeatherPipe
from pipes.save_pipe import SavePipe

def run_pipeline_for_station(station_name, station_cfg, global_cfg):
    logger = get_logger().getChild(f"main.{station_name}")
    logger.info(f"=== Starting pipeline for {station_name} ===")

    # Skip RequestPipe for SNOTEL stations (they use local .stm files, not HTTP downloads)
    if station_cfg.get("parse", {}).get("snotel_mode", False):
        logger.warning(f"[{station_name}] SNOTEL mode detected — skipping entire station (SNOTELPipe not available).")
        return

    # USCRN pipeline
    try:
        request_pipe = RequestPipe(config=station_cfg["request"])
        request_pipe.run()

        parsed = ParsePipe(config=station_cfg["parse"]).run()
        cleaned = CleanPipe(config=station_cfg["clean"]).run(parsed)
        merged = MergePipe(config=station_cfg["merge"]).run(cleaned)

        with_sat = SatellitePipe(config=global_cfg, station_name=station_name).run(merged)

        with_weather = WeatherPipe(config=global_cfg, station_name=station_name).run(with_sat)

        filled = TemporalFillPipe(config=global_cfg["temporal_fill"]).run(with_weather)
        smoothed = WhittakerPipe(config=global_cfg["whittaker"]).run(filled)
        featured = FeaturePipe(config=station_cfg.get("feature", {})).run(smoothed)

        SavePipe(config=station_cfg["save"]).run(featured)

        logger.info(f"=== Pipeline complete for {station_name} ===\n")

    except Exception as e:
        logger.error(f"[{station_name}] Pipeline failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MDR Temporal Pipeline for configured stations.")
    parser.add_argument("--station", "-s", help="Run pipeline for a single station key from config.yaml")
    parser.add_argument("--config", "-c", help="Path to config YAML file (defaults to config.yaml in repo)")
    parser.add_argument("--list-stations", action="store_true", help="List configured stations and exit")
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    setup_logger(config)
    logger = get_logger()

    stations_cfg = config.get("stations", {})
    if not stations_cfg:
        logger.error("No stations found. Check the 'config' file!")
        exit(1)
    if args.list_stations:
        logger.info("Configured stations:")
        for s in stations_cfg.keys():
            logger.info(f" - {s}")
        sys.exit(0)

    if args.station:
        station_cfg = stations_cfg.get(args.station)
        if station_cfg is None:
            logger.error(f"Station '{args.station}' not found in configuration.")
            sys.exit(1)

        # explicit skip for SNOTEL mode stations (config flag)
        if station_cfg.get("parse", {}).get("snotel_mode", False):
            logger.warning(f"[{args.station}] Skipping this station — awaiting [_station_]Pipe integration.")
            sys.exit(0)

        run_pipeline_for_station(args.station, station_cfg, config)

    else:
        for station_name, station_cfg in stations_cfg.items():

            if station_cfg.get("parse", {}).get("snotel_mode", False):
                logger.warning(f"[{station_name}] Skipping this station — awaiting {station_name}Pipe integration.")
                continue

            run_pipeline_for_station(station_name, station_cfg, config)

    logger.info("All station pipelines completed successfully.")
