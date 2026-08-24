# run_remaining_oos.py
import os
import sys
import yaml
from src.pipeline.main import run_pipeline_for_station
from src.pipeline.utils.config import load_config
from src.pipeline.utils.logger import get_logger, setup_logger

STATIONS_TO_RUN = [
    'uscrn_riley_10_wsw',
    'uscrn_murphy_10_w',
    'uscrn_redding_12_wnw',
    'uscrn_boulder_14_w',
    'uscrn_lander_11_sse',
    'uscrn_wolf_point_29_ene',
    'snotel_clackamas_lake',
    'snotel_rock_springs'
]

def main():
    config_path = 'src/pipeline/config_8.4_oos.yaml'
    config = load_config(config_path)
    setup_logger(config)
    logger = get_logger()

    stations_cfg = config.get('stations', {})
    total = len(STATIONS_TO_RUN)

    for i, st in enumerate(STATIONS_TO_RUN, 1):
        logger.info(f'[{i}/{total}] Starting pipeline for {st}...')
        st_cfg = stations_cfg.get(st)
        if not st_cfg:
            logger.error(f'Station config not found for {st}!')
            continue
        try:
            run_pipeline_for_station(st, st_cfg, config)
            logger.info(f'[{i}/{total}] Successfully finished {st}!')
        except Exception as e:
            logger.error(f'[{i}/{total}] Failed on {st}: {e}')

    logger.info('All remaining out-of-state pipelines finished.')

if __name__ == '__main__':
    main()
