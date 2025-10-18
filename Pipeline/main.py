# Jakob Balkovec
# Main Pipeline Execution

# This script orchestrates the execution of the full data processing pipeline,
# chaining together the request, parse, clean, merge, and save pipes.

# MUTE THE ANNOYING INSECURE REQUESTS WARNING
import warnings
from requests import packages

warnings.filterwarnings("ignore", category=UserWarning, module="requests")
# MUTE THE ANNOYING INSECURE REQUESTS WARNING

from utils.config import load_config
from utils.logger import setup_logger
from pipes.request_pipe import RequestPipe
from pipes.parse_pipe import ParsePipe
from pipes.clean_pipe import CleanPipe
from pipes.merge_pipe import MergePipe
from pipes.save_pipe import SavePipe
from pipes.temporal_fill_pipe import TemporalFillPipe
from pipes.feature_pipe import FeaturePipe
from pipes.satelite_pipe import SatellitePipe

def main():
    config = load_config()
    setup_logger(config)

    request = RequestPipe(config)
    parser = ParsePipe(config)
    cleaner = CleanPipe(config)
    filler = TemporalFillPipe(config)
    satellite = SatellitePipe(config)
    feature = FeaturePipe(config)
    merger = MergePipe(config)
    saver = SavePipe(config)

    request.run()
    parsed_df = parser.run()
    clean_df = cleaner.run(parsed_df)
    filled_df = filler.run(clean_df)
    sat_df = satellite.run(filled_df)
    feat_df = feature.run(sat_df)
    merged_df = merger.run(feat_df)
    saver.run(merged_df)

if __name__ == "__main__":
    main()
