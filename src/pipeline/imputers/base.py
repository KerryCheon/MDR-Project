# Jakob Balkovec
# Nov 16th 2025
# base.py

from typing import Optional
import pandas as pd

from pipeline.utils.logger import get_logger
from pipeline.utils.config import load_config

class BaseImputer:
    # desc: Base class for all imputers.

    def __init__(self, name: str, config=None):
        # load config once per instance
        self.cfg = config or load_config()
        self.imputer_cfg = self.cfg.get("imputer", {})
        self.log_cfg = self.cfg.get("logging", {})

        self.name = name
        root_logger = get_logger().getChild("imputer")
        self.logger = root_logger.getChild(self.name)

        self.active = True # for graceful exits

    def fit(self, dates: pd.Series, values: pd.Series, aux_df: Optional[pd.DataFrame] = None):
        # pre:  accepts raw date/value series
        # post: stores model state if needed

        return self

    def impute(self, dates, values, aux_df=None):
        # pre:  imputers must return (filled_series, confidence_series)
        # post: must align with original index
        # note: NOP

        raise NotImplementedError
