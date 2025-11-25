# Jakob Balkovec
# Nov 23rd 2025
# playground.py

import pandas as pd
import json as js

from Temporal.Pipeline.utils.config import load_config
from Temporal.Pipeline.utils.logger import setup_logger, get_logger
from Temporal.Pipeline.imputers.api import transform_with_ensemble

config = load_config()
setup_logger(config)

path = r'/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/experiments/missing_values/satellite/satellite_data_darrington_2022_2024.csv'

df = pd.read_csv(path)

imp_df, diag = transform_with_ensemble(
    df,
    col='NDVI',
    return_diag=True,
    auto_validate=True
)

print(js.dumps(diag, indent=4))
