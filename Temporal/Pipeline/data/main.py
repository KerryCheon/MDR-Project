# Jakob Balkovec

# The purpose of this file is to join together all the individual .csv files into one
# master file that will be used in the next stages.

# As of this moment (2025-10-23) idk if this is permanent or just a temporary fix.

import pandas as pd
from pathlib import Path
from tqdm import tqdm

SPOKANE_CSV = Path("processed/spokane/final.csv")
DARRINGTON_CSV = Path("processed/darrington/final.csv")
QUINAULT_CSV = Path("processed/quinault/final.csv")
TOUCHNET_CSV = Path("processed/touchnet/final.csv")
SOURDOUGH_CSV = Path("processed/sourdough/final.csv")

MASTER_CSV = Path("master/final_master.csv")
MASTER_PKL = Path("master/final_master.pkl")

csv_paths = [SPOKANE_CSV, DARRINGTON_CSV, QUINAULT_CSV, TOUCHNET_CSV, SOURDOUGH_CSV]

dfs = []
for p in csv_paths:
    if not p.exists():
        raise FileNotFoundError(f"Missing input CSV: {p.resolve()}")
    dfs.append(pd.read_csv(p, parse_dates=["date"]))

master_df = pd.concat(tqdm(dfs, desc="Concatenating DataFrames"), ignore_index=True)

MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
master_df.to_csv(MASTER_CSV, index=False)
master_df.to_pickle(MASTER_PKL)

print(f"Master dataset created:\n - {MASTER_CSV.resolve()}\n - {MASTER_PKL.resolve()}")
