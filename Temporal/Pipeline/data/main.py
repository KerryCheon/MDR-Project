# Jakob Balkovec

# The purpose of this file is to join together all the individual .csv files into one
# master file that will be used in the next stages.

# As of this moment (2025-10-23) idk if this is permanent or just a temporary fix.

import pandas as pd
from pathlib import Path

SPOKANE_CSV = Path("processed/spokane/final.csv")
DARRINGTON_CSV = Path("processed/darrington/final.csv")
QUINAULT_CSV = Path("processed/quinault/final.csv")

SPOKANE_XLSX = Path("processed/spokane/final.xlsx")
DARRINGTON_XLSX = Path("processed/darrington/final.xlsx")
QUINAULT_XLSX = Path("processed/quinault/final.xlsx")

MASTER_CSV = Path("master/final_master.csv")
MASTER_PKL = Path("master/final_master.pkl")
MASTER_XLSX = Path("master/final_master.xlsx")

spokane_df = pd.read_csv(SPOKANE_CSV, parse_dates=["date"])
darrington_df = pd.read_csv(DARRINGTON_CSV, parse_dates=["date"])
quinault_df = pd.read_csv(QUINAULT_CSV, parse_dates=["date"])

spokane_df.to_excel(SPOKANE_XLSX, index=False)
darrington_df.to_excel(DARRINGTON_XLSX, index=False)
quinault_df.to_excel(QUINAULT_XLSX, index=False)

master_df = pd.concat([spokane_df, darrington_df, quinault_df], ignore_index=True)

MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
master_df.to_csv(MASTER_CSV, index=False)
master_df.to_pickle(MASTER_PKL)
master_df.to_excel(MASTER_XLSX, index=False)

print(f"Master dataset created:\n - {MASTER_CSV}\n - {MASTER_PKL}\n - {MASTER_XLSX}")
