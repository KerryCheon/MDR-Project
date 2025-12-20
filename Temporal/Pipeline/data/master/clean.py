# Jakob Balkovec
# Now 8th 2025
# clean.py

# filters the dataset -> 2011-10-06 and on (where soil_moisture_5cm != NULL)

import pandas as pd
import os

MASTER = '/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master/final_master.pkl'
df = pd.read_pickle(MASTER)

# Filter out data before soil moisture becomes available
df["date"] = pd.to_datetime(df["date"])
CUTOFF_DATE = pd.Timestamp("2011-10-06")
df = df[df["date"] >= CUTOFF_DATE].copy()

print(f"Filtered Length (post-cutoff): {len(df)}")

CLEAN_DIR = '/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master_cleaned'
os.makedirs(CLEAN_DIR, exist_ok=True)

CLEAN_PATH_PKL = os.path.join(CLEAN_DIR, 'final_master_cleaned.pkl')
CLEAN_PATH_XLSX = os.path.join(CLEAN_DIR, 'final_master_cleaned.xlsx')
CLEAN_PATH_CSV = os.path.join(CLEAN_DIR, 'final_master_cleaned.csv')

df.to_pickle(CLEAN_PATH_PKL)
df.to_excel(CLEAN_PATH_XLSX, index=False)
df.to_csv(CLEAN_PATH_CSV, index=False)

print(f"Saved filtered master dataset to:"
      f"\n\t{CLEAN_PATH_PKL}"
      f"\n\t{CLEAN_PATH_XLSX}"
      f"\n\t{CLEAN_PATH_CSV}")

