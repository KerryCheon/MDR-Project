# Jakob Balkovec
# Now 8th 2025
# clean.py

# This script cleans the master dataset. It removes all rows with missing values

import pandas as pd
import os

MASTER = '/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master/final_master.pkl'
df = pd.read_pickle(MASTER)

# Drop rows with missing values, excluding SM_prev and SM_label
cols_to_check = [c for c in df.columns if c not in ["SM_prev", "SM_label"]]
df_clean = df.dropna(subset=cols_to_check)

print(f"Original Length: {len(df)}, Cleaned Length: {len(df_clean)}")

CLEAN_DIR = '/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master_cleaned'
os.makedirs(CLEAN_DIR, exist_ok=True)

CLEAN_PATH_PKL = os.path.join(CLEAN_DIR, 'final_master_cleaned.pkl')
CLEAN_PATH_XLSX = os.path.join(CLEAN_DIR, 'final_master_cleaned.xlsx')
CLEAN_PATH_CSV = os.path.join(CLEAN_DIR, 'final_master_cleaned.csv')

df_clean.to_pickle(CLEAN_PATH_PKL)
df_clean.to_excel(CLEAN_PATH_XLSX, index=False)
df_clean.to_csv(CLEAN_PATH_CSV, index=False)

print(f"Saved cleaned master dataset to:"
      f"\n\t{CLEAN_PATH_PKL}"
      f"\n\t{CLEAN_PATH_XLSX}"
      f"\n\t{CLEAN_PATH_CSV}")


