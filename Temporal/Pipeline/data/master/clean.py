# Jakob Balkovec
# Now 8th 2025
# clean.py

# filters the dataset -> 2011-10-06 and on (where soil_moisture_5cm != NULL)

import pandas as pd
import os

MASTER = '/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master/final_master.pkl'
df = pd.read_pickle(MASTER)

# Normalizing station IDs to names to match the format of SNOTEL data
station_id_map = {
    4223: "Darrington",
    4237: "Quinault",
    4136: "Spokane",
}

df["station_id"] = df["station_id"].replace(station_id_map).astype(str)

print("Station IDs after normalization:")
print(sorted(df["station_id"].dropna().unique()))

snotel_cols = [c for c in df.columns if c.startswith("SNOTEL")]
if snotel_cols:
    print(f"\nDropping {len(snotel_cols)} SNOTEL columns:")
    print(snotel_cols)
    df = df.drop(columns=snotel_cols)
else:
    print("\nNo SNOTEL columns found to drop.")

df["date"] = pd.to_datetime(df["date"])
CUTOFF_DATE = pd.Timestamp("2011-10-06")
df = df[df["date"] >= CUTOFF_DATE].copy()

print(f"\nFiltered Length (post-cutoff): {len(df)}")

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

