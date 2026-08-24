# dataset_metadata.py
# Metadata and configuration constants for derived_8.4-oos splits

import os
from typing import List

# Chronological split years matching derived_8.4
TRAIN_YEARS = [2017, 2018, 2019, 2020]
VAL_YEARS = [2021, 2022]
TEST_YEARS = [2023, 2024, 2025]

# Out-of-state stations in derived_8.4-oos
OOS_STATIONS = [
    "John_Day_35_WNW",
    "Corvallis_10_SSW",
    "Riley_10_WSW",
    "Murphy_10_W",
    "Redding_12_WNW",
    "Boulder_14_W",
    "Lander_11_SSE",
    "Wolf_Point_29_ENE",
    "Clackamas_Lake_398",
    "Rock_Springs_721",
]

TARGET_COL = "soil_moisture_5cm"
DATE_COL = "date"
STATION_COL = "station_id"

def get_split_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def get_train_csv() -> str:
    return os.path.join(get_split_dir(), "train.csv")

def get_val_csv() -> str:
    return os.path.join(get_split_dir(), "val.csv")

def get_test_csv() -> str:
    return os.path.join(get_split_dir(), "test.csv")
