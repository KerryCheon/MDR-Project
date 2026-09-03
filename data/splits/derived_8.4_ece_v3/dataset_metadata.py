# dataset_metadata.py
# Metadata and configuration constants for derived_8.4-ece splits

from __future__ import annotations

import os
from typing import List

# In-situ ECE sensor deployment stations (Washington State)
ECE_STATIONS = [
    "ECE_BBG_Main_St",
    "ECE_BBG_Lost_Meadow",
    "ECE_Renton_Home",
    "ECE_Renton_Garden_North",
    "ECE_Renton_Garden_Shed",
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


def get_eval_csv() -> str:
    return os.path.join(get_split_dir(), "eval.csv")
