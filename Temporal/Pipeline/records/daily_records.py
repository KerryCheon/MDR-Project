# Jakob Balkovec
# Nov 16th 2025
# daily_records.py

import numpy as np
import pandas as pd

class DailyRecordBuilder:
    def __init__(self, df, col):
        # pre: df contains a 'date' column and the target col
        # post: initializes the record builder
        # desc: builds daily records with metadata

        self.df = df
        self.col = col

        self.real_mask = df[col].notna().values

        self.real_dates = (
            pd.to_datetime(df["date"])
            .to_numpy(dtype="datetime64[ns]")   # proper vectorized datetime array
        )


    def compute_gap_length(self, t):
        # pre: t is a timestamp
        # post: returns the gap length in days to the nearest real observation
        # desc: computes gap length for imputed values

        if self.real_dates.size == 0:
            return None

        t = np.datetime64(t, "ns")
        diffs = np.abs(self.real_dates - t)

        nearest = diffs.min()   # timedelta64, crash-out fuel
        days = nearest / np.timedelta64(1, "D")

        return int(days)

    def make_records(self, dates, filled, conf, excluded_mask=None):
        records = []
        gap_lengths = []

        if excluded_mask is None:
            excluded_mask = np.zeros(len(dates), dtype=bool)

        for i, (t, v, c) in enumerate(zip(dates, filled, conf)):
            is_real = self.real_mask[i]
            source = "real" if is_real else "imputed"

            gap_length = None
            if not is_real:
                gap_length = self.compute_gap_length(t)

            gap_lengths.append(0 if gap_length is None else gap_length)

            records.append({
                "timestamp": str(t),
                "value": None if np.isnan(v) else float(v),
                "source": source,
                "imputer": None,
                "confidence": float(c),
                "gap_length": gap_length,
                "is_excluded": bool(excluded_mask[i]),
            })

        return records, np.array(gap_lengths, dtype=float)
