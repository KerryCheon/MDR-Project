# Jakob Balkovec
# Nov 16th 2025
# daily_records.py

import numpy as np
import pandas as pd

class DailyRecordBuilder:
    def __init__(self, df, col):
        self.df = df
        self.col = col
        self.real_mask = df[col].notna().values
        self.real_dates = pd.to_datetime(df["date"][self.real_mask]).values

    def compute_gap_length(self, t):
        # pre: t is np.datetime64
        # post: nearest distance to any real timestamp in days
        if len(self.real_dates) == 0:
            return None

        diffs = np.abs(self.real_dates - t)
        nearest = diffs.min()
        return int(nearest.astype("timedelta64[D]").item())

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
