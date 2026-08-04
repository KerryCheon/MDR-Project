"""
Sequence dataset builder for soil moisture LSTM prediction on derived_8.0 split.

Groups rows by station_id, sorts by date, and builds sliding windows of length `seq_len`.
Front-pads each station group by (seq_len - 1) rows so that EVERY row in the tabular split
has a corresponding sequence window, maintaining 1:1 row alignment across train, val, and test splits.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

TARGET = "soil_moisture_5cm"


def build_sequences(df: pd.DataFrame, feature_cols: list[str], seq_len: int = 10, stride: int = 1):
    """Return (X_seqs, y_vals) arrays built from df grouped by station_id and sorted by date.
    Front-pads each station group with (seq_len - 1) copies of its first row to preserve 1:1 sample count.
    """
    X_seqs, y_vals = [], []

    for station, grp in df.groupby("station_id", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        pad_count = seq_len - 1
        pad_df = pd.concat([grp.iloc[:1] for _ in range(pad_count)] + [grp], ignore_index=True)

        X = pad_df[feature_cols].to_numpy(dtype=np.float32)
        y = pad_df[TARGET].to_numpy(dtype=np.float32)

        for i in range(seq_len - 1, len(pad_df), stride):
            X_seqs.append(X[i - (seq_len - 1) : i + 1])
            y_vals.append(y[i])

    if not X_seqs:
        raise ValueError(f"No sequences built with seq_len={seq_len}")

    return np.stack(X_seqs, axis=0), np.array(y_vals, dtype=np.float32)


class SoilMoistureDataset(Dataset):
    """PyTorch Dataset wrapping sequence arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_datasets(train_df, val_df, test_df, feature_cols: list[str], seq_len: int = 10, train_stride: int = 1):
    """Build SoilMoistureDataset instances for train, val, and test splits."""
    X_tr, y_tr = build_sequences(train_df, feature_cols, seq_len, stride=train_stride)
    X_va, y_va = build_sequences(val_df, feature_cols, seq_len, stride=1)
    X_te, y_te = build_sequences(test_df, feature_cols, seq_len, stride=1)

    print(f"[lstm dataset] train={X_tr.shape} val={X_va.shape} test={X_te.shape} (stride={train_stride})", flush=True)

    return (
        SoilMoistureDataset(X_tr, y_tr),
        SoilMoistureDataset(X_va, y_va),
        SoilMoistureDataset(X_te, y_te),
    )
