"""
Sequence dataset for LSTM soil moisture prediction.

Groups rows by station_id, sorts by date, and builds sliding windows of
length `seq_len`. The target is `soil_moisture_5cm` at the final step.
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


TARGET = "soil_moisture_5cm"
DROP_COLS = {"station_id", "date", TARGET}


def _build_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int, stride: int = 1):
    """Return (X_seqs, y_vals) arrays built from df, grouped by station.

    stride > 1 reduces sequence overlap, yielding more independent training samples.
    Use stride=1 for val/test to retain full coverage.
    """
    X_seqs, y_vals = [], []

    for station, grp in df.groupby("station_id", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        X = grp[feature_cols].to_numpy(dtype=np.float32)
        y = grp[TARGET].to_numpy(dtype=np.float32)

        for i in range(seq_len, len(grp), stride):
            X_seqs.append(X[i - seq_len : i])   # shape (seq_len, n_features)
            y_vals.append(y[i])

    if not X_seqs:
        raise ValueError("No sequences built — check seq_len vs. data length.")

    return np.stack(X_seqs, axis=0), np.array(y_vals, dtype=np.float32)


class SoilMoistureDataset(Dataset):
    """PyTorch Dataset wrapping (X_seqs, y_vals) arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X   # (N, seq_len, n_features)
        self.y = y   # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_datasets(train_df, val_df, test_df, feature_cols: list, seq_len: int, train_stride: int = 1):
    """Build (train, val, test) SoilMoistureDataset objects.

    train_stride controls how many days to step between training sequences.
    Val/test always use stride=1 for complete coverage.
    """
    X_tr, y_tr = _build_sequences(train_df, feature_cols, seq_len, stride=train_stride)
    X_va, y_va = _build_sequences(val_df,   feature_cols, seq_len, stride=1)
    X_te, y_te = _build_sequences(test_df,  feature_cols, seq_len, stride=1)

    print(f"[dataset] train={X_tr.shape}  val={X_va.shape}  test={X_te.shape}  (train_stride={train_stride})")

    return (
        SoilMoistureDataset(X_tr, y_tr),
        SoilMoistureDataset(X_va, y_va),
        SoilMoistureDataset(X_te, y_te),
    )
