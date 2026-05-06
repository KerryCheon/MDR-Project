"""
Sequence dataset for Transformer-encoder soil moisture prediction.

Mirrors the LSTM dataset: each sample is
  x_time  : (seq_len, n_time)   float32  — daily observations (the sequence)
  x_static: (n_static,)         float32  — fixed location/terrain features
  year    : scalar int32                  — calendar year of target day
  y       : scalar float32               — soil moisture at the last timestep

The Transformer consumes x_time via multi-head self-attention across the
seq_len axis and concatenates the pooled representation with the static
branch in the prediction head.  The only difference vs. the LSTM variant
is that the time-feature list here includes a handful of engineered
memory-state signals (API, DSLR, rolling rain sums) — with only ~7k
training rows the encoder benefits from having those statistics pre-baked.
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


TARGET = "soil_moisture_5cm"


def _build_sequences(
    df: pd.DataFrame,
    time_cols: list,
    static_cols: list,
    seq_len: int,
    stride: int = 1,
):
    """Build (x_time, x_static, years, y) arrays by walking each station in time order."""
    x_time_list, x_static_list, year_list, y_list = [], [], [], []

    for _station, grp in df.groupby("station_id", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)

        X_t  = grp[time_cols].to_numpy(dtype=np.float32)   # (T, n_time)
        y    = grp[TARGET].to_numpy(dtype=np.float32)      # (T,)
        yrs  = pd.to_datetime(grp["date"]).dt.year.to_numpy(dtype=np.int32)

        static_row = grp[static_cols].iloc[0].to_numpy(dtype=np.float32)  # (n_static,)

        for i in range(seq_len, len(grp), stride):
            x_time_list.append(X_t[i - seq_len : i])
            x_static_list.append(static_row)
            year_list.append(yrs[i])
            y_list.append(y[i])

    if not x_time_list:
        raise ValueError("No sequences built — check seq_len vs. data length.")

    return (
        np.stack(x_time_list,   axis=0).astype(np.float32),
        np.stack(x_static_list, axis=0).astype(np.float32),
        np.array(year_list,  dtype=np.int32),
        np.array(y_list,     dtype=np.float32),
    )


class SoilMoistureDataset(Dataset):
    """PyTorch Dataset wrapping pre-built sequence tensors."""

    def __init__(self, x_time, x_static, years, y):
        self.x_time   = x_time
        self.x_static = x_static
        self.years    = years
        self.y        = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_time[idx], self.x_static[idx], self.years[idx], self.y[idx]


def build_datasets(
    train_df,
    val_df,
    test_df,
    time_cols: list,
    static_cols: list,
    seq_len: int,
    train_stride: int = 1,
):
    """Build (train, val, test) SoilMoistureDataset objects.  Val/test use stride=1."""
    Xt_tr, Xs_tr, yr_tr, y_tr = _build_sequences(train_df, time_cols, static_cols, seq_len, stride=train_stride)
    Xt_va, Xs_va, yr_va, y_va = _build_sequences(val_df,   time_cols, static_cols, seq_len, stride=1)
    Xt_te, Xs_te, yr_te, y_te = _build_sequences(test_df,  time_cols, static_cols, seq_len, stride=1)

    print(
        f"[dataset]  train={Xt_tr.shape}  val={Xt_va.shape}  test={Xt_te.shape}"
        f"  (stride={train_stride})"
        f"  time_feats={len(time_cols)}  static_feats={len(static_cols)}"
    )

    return (
        SoilMoistureDataset(Xt_tr, Xs_tr, yr_tr, y_tr),
        SoilMoistureDataset(Xt_va, Xs_va, yr_va, y_va),
        SoilMoistureDataset(Xt_te, Xs_te, yr_te, y_te),
    )
