"""
Sequence dataset for the GRU->Transformer hybrid soil moisture regressor.

Mirrors the LSTM raw-series dataset: each sample contains

  x_time  : (seq_len, n_time)   float32  — raw daily observations (the sequence)
  x_static: (n_static,)         float32  — fixed location/terrain features
  year    : scalar int32                  — calendar year of target day
  y       : scalar float32               — soil moisture at the last timestep

The model receives x_time as a sequence (short-term dynamics go through the
GRU, long-range context through the Transformer encoder) and x_static as a
fixed context concatenated into the prediction head.
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
    """Build sequence samples from a single split dataframe.

    Static features are constant within a station; we take them from the
    first row of each station group.

    Returns
    -------
    x_time  : (N, seq_len, n_time)   float32
    x_static: (N, n_static)          float32
    years   : (N,)                   int32
    y_vals  : (N,)                   float32
    """
    x_time_list, x_static_list, year_list, y_list = [], [], [], []

    for _station, grp in df.groupby("station_id", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)

        X_t  = grp[time_cols].to_numpy(dtype=np.float32)   # (T, n_time)
        y    = grp[TARGET].to_numpy(dtype=np.float32)      # (T,)
        yrs  = pd.to_datetime(grp["date"]).dt.year.to_numpy(dtype=np.int32)

        # static features: constant for the station — take first valid row
        static_row = grp[static_cols].iloc[0].to_numpy(dtype=np.float32)  # (n_static,)

        for i in range(seq_len, len(grp), stride):
            x_time_list.append(X_t[i - seq_len : i])  # (seq_len, n_time)
            x_static_list.append(static_row)           # (n_static,)
            year_list.append(yrs[i])
            y_list.append(y[i])

    if not x_time_list:
        raise ValueError("No sequences built — check seq_len vs. data length.")

    return (
        np.stack(x_time_list,   axis=0).astype(np.float32),   # (N, seq_len, n_time)
        np.stack(x_static_list, axis=0).astype(np.float32),   # (N, n_static)
        np.array(year_list,  dtype=np.int32),                  # (N,)
        np.array(y_list,     dtype=np.float32),                # (N,)
    )


class SoilMoistureDataset(Dataset):
    """PyTorch Dataset for the GRU->Transformer hybrid."""

    def __init__(
        self,
        x_time:   np.ndarray,   # (N, seq_len, n_time)
        x_static: np.ndarray,   # (N, n_static)
        years:    np.ndarray,   # (N,)
        y:        np.ndarray,   # (N,)
    ):
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
    """Build (train, val, test) SoilMoistureDataset objects.

    train_stride controls step between training sequences (stride=1 maximises
    samples; larger values give more independent training samples).
    Val/test always use stride=1 for complete coverage.

    Returns
    -------
    ds_train, ds_val, ds_test : SoilMoistureDataset
    """
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
