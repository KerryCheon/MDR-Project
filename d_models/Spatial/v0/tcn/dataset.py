"""
Sequence dataset for the Temporal Convolutional Network (TCN) soil moisture
model, with an added physics channel used by the mass-balance loss.

Each sample contains:
  x_time     : (seq_len, n_time)  float32  — normalised daily features
  x_static   : (n_static,)        float32  — fixed terrain / soil features
  precip_seq : (seq_len,)         float32  — UN-NORMALISED precip_mm (raw mm)
  prev_sm    : scalar             float32  — soil_moisture_5cm at t-1 (may be NaN)
  year       : scalar int32                — calendar year of the target day
  y          : scalar float32              — soil_moisture_5cm at the target day

The physics channel (`prev_sm`, `precip_seq[-1]`) supports a mass-balance
soft-constraint in the training loss:
    ΔSM = pred - prev_sm
    if precip_today ≈ 0  →  penalise ΔSM > small_tol  (can't gain moisture
                             without rain — inspired by PGDL literature)

`prev_sm` may be NaN for the earliest rows of a station; the training loop
masks those samples out of the physics term but still uses them for the data
loss.  `precip_seq` is stored raw (pre-scaling) so the loss can threshold on
actual millimetres rather than z-scores.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET     = "soil_moisture_5cm"
PRECIP_COL = "precip_mm"


def _build_sequences(
    df_scaled: pd.DataFrame,
    df_raw:    pd.DataFrame,
    time_cols: list,
    static_cols: list,
    seq_len: int,
    stride: int = 1,
):
    """Build sequence samples from a split.

    Parameters
    ----------
    df_scaled : dataframe with imputed+scaled features (used for x_time/x_static)
    df_raw    : dataframe with the ORIGINAL un-normalised values — we pull
                precip_mm from here so the physics loss sees real mm.

    Returns
    -------
    x_time     : (N, seq_len, n_time)   float32
    x_static   : (N, n_static)          float32
    precip_seq : (N, seq_len)           float32   raw mm, NaN-filled with 0
    prev_sm    : (N,)                   float32   SM at t-1 (NaN allowed)
    years      : (N,)                   int32
    y_vals     : (N,)                   float32
    """
    x_time_list, x_static_list = [], []
    precip_list, prev_sm_list  = [], []
    year_list, y_list          = [], []

    # Align by (station_id, date) just in case indexing differs
    df_scaled = df_scaled.sort_values(["station_id", "date"]).reset_index(drop=True)
    df_raw    = df_raw.sort_values(["station_id", "date"]).reset_index(drop=True)

    for station, grp_s in df_scaled.groupby("station_id", sort=False):
        grp_s = grp_s.sort_values("date").reset_index(drop=True)
        grp_r = df_raw[df_raw["station_id"] == station].sort_values("date").reset_index(drop=True)

        X_t = grp_s[time_cols].to_numpy(dtype=np.float32)     # (T, n_time)
        y   = grp_s[TARGET].to_numpy(dtype=np.float32)        # (T,)  scaled? no, target is not scaled (scaler only fits feature cols) — see train.py
        yrs = pd.to_datetime(grp_s["date"]).dt.year.to_numpy(dtype=np.int32)

        # raw precip from the pre-scaled dataframe
        precip_raw = grp_r[PRECIP_COL].to_numpy(dtype=np.float32)
        precip_raw = np.where(np.isfinite(precip_raw), precip_raw, 0.0)

        # raw target series (identical to `y` here since TARGET is not in feature list) —
        # used for prev_sm.
        sm_raw = grp_r[TARGET].to_numpy(dtype=np.float32)

        static_row = grp_s[static_cols].iloc[0].to_numpy(dtype=np.float32)

        for i in range(seq_len, len(grp_s), stride):
            x_time_list.append(X_t[i - seq_len : i])              # (seq_len, n_time)
            x_static_list.append(static_row)                      # (n_static,)
            precip_list.append(precip_raw[i - seq_len : i])       # (seq_len,)
            prev_sm_list.append(sm_raw[i - 1])                    # scalar (may be NaN)
            year_list.append(yrs[i])
            y_list.append(y[i])

    if not x_time_list:
        raise ValueError("No sequences built — check seq_len vs. data length.")

    return (
        np.stack(x_time_list,   axis=0).astype(np.float32),
        np.stack(x_static_list, axis=0).astype(np.float32),
        np.stack(precip_list,   axis=0).astype(np.float32),
        np.array(prev_sm_list,  dtype=np.float32),
        np.array(year_list,     dtype=np.int32),
        np.array(y_list,        dtype=np.float32),
    )


class SoilMoistureTCNDataset(Dataset):
    """PyTorch Dataset for the TCN model with physics channel."""

    def __init__(
        self,
        x_time:     np.ndarray,
        x_static:   np.ndarray,
        precip_seq: np.ndarray,
        prev_sm:    np.ndarray,
        years:      np.ndarray,
        y:          np.ndarray,
    ):
        self.x_time     = x_time
        self.x_static   = x_static
        self.precip_seq = precip_seq
        self.prev_sm    = prev_sm
        self.years      = years
        self.y          = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x_time[idx]),
            torch.from_numpy(self.x_static[idx]),
            torch.from_numpy(self.precip_seq[idx]),
            torch.tensor(self.prev_sm[idx], dtype=torch.float32),
            torch.tensor(self.years[idx],   dtype=torch.int32),
            torch.tensor(self.y[idx],       dtype=torch.float32),
        )


def build_datasets(
    train_scaled, val_scaled, test_scaled,
    train_raw,    val_raw,    test_raw,
    time_cols: list,
    static_cols: list,
    seq_len: int,
    train_stride: int = 1,
):
    """Build (train, val, test) SoilMoistureTCNDataset objects."""
    Xt_tr, Xs_tr, P_tr, pm_tr, yr_tr, y_tr = _build_sequences(
        train_scaled, train_raw, time_cols, static_cols, seq_len, stride=train_stride
    )
    Xt_va, Xs_va, P_va, pm_va, yr_va, y_va = _build_sequences(
        val_scaled,   val_raw,   time_cols, static_cols, seq_len, stride=1
    )
    Xt_te, Xs_te, P_te, pm_te, yr_te, y_te = _build_sequences(
        test_scaled,  test_raw,  time_cols, static_cols, seq_len, stride=1
    )

    n_prev_nan_tr = int(np.isnan(pm_tr).sum())
    print(
        f"[dataset]  train={Xt_tr.shape}  val={Xt_va.shape}  test={Xt_te.shape}"
        f"  (stride={train_stride})  "
        f"time_feats={len(time_cols)}  static_feats={len(static_cols)}  "
        f"prev_sm_nan_train={n_prev_nan_tr}/{len(pm_tr)}"
    )

    return (
        SoilMoistureTCNDataset(Xt_tr, Xs_tr, P_tr, pm_tr, yr_tr, y_tr),
        SoilMoistureTCNDataset(Xt_va, Xs_va, P_va, pm_va, yr_va, y_va),
        SoilMoistureTCNDataset(Xt_te, Xs_te, P_te, pm_te, yr_te, y_te),
    )
