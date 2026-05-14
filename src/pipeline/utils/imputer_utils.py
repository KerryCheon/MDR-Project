# Jakob Balkovec
# Nov 16th 2025
# imputer_utils.py

import pandas as pd
import numpy as np

from pipeline.Pipeline.imputers.voting import VotingImputer
from pipeline.Pipeline.imputers.interpolation import LinearInterpolationImputer
from pipeline.Pipeline.imputers.fbfill import ForwardBackwardImputer
from pipeline.Pipeline.imputers.smoothing import RollingMeanImputer
from pipeline.Pipeline.imputers.climatology import ClimatologyImputer
from pipeline.Pipeline.imputers.linear_model import LinearModelImputer
from pipeline.Pipeline.imputers.xgb_model import XGBImputer
from pipeline.Pipeline.imputers.gaussian_regression import GaussianProcessImputer
from pipeline.Pipeline.imputers.knn_temporal import KNNImputer
from pipeline.Pipeline.imputers.seasonal_naive import SeasonalNaiveImputer
from pipeline.Pipeline.imputers.spline_interpolation import SplineImputer

def _validate_inputs(df, col, logger):
    # pre:  df must have 'date' column and 'col' to impute
    # post: raises ValueError if checks fail

    logger.debug("validating inputs...")
    if "date" not in df.columns:
        logger.error("missing required 'date' column")
        raise ValueError("transform_with_ensemble requires a 'date' column")
    if col not in df.columns:
        logger.error(f"column '{col}' does not exist in dataframe")
        raise ValueError(f"column '{col}' does not exist in dataframe")
    logger.debug("input validation passed")


def _prepare_dataframe(df, logger):
    # pre:  df has 'date' column
    # post: returns df sorted by date

    logger.debug("preparing and sorting dataframe")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    logger.debug(f"sorted dataframe with {len(df)} rows")
    return df


def _apply_ndvi_lockout(df, col, cfg, logger):
    # pre: df has 'date' and col
    # post: adds '_ndvi_lock' column if col is 'NDVI'

    if col.lower() != "ndvi":
        logger.debug("column is not NDVI; setting ndvi_lock=False")
        df["_ndvi_lock"] = False
        return df

    logger.debug("applying NDVI lockout rule")
    ndvi_max_gap = cfg.get("imputer", {}).get("ndvi_max_gap", 10)

    s = df[col]
    idx = df["date"]
    mask_real = ~s.isna()

    prev = idx.where(mask_real).ffill()
    next = idx.where(mask_real).bfill()
    gap_len = (next - prev).dt.days

    df["_ndvi_lock"] = gap_len > ndvi_max_gap

    logger.info(
        f"NDVI: found {int(df['_ndvi_lock'].sum())} timestamps beyond "
        f"max_gap={ndvi_max_gap}, locking these out of imputation."
    )
    return df


def _should_skip_interpolation(col, cfg, logger):
    # pre:  col is feature name
    # post: returns True if col is in high_freq_feat list in config

    high_freq_feat = cfg.get("imputer", {}).get("high_freq_feat", [])

    if col in high_freq_feat:
        logger.info(f"feature '{col}' is marked high-frequency; skipping interpolation")
        return True
    return False


def _run_ensemble(df, col, dates, values, logger):
    # pre:  df has 'date' and col
    # post: runs VotingImputer and returns filled, conf, voter

    logger.debug(f"initializing imputers for column '{col}'")
    imputers = [
        LinearInterpolationImputer(),
        ForwardBackwardImputer(),
        RollingMeanImputer(),
        ClimatologyImputer(),
        LinearModelImputer(),
        XGBImputer(),
        GaussianProcessImputer(),
        KNNImputer(),
        SeasonalNaiveImputer(),
        SplineImputer()
    ]

    logger.debug(f"initialized {len(imputers)} imputers")

    voter = VotingImputer(imputers)

    logger.debug("fitting ensemble...")
    voter.fit(dates, values, aux_df=df)

    logger.debug("performing ensemble imputation...")
    filled, conf = voter.impute(dates, values, aux_df=df)

    logger.debug("ensemble imputation complete")
    return filled, conf, voter


def _apply_postprocessing(df, col, filled, conf, ndvi_lock, gap_lengths, logger):
    # pre:  df has 'date' and col
    # post: applies ndvi lockouts, adds metadata columns

    logger.debug("applying postprocessing...")

    # enforce NDVI lockouts
    if ndvi_lock is not None and ndvi_lock.any():
        logger.info(f"NDVI lockout: forcing {ndvi_lock.sum()} timestamps to remain sparse")
        filled.values[ndvi_lock] = np.nan
        conf.values[ndvi_lock] = 0.0

    df[col + "_interp"] = filled.values
    df[col + "_conf"] = conf.values

    # normalize confidence
    df[col + "_conf_norm"] = df[col + "_conf"] / df[col + "_conf"].max()

    # attach gap length metadata
    df[col + "_gap_length"] = gap_lengths

    logger.debug("postprocessing complete")
    return df


def _apply_gap_confidence(conf, gap_lengths, logger, tau_gap):
    # pre: conf is a pandas Series, gap_lengths is np.array
    # post: returns adjusted confidence scores

    logger.debug("applying gap-based confidence scaling")

    gap_lengths = np.asarray(gap_lengths, dtype=float)
    gap_lengths = np.nan_to_num(gap_lengths, nan=0.0)

    scale = np.exp(-gap_lengths / float(tau_gap))

    conf_scaled = conf.values * scale
    conf_scaled = np.clip(conf_scaled, 0.0, 1.0)

    logger.debug("gap-based confidence scaling complete")
    return conf_scaled


def _apply_feature_engineering(df, logger):
    # pre:  df has 'date' column
    # post: returns df with added engineered features
    # desc: adds basic temporal features for imputation models

    logger.debug("applying internal feature engineering")

    df = df.copy()

    df["year"] = df["date"].dt.year
    df["DOY"] = df["date"].dt.dayofyear

    df["month"] = df["date"].dt.month

    df["DOY_sin"] = np.sin(2 * np.pi * df["DOY"] / 365.25)
    df["DOY_cos"] = np.cos(2 * np.pi * df["DOY"] / 365.25)

    logger.debug("added year, DOY, DOY_sin, DOY_cos features")

    return df

def _run_diagnostics(voter, dates, values, df, diag_path, logger):
    # pre:  voter is VotingImputer
    # post: runs diagnostics and returns summary dict

    logger.debug("running diagnostics...")
    return voter.diagnostics(
        dates,
        values,
        aux_df=df,
        path=diag_path
    )
