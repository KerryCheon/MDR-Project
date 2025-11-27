# Jakob Balkovec
# Nov 16th 2025
# math_utils.py

import numpy as np
import pandas as pd

def db_to_linear(db: pd.Series | np.ndarray) -> pd.Series:
    # pre: db values
    # post: returns linear scale values
    # desc: converts decibel values to linear scale

    return 10 ** (db / 10)

def linear_to_db(lin: pd.Series | np.ndarray) -> pd.Series:
    # pre: linear scale values
    # post: returns decibel values
    # desc: converts linear scale values to decibel scale

    lin_safe = np.maximum(lin, 1e-12)
    return 10 * np.log10(lin_safe)

def compute_ndvi(red: pd.Series, nir: pd.Series) -> pd.Series:
    # pre: red and nir bands
    # post: returns NDVI values
    # desc: computes Normalized Difference Vegetation Index

    num = nir - red
    den = nir + red
    return num / den.replace(0, np.nan)

def compute_ndmi(nir: pd.Series, swir1: pd.Series) -> pd.Series:
    # pre: nir and swir1 bands
    # post: returns NDMI values
    # desc: computes Normalized Difference Moisture Index

    return (nir - swir1) / (nir + swir1)

def compute_msi(swir1: pd.Series, nir: pd.Series) -> pd.Series:
    # pre: swir1 and nir bands
    # post: returns MSI values
    # desc: computes Moisture Stress Index

    return swir1 / nir.replace(0, np.nan)

def compute_vv_vh_ratio(vv: pd.Series, vh: pd.Series) -> pd.Series:
    # pre : vv and vh backscatter
    # post: returns VV/VH ratio
    # desc: computes the ratio of VV to VH backscatter

    return vv / vh.replace(0, np.nan)

def compute_backscatter_diff(vv_db: pd.Series, vh_db: pd.Series) -> pd.Series:
    # pre : vv and vh backscatter in dB
    # post: returns backscatter difference (VV - VH) in dB
    # desc: computes the difference between VV and VH backscatter in dB

    return vv_db - vh_db
