"""Feature-family classification kept local to the 2.0 experiment."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


DEFAULT_FAMILIES = ("satellite", "hydro", "static", "calendar", "temporal", "other")


def infer_family(feature_name: str) -> str:
    """Classify a feature using the existing project naming conventions."""
    name = str(feature_name)
    if name in {"DOY", "D_sin_DOY", "D_cos_DOY", "year_frac", "sin_year", "cos_year"}:
        return "calendar"
    if name.endswith("_x_year") or name.startswith(("D_sa_", "D_z_", "D_fft_")):
        return "calendar"
    if name == "precip_mm" or name.startswith("G_") or any(
        token in name for token in ("G_API", "G_DSLR", "G_rain", "precip")
    ):
        return "hydro"
    if name in {"elev", "slope", "aspect", "latitude", "longitude"} or name.startswith(
        ("J_", "K_")
    ):
        return "static"
    if name.startswith(("SMAP", "s1_", "s2_", "LST", "E_", "F_", "lia_", "I_ts_")):
        return "satellite"
    satellite_bases = (
        "SMAP",
        "LST_modis",
        "E_SAR",
        "F_NDVI",
        "F_NDMI",
        "F_MSI",
        "s2_b",
        "s1_",
    )
    if any(base in name for base in satellite_bases):
        return "satellite"
    if name.startswith(("A_d_", "A_grad_", "A_pct_", "C_lag_", "C_smm_", "V_")):
        return "temporal"
    return "other"


def family_counts(features: Iterable[str]) -> dict[str, int]:
    """Return stable family counts for audit tables."""
    counts = Counter(infer_family(feature) for feature in features)
    return {family: int(counts.get(family, 0)) for family in DEFAULT_FAMILIES}

