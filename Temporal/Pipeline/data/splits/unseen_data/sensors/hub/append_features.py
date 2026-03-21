# Jakob Balkovec

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
SENSORS_DIR = THIS_DIR.parent
UNSEEN_DIR = SENSORS_DIR.parent
SPLITS_DIR = UNSEEN_DIR.parent
DERIVED6_UTILS_DIR = SPLITS_DIR / "derived_6.0" / "utils"
DERIVED7_META = SPLITS_DIR / "derived_7.0" / "split_meta.json"
DERIVED8_LIA = SPLITS_DIR / "derived_8.0" / "LIA" / "stations_lia.csv"

if str(DERIVED6_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(DERIVED6_UTILS_DIR))

from derived_features_all_math import (
    add_smap_features,
    compute_api,
    compute_days_since_last_rain,
    compute_ndmi,
    compute_ndvi,
    compute_msi,
    compute_rain_sums_days,
    compute_sar_diff,
    compute_sar_ratio,
    compute_time_since_last_spike_past_only,
    ema,
    rolling_corr,
    rolling_fft_dom_freq_and_entropy,
    rolling_max,
    rolling_mean,
    rolling_mean_abs_change,
    rolling_min,
    rolling_range,
    rolling_std,
    rolling_cv,
    series_diffs,
    series_gradient_kobs,
    series_lags,
    series_pct_change,
    smm_index,
    train_only_monthly_anomaly_global,
    train_only_monthly_zscore_global,
)


DATE_COL = "date"
GROUP_COL = "station_id"
TARGET_COL = "soil_moisture_5cm"

BASE_COLS = [
    "precip_mm",
    "s1_vv",
    "s1_vh",
    "s2_b4",
    "s2_b8",
    "s2_b11",
    "s2_b12",
    "LST_modis",
    "elev",
    "slope",
    "aspect",
    "DOY",
    "SMAP_sm_am_interp",
    "SMAP_sm_pm_interp",
]

EPS = 1e-6
RAIN_THR_MM = 0.5
API_DECAY = 0.90
SMM_ALPHA = 0.85

KOBS_LONG = 5
FFT_WIN = 30
WIN_OBS_7 = 7
WIN_OBS_14 = 14
RAIN_SUM_DAYS = (3, 7, 30)

SPIKE_COL = "s1_vv"
SPIKE_Z_THR = 2.0

PFX = {
    "META": "M",
    "BASE": "B",
    "MET": "G",
    "RAD": "E",
    "OPT": "F",
    "DYN": "A",
    "VOL": "V",
    "MEM": "C",
    "SEA": "D",
    "EVT": "I",
}

DRIFT_COLS = ["year", "year_frac", "sin_year", "cos_year", "API_x_year", "SMAP_x_year"]
LIA_COLS = ["lia_mean_asc_deg", "lia_std_asc_deg", "lia_mean_desc_deg", "lia_std_desc_deg"]
SUPPORTED_DEVICES = ("d3", "d4", "d7")
DEVICE_DEFAULTS = {
    "d3": {
        "input": SENSORS_DIR / "d3" / "final.csv",
        "out_dir": SENSORS_DIR / "d3",
        "station_id": "DEV3",
    },
    "d4": {
        "input": SENSORS_DIR / "d4" / "final.csv",
        "out_dir": SENSORS_DIR / "d4",
        "station_id": "DEV4",
    },
    "d7": {
        "input": SENSORS_DIR / "d7" / "final.csv",
        "out_dir": SENSORS_DIR / "d7",
        "station_id": "DEV7",
    },
}


def _default_input() -> Path:
    for device in SUPPORTED_DEVICES:
        p = DEVICE_DEFAULTS[device]["input"]
        if p.exists():
            return p
    return SENSORS_DIR / "d3" / "final.csv"


def _read_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    return line.split(",") if line else []


def _attach_cols(df: pd.DataFrame, cols: dict[str, pd.Series]) -> pd.DataFrame:
    if not cols:
        return df
    block = pd.DataFrame(cols, index=df.index)
    return pd.concat([df, block], axis=1)


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def _prepare_input(df: pd.DataFrame, station_override: str | None = None) -> pd.DataFrame:
    out = df.copy()

    if DATE_COL not in out.columns:
        raise ValueError(f"Input must include '{DATE_COL}'")

    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    bad_dates = int(out[DATE_COL].isna().sum())
    if bad_dates:
        out = out.dropna(subset=[DATE_COL]).copy()
        print(f"[WARN] Dropped {bad_dates} rows with invalid dates.")

    if GROUP_COL not in out.columns:
        sid = station_override or "UNSEEN_STATION"
        out[GROUP_COL] = sid

    if station_override:
        out[GROUP_COL] = station_override

    out[GROUP_COL] = out[GROUP_COL].astype(str)

    if "DOY" not in out.columns:
        out["DOY"] = out[DATE_COL].dt.dayofyear

    if "SMAP_sm_am_interp" not in out.columns and "SMAP_sm_am" in out.columns:
        out["SMAP_sm_am_interp"] = pd.to_numeric(out["SMAP_sm_am"], errors="coerce")
    if "SMAP_sm_pm_interp" not in out.columns and "SMAP_sm_pm" in out.columns:
        out["SMAP_sm_pm_interp"] = pd.to_numeric(out["SMAP_sm_pm"], errors="coerce")

    need = [GROUP_COL, DATE_COL, TARGET_COL] + BASE_COLS
    out = _ensure_columns(out, need)

    for c in BASE_COLS + [TARGET_COL]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)
    return out


def _build_derived6(df: pd.DataFrame) -> pd.DataFrame:
    full = df.copy()
    full[DATE_COL] = pd.to_datetime(full[DATE_COL], errors="coerce")

    full[f"{PFX['SEA']}_sin_DOY"] = np.sin(2 * np.pi * full["DOY"] / 365.0)
    full[f"{PFX['SEA']}_cos_DOY"] = np.cos(2 * np.pi * full["DOY"] / 365.0)

    full[f"{PFX['OPT']}_NDVI"] = compute_ndvi(full, nir_col="s2_b8", red_col="s2_b4", eps=EPS)
    full[f"{PFX['OPT']}_NDMI"] = compute_ndmi(full, nir_col="s2_b8", swir_col="s2_b11", eps=EPS)
    full[f"{PFX['OPT']}_MSI"] = compute_msi(full, swir_col="s2_b11", nir_col="s2_b8", eps=EPS)

    full[f"{PFX['RAD']}_SAR_ratio"] = compute_sar_ratio(full, vv_col="s1_vv", vh_col="s1_vh", eps=EPS)
    full[f"{PFX['RAD']}_SAR_diff"] = compute_sar_diff(full, vv_col="s1_vv", vh_col="s1_vh")

    full[f"{PFX['MET']}_API"] = compute_api(
        full, precip_col="precip_mm", decay=API_DECAY, group_col=GROUP_COL, date_col=DATE_COL
    )
    full[f"{PFX['MET']}_DSLR"] = compute_days_since_last_rain(
        full, precip_col="precip_mm", threshold_mm=RAIN_THR_MM, group_col=GROUP_COL, date_col=DATE_COL
    )
    for d in RAIN_SUM_DAYS:
        full[f"{PFX['MET']}_rain_sum_{d}d"] = compute_rain_sums_days(
            full, precip_col="precip_mm", window_days=d, group_col=GROUP_COL, date_col=DATE_COL
        )

    full = add_smap_features(
        full,
        group_col=GROUP_COL,
        date_col=DATE_COL,
        imputed=True,
        make_combined=True,
        combined_col="SMAP_sm_interp",
        lags=(1, 7, 30),
        roll_windows=(7, 30),
        ema_alpha=0.2,
        add_ampm_diff=True,
    )

    diff_kobs = [1, 2, 5, 7, 14, 30]
    grad_kobs = [WIN_OBS_7, WIN_OBS_14, 30]
    lag_kobs = [1, 2, 5, 6, 12, 30]
    wins = [WIN_OBS_7, WIN_OBS_14, 30]
    corr_wins = [WIN_OBS_7, WIN_OBS_14]

    dyn_signals = {
        f"{PFX['MET']}_API": f"{PFX['MET']}_API",
        f"{PFX['OPT']}_NDMI": f"{PFX['OPT']}_NDMI",
        f"{PFX['RAD']}_SAR_ratio": f"{PFX['RAD']}_SAR_ratio",
        "LST_modis": "LST_modis",
        f"{PFX['OPT']}_NDVI": f"{PFX['OPT']}_NDVI",
        f"{PFX['RAD']}_SAR_diff": f"{PFX['RAD']}_SAR_diff",
        "s2_b11": "s2_b11",
        "s2_b12": "s2_b12",
    }
    if "SMAP_sm_interp" in full.columns:
        dyn_signals["SMAP_sm_interp"] = "SMAP_sm_interp"

    for col in dyn_signals.values():
        new_cols: dict[str, pd.Series] = {}

        for k in diff_kobs:
            new_cols[f"{PFX['DYN']}_d_{col}_kobs{k}"] = series_diffs(
                full, col=col, kobs=k, group_col=GROUP_COL, date_col=DATE_COL
            )
        for k in grad_kobs:
            new_cols[f"{PFX['DYN']}_grad_{col}_kobs{k}"] = series_gradient_kobs(
                full, col=col, kobs=k, group_col=GROUP_COL, date_col=DATE_COL
            )
        new_cols[f"{PFX['DYN']}_pct_{col}"] = series_pct_change(
            full, col=col, group_col=GROUP_COL, date_col=DATE_COL, eps=EPS
        )

        for w in wins:
            new_cols[f"{PFX['VOL']}_rollstd_{col}_kobs{w}"] = rolling_std(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, ddof=0, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_rollrng_{col}_kobs{w}"] = rolling_range(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_rollcv_{col}_kobs{w}"] = rolling_cv(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, eps=EPS, ddof=0, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_rollmean_{col}_kobs{w}"] = rolling_mean(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_rollmin_{col}_kobs{w}"] = rolling_min(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_rollmax_{col}_kobs{w}"] = rolling_max(
                full, col=col, window=w, group_col=GROUP_COL, date_col=DATE_COL, min_periods=w
            )
            new_cols[f"{PFX['VOL']}_ema_{col}_kobs{w}"] = ema(
                full, col=col, alpha=2.0 / (w + 1.0), group_col=GROUP_COL, date_col=DATE_COL
            )

        for k in lag_kobs:
            new_cols[f"{PFX['MEM']}_lag_{col}_kobs{k}"] = series_lags(
                full, col=col, lag_kobs=k, group_col=GROUP_COL, date_col=DATE_COL
            )

        new_cols[f"{PFX['MEM']}_smm_{col}_alpha{SMM_ALPHA}_n{KOBS_LONG}"] = smm_index(
            full, col=col, alpha=SMM_ALPHA, n_lags=KOBS_LONG, group_col=GROUP_COL, date_col=DATE_COL
        )

        full = _attach_cols(full, new_cols)

    rad_cols: dict[str, pd.Series] = {}
    rad_cols[f"{PFX['RAD']}_dVV_1"] = series_diffs(full, col="s1_vv", kobs=1, group_col=GROUP_COL, date_col=DATE_COL)
    for w in corr_wins:
        rad_cols[f"{PFX['RAD']}_rough_s1_vv_kobs{w}"] = rolling_mean_abs_change(
            full, col="s1_vv", window=w, group_col=GROUP_COL, date_col=DATE_COL, past_only=True, min_periods=w
        )
        rad_cols[f"{PFX['RAD']}_rough_s1_vh_kobs{w}"] = rolling_mean_abs_change(
            full, col="s1_vh", window=w, group_col=GROUP_COL, date_col=DATE_COL, past_only=True, min_periods=w
        )
    full = _attach_cols(full, rad_cols)

    full[f"{PFX['EVT']}_ts_spike_{SPIKE_COL}"] = compute_time_since_last_spike_past_only(
        full, diff_col=f"{PFX['RAD']}_dVV_1", zthr=SPIKE_Z_THR, group_col=GROUP_COL, date_col=DATE_COL, eps=EPS
    )

    corr_cols: dict[str, pd.Series] = {}
    for w in corr_wins:
        corr_cols[f"H_corr_{PFX['RAD']}_SAR_ratio__{PFX['OPT']}_NDMI_kobs{w}"] = rolling_corr(
            full,
            x_col=f"{PFX['RAD']}_SAR_ratio",
            y_col=f"{PFX['OPT']}_NDMI",
            window=w,
            group_col=GROUP_COL,
            date_col=DATE_COL,
            min_periods=w,
            past_only=True,
        )
        corr_cols[f"H_corr_LST_modis__{PFX['OPT']}_NDMI_kobs{w}"] = rolling_corr(
            full,
            x_col="LST_modis",
            y_col=f"{PFX['OPT']}_NDMI",
            window=w,
            group_col=GROUP_COL,
            date_col=DATE_COL,
            min_periods=w,
            past_only=True,
        )
    full = _attach_cols(full, corr_cols)

    for col in [f"{PFX['OPT']}_NDMI", f"{PFX['RAD']}_SAR_ratio", "LST_modis"]:
        full[f"{PFX['SEA']}_sa_{col}"] = train_only_monthly_anomaly_global(full, full, col=col, date_col=DATE_COL).values
        full[f"{PFX['SEA']}_z_{col}"] = train_only_monthly_zscore_global(
            full, full, col=col, date_col=DATE_COL, eps=EPS
        ).values

    fft_cols: dict[str, pd.Series] = {}
    for sig in [f"{PFX['OPT']}_NDMI", f"{PFX['RAD']}_SAR_ratio", "LST_modis"]:
        dom, ent = rolling_fft_dom_freq_and_entropy(
            full, col=sig, window=FFT_WIN, group_col=GROUP_COL, date_col=DATE_COL, past_only=True, eps=1e-12
        )
        fft_cols[f"{PFX['SEA']}_fft_dom_{sig}_kobs{FFT_WIN}"] = dom
        fft_cols[f"{PFX['SEA']}_fft_ent_{sig}_kobs{FFT_WIN}"] = ent
    full = _attach_cols(full, fft_cols)

    dslr_col = f"{PFX['MET']}_DSLR"
    full[f"{PFX['MET']}_DSLR_isnan"] = full[dslr_col].isna().astype(int)
    return full


def _load_drift_ref(meta_path: Path) -> tuple[float | None, float | None]:
    if not meta_path.exists():
        return None, None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vals = meta.get("drift_year_scaling_train_minmax")
        if isinstance(vals, list) and len(vals) == 2:
            return float(vals[0]), float(vals[1])
    except Exception:
        return None, None
    return None, None


def _add_drift_features(df: pd.DataFrame, ref_min_year: float | None, ref_max_year: float | None) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out["year"] = out[DATE_COL].dt.year.astype(float)

    if ref_min_year is None:
        ref_min_year = float(out["year"].min())
    if ref_max_year is None:
        ref_max_year = float(out["year"].max())

    denom = (ref_max_year - ref_min_year) if (ref_max_year > ref_min_year) else 1.0
    out["year_frac"] = (out["year"] - ref_min_year) / denom

    theta = 2 * np.pi * out["year_frac"]
    out["sin_year"] = np.sin(theta)
    out["cos_year"] = np.cos(theta)

    out["API_x_year"] = out["G_API"] * out["year_frac"] if "G_API" in out.columns else np.nan
    smap_col = "SMAP_sm_pm_interp_ema02"
    out["SMAP_x_year"] = out[smap_col] * out["year_frac"] if smap_col in out.columns else np.nan

    return out


def _add_lia_features(
    df: pd.DataFrame,
    lia_csv: Path,
    default_lia: tuple[float | None, float | None, float | None, float | None],
) -> pd.DataFrame:
    out = df.copy()
    out[GROUP_COL] = out[GROUP_COL].astype(str)

    for c in LIA_COLS:
        if c not in out.columns:
            out[c] = np.nan

    if lia_csv.exists():
        lia = pd.read_csv(lia_csv)
        need = [GROUP_COL] + LIA_COLS
        missing = [c for c in need if c not in lia.columns]
        if missing:
            print(f"[WARN] LIA CSV missing columns {missing}; keeping LIA features as NaN/defaults.")
        else:
            lia = lia[need].copy()
            lia[GROUP_COL] = lia[GROUP_COL].astype(str)
            out = out.drop(columns=LIA_COLS).merge(lia, on=GROUP_COL, how="left")

    a_mean, a_std, d_mean, d_std = default_lia
    defaults = {
        "lia_mean_asc_deg": a_mean,
        "lia_std_asc_deg": a_std,
        "lia_mean_desc_deg": d_mean,
        "lia_std_desc_deg": d_std,
    }
    for c, v in defaults.items():
        if v is not None:
            out[c] = out[c].fillna(float(v))

    return out


def _align_to_reference(df: pd.DataFrame, reference_csv: Path, keep_extra: bool) -> pd.DataFrame:
    ref_cols = _read_header(reference_csv)
    if not ref_cols:
        return df

    out = df.copy()
    for c in ref_cols:
        if c not in out.columns:
            out[c] = np.nan

    if keep_extra:
        extras = [c for c in out.columns if c not in ref_cols]
        return out[ref_cols + extras]

    return out[ref_cols]


def _build_one(args: argparse.Namespace, in_path: Path, out_dir: Path, station_id: str | None, device: str | None) -> None:
    in_path = in_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df0 = pd.read_csv(in_path, low_memory=False)
    df0 = _prepare_input(df0, station_override=station_id)

    prefix = f"[{device}] " if device else ""
    print(f"{prefix}Loaded input: {in_path}")
    print(f"Rows: {len(df0)} | Cols: {len(df0.columns)}")

    d6 = _build_derived6(df0)
    d6 = _align_to_reference(d6, Path(args.ref_derived6).resolve(), keep_extra=args.keep_extra)
    d6_path = out_dir / "unseen_derived_6.csv"
    d6.to_csv(d6_path, index=False)

    y0, y1 = _load_drift_ref(Path(args.drift_meta).resolve())
    d7 = _add_drift_features(d6, ref_min_year=y0, ref_max_year=y1)
    d7 = _align_to_reference(d7, Path(args.ref_derived7).resolve(), keep_extra=args.keep_extra)
    d7_path = out_dir / "unseen_derived_7.csv"
    d7.to_csv(d7_path, index=False)

    d8 = _add_lia_features(
        d7,
        lia_csv=Path(args.lia_csv).resolve(),
        default_lia=(args.lia_mean_asc_deg, args.lia_std_asc_deg, args.lia_mean_desc_deg, args.lia_std_desc_deg),
    )
    d8 = _align_to_reference(d8, Path(args.ref_derived8).resolve(), keep_extra=args.keep_extra)
    d8_path = out_dir / "unseen_derived_8.csv"
    d8.to_csv(d8_path, index=False)

    summary = {
        "device": device,
        "input_csv": str(in_path),
        "output_dir": str(out_dir),
        "rows": int(len(d8)),
        "columns": {
            "derived_6": int(len(d6.columns)),
            "derived_7": int(len(d7.columns)),
            "derived_8": int(len(d8.columns)),
        },
        "drift_ref_year_minmax": [y0, y1],
        "lia_csv": str(Path(args.lia_csv).resolve()),
        "lia_missing_rate": {
            c: float(d8[c].isna().mean()) if c in d8.columns else None for c in LIA_COLS
        },
    }
    meta_path = out_dir / "unseen_meta.json"
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  {d6_path}")
    print(f"  {d7_path}")
    print(f"  {d8_path}")
    print(f"  {meta_path}")
    print("\nColumn counts:")
    print(f"  derived_6: {len(d6.columns)}")
    print(f"  derived_7: {len(d7.columns)}")
    print(f"  derived_8: {len(d8.columns)}")


def _resolve_targets(args: argparse.Namespace) -> list[dict[str, Path | str | None]]:
    if args.device == "all":
        if args.input or args.out_dir or args.station_id:
            raise ValueError("--device all cannot be combined with --input, --out-dir, or --station-id")
        targets: list[dict[str, Path | str | None]] = []
        for device in SUPPORTED_DEVICES:
            defaults = DEVICE_DEFAULTS[device]
            targets.append(
                {
                    "device": device,
                    "input": Path(defaults["input"]),
                    "out_dir": Path(defaults["out_dir"]),
                    "station_id": str(defaults["station_id"]),
                }
            )
        return targets

    if args.device in SUPPORTED_DEVICES:
        defaults = DEVICE_DEFAULTS[args.device]
        return [
            {
                "device": args.device,
                "input": Path(args.input).resolve() if args.input else Path(defaults["input"]),
                "out_dir": Path(args.out_dir).resolve() if args.out_dir else Path(defaults["out_dir"]),
                "station_id": args.station_id if args.station_id is not None else str(defaults["station_id"]),
            }
        ]

    input_path = Path(args.input).resolve() if args.input else _default_input().resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else input_path.parent.resolve()
    return [{"device": None, "input": input_path, "out_dir": out_dir, "station_id": args.station_id}]


def build(args: argparse.Namespace) -> None:
    targets = _resolve_targets(args)
    for t in targets:
        _build_one(
            args=args,
            in_path=Path(t["input"]),
            out_dir=Path(t["out_dir"]),
            station_id=t["station_id"] if isinstance(t["station_id"], str) else None,
            device=t["device"] if isinstance(t["device"], str) else None,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build derived split-style features for unseen data.",
        epilog=(
            "Examples:\n"
            "  python append_features.py --device d3\n"
            "  python append_features.py --device all\n"
            "  python append_features.py --input /path/to/final.csv --out-dir /path/to/out"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device",
        choices=[*SUPPORTED_DEVICES, "all"],
        default=None,
        help="Use defaults for a device folder (d3, d4, d7) or run all three with 'all'",
    )
    parser.add_argument("--input", default=None, help="Input final.csv for unseen station")
    parser.add_argument("--out-dir", default=None, help="Output folder for unseen derived files")
    parser.add_argument("--station-id", default=None, help="Optional override for station_id on all rows")
    parser.add_argument("--keep-extra", action="store_true", help="Keep columns not present in reference split schema")

    parser.add_argument(
        "--ref-derived6",
        default=str(SPLITS_DIR / "derived_6.0" / "train.csv"),
        help="Reference CSV schema for derived_6 output",
    )
    parser.add_argument(
        "--ref-derived7",
        default=str(SPLITS_DIR / "derived_7.0" / "train.csv"),
        help="Reference CSV schema for derived_7 output",
    )
    parser.add_argument(
        "--ref-derived8",
        default=str(SPLITS_DIR / "derived_8.0" / "train.csv"),
        help="Reference CSV schema for derived_8 output",
    )
    parser.add_argument(
        "--drift-meta",
        default=str(DERIVED7_META),
        help="derived_7.0 split_meta.json used for drift year scaling reference",
    )
    parser.add_argument(
        "--lia-csv",
        default=str(DERIVED8_LIA),
        help="LIA CSV for derived_8 features",
    )

    parser.add_argument("--lia-mean-asc-deg", type=float, default=None, help="Optional fallback for lia_mean_asc_deg")
    parser.add_argument("--lia-std-asc-deg", type=float, default=None, help="Optional fallback for lia_std_asc_deg")
    parser.add_argument("--lia-mean-desc-deg", type=float, default=None, help="Optional fallback for lia_mean_desc_deg")
    parser.add_argument("--lia-std-desc-deg", type=float, default=None, help="Optional fallback for lia_std_desc_deg")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
