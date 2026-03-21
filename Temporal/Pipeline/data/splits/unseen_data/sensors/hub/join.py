from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
SENSORS_DIR = THIS_DIR.parent

SUPPORTED_DEVICES = ("d3", "d4", "d7")
DEVICE_TO_STATION = {"d3": "DEV3", "d4": "DEV4", "d7": "DEV7"}

STATION_COL = "station_id"
DATE_COL = "date"
AGG_MOISTURE_COL = "MoistureDailyAvg (%)"
SOIL_COL = "soil_moisture_5cm"

DEVICE_DEFAULTS = {
    "d3": {
        "aggregated": SENSORS_DIR / "d3" / "aggregated_d3.csv",
        "derived": SENSORS_DIR / "d3" / "derived_d3.csv",
        "output": SENSORS_DIR / "d3" / "final_d3.csv",
    },
    "d4": {
        "aggregated": SENSORS_DIR / "d4" / "aggregated_d4.csv",
        "derived": SENSORS_DIR / "d4" / "derived_d4.csv",
        "output": SENSORS_DIR / "d4" / "final_d4.csv",
    },
    "d7": {
        "aggregated": SENSORS_DIR / "d7" / "aggregated_d7.csv",
        "derived": SENSORS_DIR / "d7" / "derived_d7.csv",
        "output": SENSORS_DIR / "d7" / "final_d7.csv",
    },
}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


def _normalize_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.date.astype("string")
    return out


def _prepare_aggregated(df: pd.DataFrame, device: str) -> pd.DataFrame:
    out = _clean_columns(df)

    if STATION_COL not in out.columns and "DevEUI" in out.columns:
        out[STATION_COL] = DEVICE_TO_STATION[device]

    if DATE_COL not in out.columns and "Timestamp (UTC)" in out.columns:
        out[DATE_COL] = pd.to_datetime(out["Timestamp (UTC)"], errors="coerce").dt.date.astype("string")

    if STATION_COL not in out.columns:
        raise ValueError(f"Aggregated CSV is missing '{STATION_COL}'")
    if DATE_COL not in out.columns:
        raise ValueError(f"Aggregated CSV is missing '{DATE_COL}' (or 'Timestamp (UTC)')")

    if AGG_MOISTURE_COL not in out.columns:
        if "Moisture (%)" in out.columns:
            out[AGG_MOISTURE_COL] = pd.to_numeric(out["Moisture (%)"], errors="coerce")
        else:
            raise ValueError(f"Aggregated CSV is missing '{AGG_MOISTURE_COL}'")
    else:
        out[AGG_MOISTURE_COL] = pd.to_numeric(out[AGG_MOISTURE_COL], errors="coerce")

    out = _normalize_date_col(out, DATE_COL)
    out[STATION_COL] = out[STATION_COL].astype(str)
    out = out.dropna(subset=[STATION_COL, DATE_COL, AGG_MOISTURE_COL]).copy()

    out = (
        out.groupby([STATION_COL, DATE_COL], as_index=False)[AGG_MOISTURE_COL]
        .mean()
        .sort_values([STATION_COL, DATE_COL])
        .reset_index(drop=True)
    )
    return out


def _prepare_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = _clean_columns(df)
    if STATION_COL not in out.columns:
        raise ValueError(f"Derived CSV is missing '{STATION_COL}'")
    if DATE_COL not in out.columns:
        raise ValueError(f"Derived CSV is missing '{DATE_COL}'")

    out = _normalize_date_col(out, DATE_COL)
    out[STATION_COL] = out[STATION_COL].astype(str)
    out = out.dropna(subset=[STATION_COL, DATE_COL]).copy()
    return out


def _join_one(device: str, aggregated_path: Path, derived_path: Path, output_path: Path, how: str, station_only: bool) -> None:
    if not aggregated_path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {aggregated_path}")
    if not derived_path.exists():
        raise FileNotFoundError(f"Derived file not found: {derived_path}")

    agg_raw = pd.read_csv(aggregated_path, low_memory=False, encoding="utf-8-sig")
    drv_raw = pd.read_csv(derived_path, low_memory=False, encoding="utf-8-sig")

    agg = _prepare_aggregated(agg_raw, device=device)
    drv = _prepare_derived(drv_raw)

    keys = [STATION_COL]
    if (not station_only) and (DATE_COL in drv.columns) and (DATE_COL in agg.columns):
        keys.append(DATE_COL)

    joined = drv.merge(agg, on=keys, how=how)
    sort_cols = [c for c in (STATION_COL, DATE_COL) if c in joined.columns]
    if sort_cols:
        joined = joined.sort_values(sort_cols).reset_index(drop=True)

    updated_rows = 0
    if AGG_MOISTURE_COL in joined.columns:
        joined[AGG_MOISTURE_COL] = pd.to_numeric(joined[AGG_MOISTURE_COL], errors="coerce")
        updated_rows = int(joined[AGG_MOISTURE_COL].notna().sum())
        if SOIL_COL in joined.columns:
            joined[SOIL_COL] = joined[AGG_MOISTURE_COL].combine_first(pd.to_numeric(joined[SOIL_COL], errors="coerce"))
        else:
            joined[SOIL_COL] = joined[AGG_MOISTURE_COL]
        joined = joined.drop(columns=[AGG_MOISTURE_COL])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_path, index=False)

    print(f"[{device}] aggregated: {aggregated_path}")
    print(f"[{device}] derived: {derived_path}")
    print(f"[{device}] output: {output_path}")
    print(f"[{device}] join keys: {keys} | how: {how}")
    print(f"[{device}] rows where '{SOIL_COL}' set from aggregated daily average: {updated_rows}")
    print(f"[{device}] rows -> derived: {len(drv)} | aggregated: {len(agg)} | final: {len(joined)}")


def _resolve_targets(args: argparse.Namespace) -> list[tuple[str, Path, Path, Path]]:
    if args.device == "all":
        if args.aggregated or args.derived or args.output:
            raise ValueError("--device all cannot be combined with --aggregated, --derived, or --output")
        return [
            (
                d,
                Path(DEVICE_DEFAULTS[d]["aggregated"]),
                Path(DEVICE_DEFAULTS[d]["derived"]),
                Path(DEVICE_DEFAULTS[d]["output"]),
            )
            for d in SUPPORTED_DEVICES
        ]

    defaults = DEVICE_DEFAULTS[args.device]
    aggregated = Path(args.aggregated).resolve() if args.aggregated else Path(defaults["aggregated"])
    derived = Path(args.derived).resolve() if args.derived else Path(defaults["derived"])
    output = Path(args.output).resolve() if args.output else Path(defaults["output"])
    return [(args.device, aggregated, derived, output)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join aggregated_d[n] with derived_d[n], write final_d[n], and "
            "populate soil_moisture_5cm from MoistureDailyAvg (%)."
        ),
        epilog=(
            "Examples:\n"
            "  python join.py --device d3\n"
            "  python join.py --device d7\n"
            "  python join.py --device all\n"
            "  python join.py --device d4 --how inner\n"
            "  python join.py --device d3 --station-only"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", choices=[*SUPPORTED_DEVICES, "all"], required=True, help="Device key to process")
    parser.add_argument("--aggregated", default=None, help="Optional override for aggregated_d[n].csv")
    parser.add_argument("--derived", default=None, help="Optional override for derived_d[n].csv")
    parser.add_argument("--output", default=None, help="Optional override for final_d[n].csv output path")
    parser.add_argument("--how", choices=["left", "inner", "right", "outer"], default="left", help="Pandas merge mode")
    parser.add_argument("--station-only", action="store_true", help="Join on station_id only (can create duplicates)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = _resolve_targets(args)
    for device, aggregated_path, derived_path, output_path in targets:
        _join_one(
            device=device,
            aggregated_path=aggregated_path,
            derived_path=derived_path,
            output_path=output_path,
            how=args.how,
            station_only=args.station_only,
        )


if __name__ == "__main__":
    main()
