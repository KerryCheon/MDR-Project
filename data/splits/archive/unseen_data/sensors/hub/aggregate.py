from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
SENSORS_DIR = THIS_DIR.parent

SUPPORTED_DEVICES = ("d3", "d4", "d7")
TIMESTAMP_COL = "Timestamp (UTC)"
MOISTURE_COL = "Moisture (%)"
DEVICE_COL = "DevEUI"

DEVICE_DEFAULTS = {
    "d3": {
        "input": SENSORS_DIR / "d3" / "moisture_data_d3.csv",
        "output": SENSORS_DIR / "d3" / "aggregated_d3.csv",
    },
    "d4": {
        "input": SENSORS_DIR / "d4" / "moisture_data_d4.csv",
        "output": SENSORS_DIR / "d4" / "aggregated_d4.csv",
    },
    "d7": {
        "input": SENSORS_DIR / "d7" / "moisture_data_d7.csv",
        "output": SENSORS_DIR / "d7" / "aggregated_d7.csv",
    },
}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


def _aggregate_to_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = _clean_columns(df)

    required = {DEVICE_COL, TIMESTAMP_COL, MOISTURE_COL}
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    out[TIMESTAMP_COL] = pd.to_datetime(out[TIMESTAMP_COL], errors="coerce")
    out[MOISTURE_COL] = pd.to_numeric(out[MOISTURE_COL], errors="coerce")

    bad_rows = int((out[TIMESTAMP_COL].isna() | out[MOISTURE_COL].isna()).sum())
    out = out.dropna(subset=[TIMESTAMP_COL, MOISTURE_COL]).copy()

    out["date"] = out[TIMESTAMP_COL].dt.date

    daily = (
        out.groupby([DEVICE_COL, "date"], as_index=False)[MOISTURE_COL]
        .mean()
        .rename(columns={MOISTURE_COL: "MoistureDailyAvg (%)"})
        .sort_values([DEVICE_COL, "date"])
        .reset_index(drop=True)
    )
    return daily, bad_rows


def _aggregate_device(device: str, input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False, encoding="utf-8-sig")
    daily, bad_rows = _aggregate_to_daily(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)

    print(f"[{device}] input: {input_path}")
    print(f"[{device}] output: {output_path}")
    print(f"[{device}] rows in: {len(df)} | rows out: {len(daily)}")
    if bad_rows:
        print(f"[{device}] dropped rows with invalid timestamp/moisture: {bad_rows}")


def _resolve_targets(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.device == "all":
        if args.input or args.output:
            raise ValueError("--device all cannot be combined with --input or --output")
        return [
            (device, Path(DEVICE_DEFAULTS[device]["input"]), Path(DEVICE_DEFAULTS[device]["output"]))
            for device in SUPPORTED_DEVICES
        ]

    defaults = DEVICE_DEFAULTS[args.device]
    in_path = Path(args.input).resolve() if args.input else Path(defaults["input"])
    out_path = Path(args.output).resolve() if args.output else Path(defaults["output"])
    return [(args.device, in_path, out_path)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate sensor moisture time-series to daily means.",
        epilog=(
            "Examples:\n"
            "  python aggregate.py --device d3\n"
            "  python aggregate.py --device d7\n"
            "  python aggregate.py --device all\n"
            "  python aggregate.py --device d4 --input /path/to/raw.csv --output /path/to/aggregated_d4.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", choices=[*SUPPORTED_DEVICES, "all"], required=True, help="Device key to process")
    parser.add_argument("--input", default=None, help="Optional override input CSV (single-device mode only)")
    parser.add_argument("--output", default=None, help="Optional override output CSV (single-device mode only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = _resolve_targets(args)
    for device, input_path, output_path in targets:
        _aggregate_device(device=device, input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
