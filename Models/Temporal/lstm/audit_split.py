"""
Split audit for derived_8.0: temporal distribution analysis.

Checks whether the val->test gap (v9: 0.845 -> 0.747 R^2) is driven by
temporal distribution shift between train (2017-2020), val (2021-2022),
and test (2023-2025) periods.

Usage:
    python -m Models.Temporal.lstm.audit_split
"""

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data/splits/derived_8.0"
OUT_DIR  = Path(__file__).parent / "outputs_audit"
OUT_DIR.mkdir(exist_ok=True)

TARGET = "soil_moisture_5cm"

AUDIT_FEATURES = [
    TARGET,
    "SMAP_sm_am_interp",
    "SMAP_sm_pm_interp",
    "LST_modis",
    "F_NDVI",
    "F_NDMI",
    "precip_mm",
    "latitude",
    "elev",
]


# -- helpers ------------------------------------------------------------------

def dist_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {k: float("nan") for k in ["n", "mean", "std", "min", "p10", "p50", "p90", "max"]}
    return {
        "n":    int(len(s)),
        "mean": float(s.mean()),
        "std":  float(s.std()),
        "min":  float(s.min()),
        "p10":  float(s.quantile(0.10)),
        "p50":  float(s.quantile(0.50)),
        "p90":  float(s.quantile(0.90)),
        "max":  float(s.max()),
    }


def ood_fraction(test_series: pd.Series, train_series: pd.Series) -> float:
    """Fraction of test rows outside train [p1, p99]."""
    lo = train_series.quantile(0.01)
    hi = train_series.quantile(0.99)
    s = test_series.dropna()
    if len(s) == 0:
        return float("nan")
    return float(((s < lo) | (s > hi)).mean())


# -- main ---------------------------------------------------------------------

def main():
    lines = []  # audit report lines

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("SPLIT AUDIT -- derived_8.0")
    log("Train: 2017-2020 | Val: 2021-2022 | Test: 2023-2025")
    log("=" * 70)

    # -- load -----------------------------------------------------------------
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df   = pd.read_csv(DATA_DIR / "val.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")
    log(f"\n[load]  train={train_df.shape}  val={val_df.shape}  test={test_df.shape}")

    # -- 1. station overlap ---------------------------------------------------
    log("\n" + "-" * 70)
    log("1. STATION OVERLAP")
    log("-" * 70)

    tr_stations  = set(train_df["station_id"].unique())
    val_stations = set(val_df["station_id"].unique())
    te_stations  = set(test_df["station_id"].unique())

    log(f"  train stations : {len(tr_stations)}")
    log(f"  val   stations : {len(val_stations)}")
    log(f"  test  stations : {len(te_stations)}")
    log(f"  in train  intersect test: {len(tr_stations & te_stations)}")
    log(f"  only in test   : {te_stations - tr_stations}")
    log(f"  only in train  : {tr_stations - te_stations}")

    # rows per station per split
    rows_records = []
    all_stations = sorted(tr_stations | val_stations | te_stations)
    for sid in all_stations:
        rows_records.append({
            "station_id": sid,
            "train_rows": int((train_df["station_id"] == sid).sum()),
            "val_rows":   int((val_df["station_id"]   == sid).sum()),
            "test_rows":  int((test_df["station_id"]  == sid).sum()),
        })
    station_summary = pd.DataFrame(rows_records).set_index("station_id")
    station_summary.to_csv(OUT_DIR / "station_split_summary.csv")
    log(f"\n  Saved: station_split_summary.csv")
    log("\n  Rows per station (train / val / test):")
    log(station_summary.to_string())

    # -- 2. feature distribution stats ----------------------------------------
    log("\n" + "-" * 70)
    log("2. FEATURE DISTRIBUTION BY SPLIT")
    log("-" * 70)

    feat_records = []
    for feat in AUDIT_FEATURES:
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if feat not in df.columns:
                log(f"  [MISSING] {feat} in {split_name}")
                continue
            s = dist_stats(df[feat])
            s["feature"] = feat
            s["split"]   = split_name
            feat_records.append(s)

    feat_df = pd.DataFrame(feat_records).set_index(["feature", "split"])
    feat_df.to_csv(OUT_DIR / "feature_distribution_summary.csv")
    log(f"\n  Saved: feature_distribution_summary.csv\n")

    for feat in AUDIT_FEATURES:
        sub = feat_df.loc[feat] if feat in feat_df.index else None
        if sub is None:
            continue
        log(f"  {feat}:")
        for split in ["train", "val", "test"]:
            if split not in sub.index:
                continue
            r = sub.loc[split]
            log(f"    {split:5s}  mean={r['mean']:+.4f}  std={r['std']:.4f}  "
                f"[{r['p10']:.4f}, {r['p90']:.4f}]")

    # -- 3. per-station target shift ------------------------------------------
    log("\n" + "-" * 70)
    log("3. PER-STATION TARGET SHIFT (soil_moisture_5cm)")
    log("-" * 70)

    tr_means  = train_df.groupby("station_id")[TARGET].mean().rename("train_mean")
    val_means = val_df.groupby("station_id")[TARGET].mean().rename("val_mean")
    te_means  = test_df.groupby("station_id")[TARGET].mean().rename("test_mean")

    shift_df = pd.concat([tr_means, val_means, te_means], axis=1)
    shift_df["train_test_diff"]  = shift_df["test_mean"]  - shift_df["train_mean"]
    shift_df["train_test_ratio"] = shift_df["test_mean"]  / shift_df["train_mean"]
    shift_df["train_val_diff"]   = shift_df["val_mean"]   - shift_df["train_mean"]
    shift_df = shift_df.sort_values("train_test_diff", key=abs, ascending=False)
    shift_df.to_csv(OUT_DIR / "station_target_shift.csv")
    log(f"\n  Saved: station_target_shift.csv\n")
    log(shift_df.to_string(float_format=lambda x: f"{x:+.4f}"))

    mean_abs_diff = shift_df["train_test_diff"].abs().mean()
    n_shifted = (shift_df["train_test_diff"].abs() > 0.02).sum()
    log(f"\n  Mean |train-test diff|: {mean_abs_diff:.4f}  "
        f"Stations with |diff| > 0.02: {n_shifted}/{len(shift_df)}")

    # scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    common = shift_df.dropna(subset=["train_mean", "test_mean"])
    ax.scatter(common["train_mean"], common["test_mean"], alpha=0.8, s=60)
    for sid, row in common.iterrows():
        ax.annotate(str(sid)[:12], (row["train_mean"], row["test_mean"]),
                    fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    lo = min(common["train_mean"].min(), common["test_mean"].min()) - 0.01
    hi = max(common["train_mean"].max(), common["test_mean"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="identity")
    ax.set_xlabel("Train mean soil_moisture_5cm")
    ax.set_ylabel("Test mean soil_moisture_5cm")
    ax.set_title("Per-station target mean: train vs. test (derived_8.0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "target_shift_scatter.png", dpi=120)
    plt.close(fig)
    log(f"\n  Saved: target_shift_scatter.png")

    # -- 4. out-of-distribution feature fraction ------------------------------
    log("\n" + "-" * 70)
    log("4. OUT-OF-DISTRIBUTION FEATURE FRACTIONS (test vs. train [p1, p99])")
    log("-" * 70)

    ood_records = []
    for feat in AUDIT_FEATURES:
        if feat not in train_df.columns or feat not in test_df.columns:
            continue
        frac = ood_fraction(test_df[feat], train_df[feat])
        ood_records.append({"feature": feat, "ood_fraction": frac})
        log(f"  {feat:<30s}  {frac:.3f}  ({frac*100:.1f}% of test rows OOD)")

    ood_df = pd.DataFrame(ood_records).set_index("feature")
    ood_df.to_csv(OUT_DIR / "feature_ood_fraction.csv")
    log(f"\n  Saved: feature_ood_fraction.csv")

    # -- 5. temporal trend in target ------------------------------------------
    log("\n" + "-" * 70)
    log("5. TEMPORAL TREND IN TARGET (annual mean soil_moisture_5cm)")
    log("-" * 70)

    all_df = pd.concat(
        [train_df.assign(split="train"),
         val_df.assign(split="val"),
         test_df.assign(split="test")],
        ignore_index=True
    )
    if "date" in all_df.columns:
        all_df["year"] = pd.to_datetime(all_df["date"]).dt.year
        annual = all_df.groupby("year")[TARGET].agg(["mean", "std", "count"])
        log("\n  Year | mean  | std   | n")
        for yr, row in annual.iterrows():
            log(f"  {yr}  {row['mean']:+.4f}  {row['std']:.4f}  {int(row['count'])}")

    # -- summary --------------------------------------------------------------
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Split type         : TEMPORAL (same stations, different years)")
    log(f"  Train  intersect Test overlap: {len(tr_stations & te_stations)}/{len(te_stations)} stations")
    log(f"  Mean |target shift|: {mean_abs_diff:.4f} m^3/m^3 (train->test per station)")
    log(f"  Stations w/ large shift: {n_shifted} with |diff| > 0.02")
    log(f"\n  -> If target means shift substantially between train and test years,")
    log(f"    per-station normalization (v17) should reduce the systematic component.")
    log(f"  -> If feature OOD fractions are high, consider retraining with more")
    log(f"    recent data or applying domain adaptation.")

    # write report
    report_path = OUT_DIR / "audit_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] {report_path}")


if __name__ == "__main__":
    main()
