"""Collects test-split metrics (r2/rmse/ubrmse/bias/mae/q90/n) across milestone
LSTM model versions (v9 -> v23) for the standup's version-comparison table."""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent

RUNS = [
    ("v9", "30 feat, seq10", HERE / "outputs_v9/metrics.json", ["test"]),
    ("v20", "top25/seq30", HERE / "outputs_v20/window_sweep/top25/seq30/metrics.json", ["test"]),
    ("v21", "full58/seq10", HERE / "outputs_v21/full58/metrics.json", ["test"]),
    ("v22", "top25/seq30, SMAP z-scored", HERE / "outputs_v22_smap_norm/top25_seq30/metrics.json", ["test"]),
    (
        "v23 ensemble",
        "top25/seq30, 5-seed avg",
        HERE / "outputs_v23_baseline/ensemble_metrics.json",
        ["top25_seq30", "ensemble_test_eval", "overall"],
    ),
]

SEED_FILES = sorted((HERE / "outputs_v23_baseline/top25_seq30").glob("seed*/metrics.json"))

FIELDS = ["r2", "rmse", "ubrmse", "bias", "mae", "q90", "n"]


def load(path, key_path):
    d = json.loads(path.read_text())
    for k in key_path:
        d = d[k]
    return d


def main():
    rows = []
    for label, config, path, key_path in RUNS:
        m = load(path, key_path)
        rows.append((label, config, {f: m[f] for f in FIELDS}))

    seed_metrics = [load(p, ["test"]) for p in SEED_FILES]
    mean_row = {}
    std_row = {}
    for f in FIELDS:
        vals = [m[f] for m in seed_metrics]
        mean_row[f] = statistics.mean(vals)
        std_row[f] = statistics.pstdev(vals) if f != "n" else 0

    baseline_label = "v23 baseline (5-seed mean)"
    baseline_config = "top25/seq30"
    rows.insert(4, (baseline_label, baseline_config, mean_row))

    header = ["Version", "Config", "Test R²", "Test RMSE", "Test ubRMSE", "Test Bias", "Test MAE", "Test Q90", "n"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for label, config, m in rows:
        if label == baseline_label:
            r2 = f"{m['r2']:.4f} ± {std_row['r2']:.4f}"
            rmse = f"{m['rmse']:.5f} ± {std_row['rmse']:.5f}"
            ubrmse = f"{m['ubrmse']:.5f} ± {std_row['ubrmse']:.5f}"
            bias = f"{m['bias']:.5f} ± {std_row['bias']:.5f}"
            mae = f"{m['mae']:.5f} ± {std_row['mae']:.5f}"
            q90 = f"{m['q90']:.5f} ± {std_row['q90']:.5f}"
        else:
            r2 = f"{m['r2']:.4f}"
            rmse = f"{m['rmse']:.5f}"
            ubrmse = f"{m['ubrmse']:.5f}"
            bias = f"{m['bias']:.5f}"
            mae = f"{m['mae']:.5f}"
            q90 = f"{m['q90']:.5f}"
        n = int(m["n"])
        print(f"| {label} | {config} | {r2} | {rmse} | {ubrmse} | {bias} | {mae} | {q90} | {n} |")

    csv_path = HERE / "version_comparison_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "config"] + FIELDS + ["std_r2", "std_rmse", "std_ubrmse", "std_bias", "std_mae", "std_q90"])
        for label, config, m in rows:
            if label == baseline_label:
                std_vals = [std_row[k] for k in ["r2", "rmse", "ubrmse", "bias", "mae", "q90"]]
            else:
                std_vals = ["", "", "", "", "", ""]
            w.writerow([label, config] + [m[f] for f in FIELDS] + std_vals)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
