# Jakob Balkovec
# output parser

# parses the output of prune.py and compiles it into a readable format

# to run: python prune_output_parser.py –run-dir ./prune_runs/run_* –top-k * –show-rounds *
# *: integer

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def color(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if USE_COLOR else text


def c_bold(text: str) -> str:
    return color(BOLD, text)


def c_dim(text: str) -> str:
    return color(DIM, text)


def c_red(text: str) -> str:
    return color(RED, text)


def c_green(text: str) -> str:
    return color(GREEN, text)


def c_yellow(text: str) -> str:
    return color(YELLOW, text)


def c_cyan(text: str) -> str:
    return color(CYAN, text)


def c_importance(v, fmt="{:+.6f}", yellow_thresh=0.00005):
    try:
        x = float(v)
    except Exception:
        return str(v)

    s = fmt.format(x)
    if x < 0:
        return c_red(s)
    if x < yellow_thresh:
        return c_yellow(s)
    return c_green(s)


def header(title: str) -> None:
    print(f"\n{c_bold(title)}")


def info(label: str, value: str) -> None:
    print(f"{c_dim(label)} {value}")


def metric_delta(curr: float, prev: float, higher_is_better: bool, fmt: str = "{:+.6f}") -> str:
    d = float(curr) - float(prev)
    txt = fmt.format(d)
    good = d > 0 if higher_is_better else d < 0
    bad = d < 0 if higher_is_better else d > 0
    if good:
        return c_green(txt)
    if bad:
        return c_red(txt)
    return txt


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def read_feature_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.shape[1] == 0 or df.empty:
        return []
    col = df.columns[0]
    return df[col].dropna().astype(str).tolist()


ROUND_RX = re.compile(r"_round_(\d+)_")


def extract_round_num(path: Path) -> int:
    m = ROUND_RX.search(path.name)
    return int(m.group(1)) if m else -1


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{(100.0 * n / d):.1f}%"


def print_inventory(run_dir: Path) -> None:
    header("File Inventory")
    csvs = list(run_dir.rglob("*.csv"))

    counts = {
        "history": 0,
        "best_features": 0,
        "round_ranked": 0,
        "round_features": 0,
        "start_features": 0,
        "unmatched_features": 0,
        "final_best_features": 0,
        "other_csv": 0,
    }

    for p in csvs:
        n = p.name
        if n.endswith("_history.csv"):
            counts["history"] += 1
        elif n.endswith("_best_features.csv"):
            counts["best_features"] += 1
        elif "_round_" in n and n.endswith("_ranked.csv"):
            counts["round_ranked"] += 1
        elif "_round_" in n and n.endswith("_features.csv"):
            counts["round_features"] += 1
        elif n == "start_features.csv":
            counts["start_features"] += 1
        elif n == "unmatched_features.csv":
            counts["unmatched_features"] += 1
        elif n == "FINAL_best_features.csv":
            counts["final_best_features"] += 1
        else:
            counts["other_csv"] += 1

    info("Run directory:", str(run_dir))
    info("CSV files:", str(len(csvs)))
    for k, v in counts.items():
        print(f"  {k:>18}: {v}")


def print_config_summary(run_dir: Path) -> None:
    cfg_path = run_dir / "config_used.yaml"
    cfg = read_yaml(cfg_path)
    if not cfg:
        header("Config")
        print(c_yellow("config_used.yaml not found"))
        return

    header("Config")
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    families = cfg.get("families", {})
    perm = cfg.get("prune", {}).get("perm", {})

    info("Config file:", str(cfg_path))
    info("Target:", str(data.get("target_col")))
    info("Date parse:", f"parse_dates={data.get('parse_dates')} date_col={data.get('date_col')}")
    info("Families:", f"include={families.get('include', [])}")
    info("Permutation:", f"seeds={perm.get('seeds')} repeats={perm.get('repeats')} scoring={perm.get('scoring')}")
    info("Train split:", str(paths.get("train")))
    info("Val split:", str(paths.get("val")))
    info("Test split:", str(paths.get("test")))


def print_feature_set_summary(run_dir: Path, sample_n: int = 12) -> None:
    header("Feature Set Flow")

    start = read_feature_list(run_dir / "start_features.csv")
    coarse = read_feature_list(run_dir / "stage_coarse" / "coarse_best_features.csv")
    fine = read_feature_list(run_dir / "stage_fine" / "fine_best_features.csv")
    finalf = read_feature_list(run_dir / "FINAL_best_features.csv")
    unmatched = read_feature_list(run_dir / "unmatched_features.csv")

    print(f"{'start':>10}: {len(start):>4}")
    print(f"{'coarse_best':>10}: {len(coarse):>4} ({pct(len(coarse), len(start))} retained)")
    print(f"{'fine_best':>10}: {len(fine):>4} ({pct(len(fine), len(start))} retained)")
    print(f"{'final_best':>10}: {len(finalf):>4} ({pct(len(finalf), len(start))} retained)")
    print(f"{'unmatched':>10}: {len(unmatched):>4}")

    def transition(name: str, prev: Iterable[str], curr: Iterable[str]) -> None:
        p = set(prev)
        c = set(curr)
        dropped = sorted(p - c)
        added = sorted(c - p)
        print(f"  {name}: -{len(dropped)} +{len(added)}")
        if dropped:
            print(f"    dropped sample: {', '.join(dropped[:sample_n])}")
        if added:
            print(f"    added sample:   {', '.join(added[:sample_n])}")

    print()
    transition("start -> coarse_best", start, coarse)
    transition("coarse_best -> fine_best", coarse, fine)
    transition("fine_best -> final_best", fine, finalf)


def print_stage_summary(run_dir: Path, stage: str, top_k: int, show_rounds: int) -> None:
    stage_dir = run_dir / f"stage_{stage}"
    hist_path = stage_dir / f"{stage}_history.csv"
    summary_path = stage_dir / f"{stage}_summary.json"

    header(f"Stage: {stage}")

    if not hist_path.exists():
        print(c_red(f"Missing file: {hist_path}"))
        return

    hist = pd.read_csv(hist_path)
    if hist.empty:
        print(c_red(f"Empty history: {hist_path}"))
        return

    for col in ("round", "n_features", "r2", "rmse", "mae", "p90_abs_err"):
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")

    start = hist.iloc[0]
    final = hist.iloc[-1]
    best_idx = hist["r2"].idxmax()
    best = hist.loc[best_idx]

    n_start = int(start["n_features"])
    n_final = int(final["n_features"])
    pruned = n_start - n_final

    print(f"features: {n_start} -> {n_final} ({pct(pruned, n_start)} pruned)")
    print(
        "val_r2: "
        f"{start['r2']:.6f} -> {final['r2']:.6f} "
        f"({metric_delta(final['r2'], start['r2'], higher_is_better=True)}) | "
        f"best {best['r2']:.6f} @ round {int(best['round'])}"
    )
    print(
        "val_rmse: "
        f"{start['rmse']:.6f} -> {final['rmse']:.6f} "
        f"({metric_delta(final['rmse'], start['rmse'], higher_is_better=False)})"
    )
    print(
        "val_p90: "
        f"{start['p90_abs_err']:.6f} -> {final['p90_abs_err']:.6f} "
        f"({metric_delta(final['p90_abs_err'], start['p90_abs_err'], higher_is_better=False)})"
    )

    stage_summary = read_json(summary_path)
    if stage_summary:
        info("summary.json:", json.dumps(stage_summary, sort_keys=True))

    print(c_dim("recent rounds:"))
    recent = hist.tail(max(1, show_rounds)).copy()
    prev = None
    for _, r in recent.iterrows():
        rr = int(r["round"])
        nf = int(r["n_features"])
        if prev is None:
            print(f"  round {rr:>3} | n={nf:>4} | r2={r['r2']:.6f} | rmse={r['rmse']:.6f} | p90={r['p90_abs_err']:.6f}")
        else:
            dr2 = metric_delta(r["r2"], prev["r2"], higher_is_better=True)
            drmse = metric_delta(r["rmse"], prev["rmse"], higher_is_better=False)
            dp90 = metric_delta(r["p90_abs_err"], prev["p90_abs_err"], higher_is_better=False)
            print(
                f"  round {rr:>3} | n={nf:>4} | r2={r['r2']:.6f} ({dr2}) | "
                f"rmse={r['rmse']:.6f} ({drmse}) | p90={r['p90_abs_err']:.6f} ({dp90})"
            )
        prev = r

    ranked_files = sorted(stage_dir.glob(f"{stage}_round_*_ranked.csv"), key=extract_round_num)
    if ranked_files:
        latest = ranked_files[-1]
        ranked = pd.read_csv(latest)
        print(c_dim(f"top {top_k} ranked features from {latest.name}:"))
        for i, (_, row) in enumerate(ranked.head(top_k).iterrows(), start=1):
            f = str(row.get("feature", ""))
            imp = float(row.get("importance", float("nan")))
            print(f"  {i:>2}. {f:<50} {c_importance(imp)}")
    else:
        print(c_yellow("No ranked CSV files found for this stage"))

    round_feature_files = sorted(stage_dir.glob(f"{stage}_round_*_features.csv"), key=extract_round_num)
    if len(round_feature_files) >= 2:
        print(c_dim("feature churn (between saved rounds):"))
        rows = []
        for prev_fp, curr_fp in zip(round_feature_files[:-1], round_feature_files[1:]):
            r_prev = extract_round_num(prev_fp)
            r_curr = extract_round_num(curr_fp)
            prev_set = set(read_feature_list(prev_fp))
            curr_set = set(read_feature_list(curr_fp))
            removed = len(prev_set - curr_set)
            added = len(curr_set - prev_set)
            rows.append((r_prev, r_curr, removed, added))

        for r_prev, r_curr, removed, added in rows[-show_rounds:]:
            add_txt = c_green(f"+{added}") if added > 0 else f"+{added}"
            rem_txt = c_red(f"-{removed}") if removed > 0 else f"-{removed}"
            print(f"  round {r_prev:>3} -> {r_curr:>3} | {rem_txt} {add_txt}")


def print_final_metrics(run_dir: Path) -> None:
    header("Final Metrics")
    p = run_dir / "FINAL_metrics.json"
    m = read_json(p)
    if not m:
        print(c_red(f"Missing or unreadable: {p}"))
        return

    val = m.get("val", {})
    test = m.get("test", {})
    n_features = m.get("n_features")

    info("n_features:", str(n_features))

    for key, higher_is_better in (
        ("r2", True),
        ("rmse", False),
        ("mae", False),
        ("p90_abs_err", False),
        ("bias", False),
    ):
        v = val.get(key)
        t = test.get(key)
        if v is None or t is None:
            continue
        d = metric_delta(t, v, higher_is_better=higher_is_better)
        print(f"{key:>12}: val={float(v): .6f}  test={float(t): .6f}  gap(test-val)={d}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interpret prune.py run outputs")
    ap.add_argument("--run-dir", required=True, help="Path to prune run directory (example: ./prune_runs/run01)")
    ap.add_argument("--top-k", type=int, default=15, help="Top ranked features to display per stage")
    ap.add_argument("--show-rounds", type=int, default=8, help="How many recent rounds to print")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Invalid --run-dir: {run_dir}")

    print(c_bold("Prune Output Interpreter"))
    print(c_dim("=" * 72))

    print_inventory(run_dir)
    print_config_summary(run_dir)
    print_feature_set_summary(run_dir)
    print_stage_summary(run_dir, stage="coarse", top_k=args.top_k, show_rounds=args.show_rounds)
    print_stage_summary(run_dir, stage="fine", top_k=args.top_k, show_rounds=args.show_rounds)
    print_final_metrics(run_dir)

    print(f"\n{c_dim('Done')}")


if __name__ == "__main__":
    main()
