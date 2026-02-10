#!/usr/bin/env python3
"""Compute global SHAP importance and feature correlations.

Outputs are written to analysis/global_shap/.
"""
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
TOP_N = 15
SELECTION_MODE = "topN"  # "topN" or "threshold"
GLOBAL_IMPORTANCE_MIN = None  # used when SELECTION_MODE == "threshold"
GLOBAL_IMPORTANCE_PERCENTILE = None  # used when SELECTION_MODE == "threshold"

USE_MODEL_FOR_SET10 = "xgb"  # "xgb" | "rf" | "avg"

PEARSON_ABS_THR = 0.80
SPEARMAN_ABS_THR = 0.80
MUTUAL_INFO_THR = 0.10

DATASET_PATH_OVERRIDE = None  # e.g., "MDR/Temporal/Pipeline/data/splits/derived_all/test_derived_all.csv"
MI_BINS = 20


# -----------------------------
# Paths
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
SHAP_MD_PATH = REPO_ROOT / "MDR/Temporal/Pipeline/data/features/review/SHAP_ANALYSIS.md"

DEFAULT_DATASET_CANDIDATES = [
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/derived_all/test_derived_all.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/derived_new/test_derived_new.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/derived/test_derived.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/base/test_base.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/base_no_met/test_base_no_met.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/master_cleaned/final_master_cleaned.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/master/final_master.csv",
]


# -----------------------------
# Parsing
# -----------------------------

def parse_shap_tables(md_path: Path) -> Dict[int, Dict[str, List[Tuple[str, float]]]]:
    if not md_path.exists():
        raise FileNotFoundError(f"SHAP analysis file not found: {md_path}")

    lines = md_path.read_text().splitlines()
    tables: Dict[int, Dict[str, List[Tuple[str, float]]]] = {}
    current_set = None
    current_model = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"^## Feature set (\d{2})$", line)
        if m:
            current_set = int(m.group(1))
            current_model = None
            tables.setdefault(current_set, {})
            i += 1
            continue

        m = re.match(r"^### Model: (.+)$", line)
        if m:
            model = m.group(1).strip().lower()
            if "xgb" in model:
                current_model = "xgb"
            elif model in {"rf", "randomforest", "random_forest", "random forest"}:
                current_model = "rf"
            else:
                current_model = model.replace(" ", "_")
            i += 1
            continue

        if line.startswith("**Top features by mean |SHAP| (normalized):**"):
            if current_set is None:
                raise ValueError("Found SHAP table before any feature set header.")
            if current_model is None:
                current_model = "xgb"

            # Advance to first table line
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("|"):
                j += 1

            rows: List[Tuple[str, float]] = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                parts = [p.strip() for p in lines[j].strip().strip("|").split("|")]
                if len(parts) >= 3 and parts[0].isdigit():
                    feature = parts[1]
                    try:
                        shap = float(parts[2])
                    except ValueError:
                        shap = float(parts[2].replace(",", ""))
                    rows.append((feature, shap))
                j += 1

            tables[current_set][current_model] = rows
            i = j
            continue

        i += 1

    return tables


def average_tables(
    xgb_rows: List[Tuple[str, float]],
    rf_rows: List[Tuple[str, float]],
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    xgb = {f: v for f, v in xgb_rows}
    rf = {f: v for f, v in rf_rows}
    all_features = set(xgb) | set(rf)
    avg = {f: (xgb.get(f, 0.0) + rf.get(f, 0.0)) / 2.0 for f in all_features}
    return sorted(avg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


# -----------------------------
# Dataset discovery
# -----------------------------

def find_dataset() -> Tuple[Path, List[Path]]:
    searched: List[Path] = []
    if DATASET_PATH_OVERRIDE:
        candidate = (REPO_ROOT / DATASET_PATH_OVERRIDE).resolve() if not Path(DATASET_PATH_OVERRIDE).is_absolute() else Path(DATASET_PATH_OVERRIDE)
        searched.append(candidate)
        if candidate.exists():
            return candidate, searched
        raise FileNotFoundError(f"Dataset override not found: {candidate}")

    for path in DEFAULT_DATASET_CANDIDATES:
        searched.append(path)
        if path.exists():
            return path, searched

    # Fallback: find any test split in pipeline
    fallback = sorted((REPO_ROOT / "MDR/Temporal/Pipeline/data/splits").glob("**/test_*.csv"))
    searched.extend(fallback)
    if fallback:
        return fallback[0], searched

    return None, searched


# -----------------------------
# Correlations
# -----------------------------

def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    means = df.mean(axis=0)
    stds = df.std(axis=0, ddof=0).replace(0, np.nan)
    return (df - means) / stds


def mutual_info_binned(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    if c_xy.sum() == 0:
        return 0.0
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    p_x_p_y = p_x[:, None] * p_y[None, :]
    nz = p_xy > 0
    return float(np.sum(p_xy[nz] * np.log(p_xy[nz] / p_x_p_y[nz])))


def compute_mi_matrix(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    features = df.columns.tolist()
    n = len(features)
    mi = np.zeros((n, n), dtype=float)

    X = zscore_frame(df).fillna(0.0).values

    for i in range(n):
        for j in range(i + 1, n):
            xi = X[:, i]
            xj = X[:, j]
            mi_sym = mutual_info_binned(xi, xj, bins=bins)
            mi[i, j] = mi_sym
            mi[j, i] = mi_sym

    return pd.DataFrame(mi, index=features, columns=features)


# -----------------------------
# Reporting
# -----------------------------

def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse SHAP tables
    tables = parse_shap_tables(SHAP_MD_PATH)
    expected_sets = list(range(1, 12))
    found_sets = sorted(tables.keys())
    if found_sets != expected_sets:
        missing = [s for s in expected_sets if s not in found_sets]
        raise ValueError(
            f"Expected 11 feature sets (01..11). Found: {found_sets}. Missing: {missing}"
        )

    # Select table per feature set
    selected_tables: Dict[int, List[Tuple[str, float]]] = {}
    for set_id in expected_sets:
        set_tables = tables.get(set_id, {})
        if set_id == 10:
            if USE_MODEL_FOR_SET10 == "xgb":
                if "xgb" not in set_tables:
                    raise ValueError("Feature set 10 XGB table not found.")
                selected_tables[set_id] = set_tables["xgb"]
            elif USE_MODEL_FOR_SET10 == "rf":
                if "rf" not in set_tables:
                    raise ValueError("Feature set 10 RF table not found.")
                selected_tables[set_id] = set_tables["rf"]
            elif USE_MODEL_FOR_SET10 == "avg":
                if "xgb" not in set_tables or "rf" not in set_tables:
                    raise ValueError("Feature set 10 requires both XGB and RF tables for avg.")
                selected_tables[set_id] = average_tables(set_tables["xgb"], set_tables["rf"], top_n=10)
            else:
                raise ValueError(f"Unknown USE_MODEL_FOR_SET10 option: {USE_MODEL_FOR_SET10}")
        else:
            if "xgb" not in set_tables:
                raise ValueError(f"Feature set {set_id:02d} XGB table not found.")
            selected_tables[set_id] = set_tables["xgb"]

    # Global importance
    shap_values: Dict[str, List[float]] = {}
    presence: Dict[str, set] = {}

    for set_id, rows in selected_tables.items():
        seen = set()
        for feature, shap in rows:
            if feature not in seen:
                shap_values.setdefault(feature, []).append(shap)
                presence.setdefault(feature, set()).add(set_id)
                seen.add(feature)

    data_rows = []
    for feature, values in shap_values.items():
        freq_count = len(presence.get(feature, []))
        frequency = freq_count / 11.0
        avg_shap = float(np.mean(values)) if values else 0.0
        global_importance = frequency * avg_shap
        data_rows.append(
            {
                "feature": feature,
                "freq_count": freq_count,
                "frequency": frequency,
                "shap_values": ";".join(f"{v:.6f}" for v in values),
                "avg_shap": avg_shap,
                "global_importance": global_importance,
            }
        )

    df_global = pd.DataFrame(data_rows).sort_values("global_importance", ascending=False).reset_index(drop=True)
    global_path = OUTPUT_DIR / "global_importance.csv"
    df_global.to_csv(global_path, index=False)

    df_top = df_global.head(TOP_N)
    top_path = OUTPUT_DIR / f"global_importance_top{TOP_N}.csv"
    df_top.to_csv(top_path, index=False)

    if SELECTION_MODE == "threshold":
        if GLOBAL_IMPORTANCE_MIN is not None:
            selected = df_global[df_global["global_importance"] >= GLOBAL_IMPORTANCE_MIN]
            selection_desc = f"global_importance >= {GLOBAL_IMPORTANCE_MIN}"
        elif GLOBAL_IMPORTANCE_PERCENTILE is not None:
            thr = df_global["global_importance"].quantile(GLOBAL_IMPORTANCE_PERCENTILE)
            selected = df_global[df_global["global_importance"] >= thr]
            selection_desc = f"global_importance >= {GLOBAL_IMPORTANCE_PERCENTILE:.2f} percentile (>= {thr:.6f})"
        else:
            selected = df_top
            selection_desc = f"top {TOP_N} by global_importance (fallback)"
    else:
        selected = df_top
        selection_desc = f"top {TOP_N} by global_importance"

    selected_features = selected["feature"].tolist()

    consistent_features = df_global[df_global["freq_count"] >= 6]["feature"].tolist()

    # Dataset
    dataset_path, searched = find_dataset()
    if dataset_path is None:
        expected_names = [p.name for p in DEFAULT_DATASET_CANDIDATES]
        searched_str = "\n".join(str(p) for p in searched)
        raise FileNotFoundError(
            "No dataset found. Searched candidates and fallback paths:\n"
            f"Expected names: {expected_names}\n"
            f"Searched:\n{searched_str}"
        )

    header_df = pd.read_csv(dataset_path, nrows=0)
    available_cols = set(header_df.columns)
    missing_features = sorted([f for f in selected_features if f not in available_cols])
    present_features = [f for f in selected_features if f in available_cols]

    if not present_features:
        raise ValueError(
            "None of the selected features were found in the dataset columns. "
            f"Dataset: {dataset_path}"
        )

    df = pd.read_csv(dataset_path, usecols=present_features)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.select_dtypes(include=[np.number])

    before_rows = len(df)
    df = df.dropna()
    after_rows = len(df)
    dropped_rows = before_rows - after_rows

    if df.shape[0] < 2:
        raise ValueError("Not enough rows after dropping NaNs to compute correlations.")

    # Correlations
    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")
    mi = compute_mi_matrix(df, MI_BINS)

    pearson.to_csv(OUTPUT_DIR / "correlation_pearson.csv")
    spearman.to_csv(OUTPUT_DIR / "correlation_spearman.csv")
    mi.to_csv(OUTPUT_DIR / "correlation_mutual_info.csv")

    # High-correlation pairs
    pairs = []
    features = pearson.columns.tolist()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            a = features[i]
            b = features[j]
            p = float(pearson.loc[a, b])
            s = float(spearman.loc[a, b])
            m = float(mi.loc[a, b])
            if abs(p) >= PEARSON_ABS_THR or abs(s) >= SPEARMAN_ABS_THR or m >= MUTUAL_INFO_THR:
                score = max(abs(p), abs(s), m)
                pairs.append((score, a, b, p, s, m))

    pairs.sort(key=lambda x: x[0], reverse=True)

    pair_rows = [
        {"feature_a": a, "feature_b": b, "pearson": p, "spearman": s, "mutual_info": m}
        for _, a, b, p, s, m in pairs
    ]
    df_pairs = pd.DataFrame(pair_rows)
    df_pairs.to_csv(OUTPUT_DIR / "high_corr_pairs.csv", index=False)

    # Report
    top_table_rows = []
    for _, row in df_top.iterrows():
        top_table_rows.append(
            [
                row["feature"],
                str(int(row["freq_count"])),
                f"{row['frequency']:.3f}",
                f"{row['avg_shap']:.6f}",
                f"{row['global_importance']:.6f}",
            ]
        )

    top_table_md = format_markdown_table(
        ["Feature", "Freq Count", "Frequency", "Avg SHAP", "Global Importance"], top_table_rows
    )

    report_pairs = pair_rows[:30]
    if report_pairs:
        pair_table_rows = [
            [
                r["feature_a"],
                r["feature_b"],
                f"{r['pearson']:.3f}",
                f"{r['spearman']:.3f}",
                f"{r['mutual_info']:.3f}",
            ]
            for r in report_pairs
        ]
        pair_table_md = format_markdown_table(
            ["Feature A", "Feature B", "Pearson", "Spearman", "Mutual Info"], pair_table_rows
        )
    else:
        pair_table_md = "No pairs exceeded thresholds."

    try:
        dataset_rel = dataset_path.relative_to(REPO_ROOT)
    except ValueError:
        dataset_rel = dataset_path

    report_lines = [
        "# Global SHAP Feature Importance + Correlation Report",
        "",
        "## Method (brief)",
        f"- Parsed top-10 mean |SHAP| tables from `{SHAP_MD_PATH.relative_to(REPO_ROOT)}`.",
        f"- Feature set 10 handling: `{USE_MODEL_FOR_SET10}`.",
        f"- Computed global importance: frequency (appears in top-10 across 11 sets) * avg SHAP.",
        f"- Selected features for correlation using `{selection_desc}`.",
        f"- Correlations computed on dataset: `{dataset_rel}` (rows: {after_rows}, dropped NaN rows: {dropped_rows}).",
        f"- Mutual information estimated via binned histogram (bins={MI_BINS}) on z-scored features.",
        "",
        "## Top Features by Global Importance",
        top_table_md,
        "",
        "## High-Correlation Thresholds",
        f"- Pearson |r| >= {PEARSON_ABS_THR}",
        f"- Spearman |rho| >= {SPEARMAN_ABS_THR}",
        f"- Mutual information >= {MUTUAL_INFO_THR}",
        "",
        "## Top Correlated Pairs (max 30)",
        pair_table_md,
        "",
        "## Consistency",
        f"- Features appearing in >=6 of 11 sets: {len(consistent_features)}",
        f"- List: {', '.join(consistent_features) if consistent_features else 'None'}",
        "",
        "## Missing Features",
        f"- Missing from dataset columns: {', '.join(missing_features) if missing_features else 'None'}",
        "",
        "## Notes",
        "- Pearson captures linear relationships.",
        "- Spearman captures monotonic relationships.",
        "- Mutual information captures nonlinear dependencies.",
    ]

    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text("\n".join(report_lines))

    # Validation prints
    print("Global importance rows:", len(df_global))
    print("Top features saved to:", top_path)
    print("Dataset used:", dataset_path)
    print("Dataset shape (after dropna):", df.shape)
    print("Selected features for correlation:", len(present_features))
    if missing_features:
        print("Missing features:")
        for f in missing_features:
            print(" -", f)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc)
        sys.exit(1)
