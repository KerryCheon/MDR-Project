from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TOP_N = 15
SELECTION_MODE = "topN"              # "topN" or "threshold"
GLOBAL_IMPORTANCE_MIN = None         # used when SELECTION_MODE == "threshold"
GLOBAL_IMPORTANCE_PERCENTILE = None  # used when SELECTION_MODE == "threshold"

USE_MODEL_FOR_SET10 = "xgb"          # "xgb" | "rf" | "avg"

PEARSON_ABS_THR  = 0.6  # per request (dropped from 0.8)
SPEARMAN_ABS_THR = 0.6  # per request (dropped from 0.8)
MUTUAL_INFO_THR  = 0.3  # per request (dropped from 0.1)
MI_NORMALIZE = True     # normalize MI by max off-diagonal

DATASET_PATH_OVERRIDE = None         # e.g., "MDR/Temporal/Pipeline/data/splits/derived_new/test_derived_new.csv"
MI_BINS = 20
USE_SKLEARN_MI = True
RANDOM_STATE = 42
DATASET_SELECTION_MODE = "max_coverage"  # "max_coverage" | "prefer_base" | "first_match"

FEATURE_ALIASES = {
    "NDVI": ["F_NDVI"],
    "NDMI": ["F_NDMI"],
    "MSI": ["F_MSI"],
    "SAR_ratio": ["E_SAR_ratio"],
    "SAR_diff": ["E_SAR_diff"],
    "API": ["G_API"],
}

PREFIX_CANDIDATES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "V"]
EXCLUDE_FEATURES = {"air_temp_mean", "rh_mean"}

def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "MDR").exists():
            return parent
    return here.parents[2]

REPO_ROOT = find_repo_root()
OUTPUT_DIR = Path(__file__).resolve().parent
RUNS_ROOT = REPO_ROOT / "MDR/Models/Temporal"
FEATURE_IMPORTANCE_FILENAME = "feature_importance.csv"
TOP_K_PER_SET = 10

DEFAULT_DATASET_CANDIDATES = [
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/derived_new_updated/test_derived_updated.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/derived_new/test_derived_new.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/derived_all/test_derived_all.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/derived/test_derived.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/base/test_base.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/base_no_met/test_base.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/splits/base_no_met/test_base_no_met.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/master_cleaned/final_master_cleaned.csv",
    REPO_ROOT / "MDR/Temporal/Pipeline/data/master/final_master.csv",
]

try:
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

def parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except Exception:
        return None


def extract_test_r2(meta: dict) -> float | None:
    perf = meta.get("performance") or {}
    for key in ("test_R2", "test_r2", "r2", "R2", "test_r2_score"):
        val = perf.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def infer_model_type(meta: dict, run_dir: Path) -> str:
    alg = (meta.get("algorithm") or "").lower()
    model_name = (meta.get("model_name") or "").lower()
    run_name = (meta.get("run_name") or run_dir.name).lower()

    if "randomforest" in alg or "random forest" in alg or alg == "rf":
        return "rf"
    if "rf" in run_name or "randomforest" in model_name or "random forest" in model_name:
        return "rf"
    if "xgb" in alg or "xgboost" in alg:
        return "xgb"
    if "xgb" in run_name or "xgboost" in model_name:
        return "xgb"
    if "model_config" in meta:
        return "xgb"
    return "unknown"


def load_run_info(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    run_dir = meta_path.parent
    features = meta.get("features") or []
    timestamp = meta.get("timestamp")
    return {
        "run_dir": run_dir,
        "meta_path": meta_path,
        "meta": meta,
        "features": features,
        "features_key": tuple(sorted(features)),
        "n_features": len(features),
        "timestamp": timestamp,
        "timestamp_dt": parse_timestamp(timestamp),
        "test_r2": extract_test_r2(meta),
        "model_type": infer_model_type(meta, run_dir),
        "importance_path": run_dir / FEATURE_IMPORTANCE_FILENAME,
    }


def collect_runs(runs_root: Path) -> List[dict]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")
    meta_paths = sorted(runs_root.rglob("run_metadata.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No run_metadata.json found under: {runs_root}")

    runs = []
    for meta_path in meta_paths:
        info = load_run_info(meta_path)
        if not info["features"]:
            continue
        runs.append(info)
    if not runs:
        raise ValueError("No runs with feature lists found under runs root.")
    return runs


def group_runs_by_feature_set(runs: List[dict]) -> List[dict]:
    groups: Dict[Tuple[str, ...], List[dict]] = {}
    for run in runs:
        groups.setdefault(run["features_key"], []).append(run)

    result = []
    for features_key, group_runs in groups.items():
        timestamps = [r["timestamp_dt"] for r in group_runs if r["timestamp_dt"] is not None]
        min_timestamp = min(timestamps) if timestamps else None
        result.append(
            {
                "features_key": features_key,
                "features": list(features_key),
                "n_features": len(features_key),
                "runs": group_runs,
                "min_timestamp": min_timestamp,
                "signature": "|".join(features_key),
            }
        )
    return result


def assign_feature_set_ids(groups: List[dict]) -> Dict[int, dict]:
    groups_sorted = sorted(
        groups,
        key=lambda g: (g["n_features"], g["min_timestamp"] or datetime.min, g["signature"]),
    )
    return {idx + 1: group for idx, group in enumerate(groups_sorted)}


def select_best_run(runs: List[dict], model_type: str | None = None) -> dict | None:
    candidates = [r for r in runs if r["importance_path"].exists()]
    if model_type:
        candidates = [r for r in candidates if r["model_type"] == model_type]
    if not candidates:
        return None

    with_r2 = [r for r in candidates if r["test_r2"] is not None]
    if with_r2:
        return max(
            with_r2,
            key=lambda r: (r["test_r2"], r["timestamp_dt"] or datetime.min),
        )
    return max(candidates, key=lambda r: (r["timestamp_dt"] or datetime.min))


def load_importance_rows(path: Path, top_k: int) -> List[Tuple[str, float]]:
    df = pd.read_csv(path)
    feature_col = "feature"
    if feature_col not in df.columns:
        if "Unnamed: 0" in df.columns:
            feature_col = "Unnamed: 0"
        else:
            feature_col = df.columns[0]

    value_col = None
    for candidate in ("importance", "mean_abs_shap_norm", "mean_abs_shap", "mean_shap"):
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        raise ValueError(f"Missing importance column in {path}")

    df = df[[feature_col, value_col]].copy()
    df = df.rename(columns={feature_col: "feature", value_col: "importance"})
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df = df.dropna(subset=["feature", "importance"])
    df = df.sort_values("importance", ascending=False)
    if top_k is not None:
        df = df.head(top_k)
    return [(row["feature"], float(row["importance"])) for _, row in df.iterrows()]


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

def evaluate_dataset(path: Path, selected_features: List[str]) -> Tuple[List[str], List[dict], List[str]]:
    header_df = pd.read_csv(path, nrows=0)
    columns = header_df.columns.tolist()
    return resolve_feature_list(selected_features, columns)


def select_dataset(selected_features: List[str]) -> Tuple[Path, List[Path], List[str], List[dict], List[str], str]:
    searched: List[Path] = []

    if DATASET_PATH_OVERRIDE:
        candidate = (REPO_ROOT / DATASET_PATH_OVERRIDE).resolve() if not Path(DATASET_PATH_OVERRIDE).is_absolute() else Path(DATASET_PATH_OVERRIDE)
        searched.append(candidate)
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset override not found: {candidate}")
        present, mapping, missing = evaluate_dataset(candidate, selected_features)
        return candidate, searched, present, mapping, missing, "override"

    candidates = [p for p in DEFAULT_DATASET_CANDIDATES if p.exists()]
    searched.extend(DEFAULT_DATASET_CANDIDATES)

    if not candidates:
        fallback = sorted((REPO_ROOT / "MDR/Temporal/Pipeline/data/splits").glob("**/test_*.csv"))
        searched.extend(fallback)
        candidates = fallback

    if not candidates:
        expected_names = [p.name for p in DEFAULT_DATASET_CANDIDATES]
        searched_str = "\n".join(str(p) for p in searched)
        raise FileNotFoundError(
            "No dataset found. Searched candidates and fallback paths:\n"
            f"Expected names: {expected_names}\n"
            f"Searched:\n{searched_str}"
        )

    if DATASET_SELECTION_MODE == "prefer_base":
        for path in candidates:
            if path.name == "test_base.csv":
                present, mapping, missing = evaluate_dataset(path, selected_features)
                return path, searched, present, mapping, missing, "prefer_base"

    if DATASET_SELECTION_MODE == "first_match":
        path = candidates[0]
        present, mapping, missing = evaluate_dataset(path, selected_features)
        return path, searched, present, mapping, missing, "first_match"

    # default: max_coverage
    best = None
    for path in candidates:
        present, mapping, missing = evaluate_dataset(path, selected_features)
        score = (len(present), -len(missing))
        if best is None or score > best["score"]:
            best = {
                "path": path,
                "present": present,
                "mapping": mapping,
                "missing": missing,
                "score": score,
            }

    return best["path"], searched, best["present"], best["mapping"], best["missing"], "max_coverage"


def resolve_feature_name(feature: str, columns: List[str]) -> Tuple[str | None, str, List[str]]:
    if feature in columns:
        return feature, "exact", [feature]

    candidates: List[str] = []
    for alias in FEATURE_ALIASES.get(feature, []):
        candidates.append(alias)

    for prefix in PREFIX_CANDIDATES:
        candidates.append(f"{prefix}_{feature}")

    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    valid = [c for c in ordered if c in columns]

    if len(valid) == 1:
        if valid[0] in FEATURE_ALIASES.get(feature, []):
            return valid[0], "alias", valid
        if valid[0].split("_", 1)[0] in PREFIX_CANDIDATES and valid[0].endswith(f"_{feature}"):
            return valid[0], "prefixed", valid
        return valid[0], "suffix", valid

    if len(valid) > 1:
        return None, "ambiguous", valid

    return None, "missing", []


def resolve_feature_list(features: List[str], columns: List[str]) -> Tuple[List[str], List[dict], List[str]]:
    resolved = []
    mapping_rows = []
    missing = []
    used = set()

    for feature in features:
        resolved_name, status, candidates = resolve_feature_name(feature, columns)
        if resolved_name is None:
            mapping_rows.append(
                {
                    "original": feature,
                    "resolved": "",
                    "status": status,
                    "candidates": ", ".join(candidates),
                }
            )
            missing.append(feature)
            continue

        if resolved_name in used:
            mapping_rows.append(
                {
                    "original": feature,
                    "resolved": resolved_name,
                    "status": "duplicate",
                    "candidates": ", ".join(candidates),
                }
            )
            continue

        used.add(resolved_name)
        resolved.append(resolved_name)
        mapping_rows.append(
            {
                "original": feature,
                "resolved": resolved_name,
                "status": status,
                "candidates": ", ".join(candidates),
            }
        )

    return resolved, mapping_rows, missing

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


def compute_mi_matrix_binned(df: pd.DataFrame, bins: int) -> pd.DataFrame:
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


def compute_mi_matrix_sklearn(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    features = df.columns.tolist()
    n = len(features)
    mi = np.zeros((n, n), dtype=float)

    X = zscore_frame(df).fillna(0.0).values

    for i in range(n):
        for j in range(i + 1, n):
            xi = X[:, [i]]
            yj = X[:, j]
            xj = X[:, [j]]
            yi = X[:, i]
            mi_ij = mutual_info_regression(xi, yj, random_state=random_state)
            mi_ji = mutual_info_regression(xj, yi, random_state=random_state)
            mi_sym = 0.5 * (float(mi_ij[0]) + float(mi_ji[0]))
            mi[i, j] = mi_sym
            mi[j, i] = mi_sym

    return pd.DataFrame(mi, index=features, columns=features)


def compute_mi_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if USE_SKLEARN_MI and SKLEARN_AVAILABLE:
        return compute_mi_matrix_sklearn(df, RANDOM_STATE), "sklearn"
    return compute_mi_matrix_binned(df, MI_BINS), "binned"


def normalize_mi_matrix(mi: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if mi.empty:
        return mi, "none"
    values = mi.values.copy()
    mask = ~np.eye(values.shape[0], dtype=bool)
    off_diag = values[mask]
    max_val = float(np.nanmax(off_diag)) if off_diag.size else 0.0
    if not np.isfinite(max_val) or max_val <= 0:
        return mi, "none"
    return mi / max_val, "max"

def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_heatmap(matrix: pd.DataFrame, path: Path, title: str, vmin=None, vmax=None, cmap: str = "coolwarm") -> bool:
    mpl_dir = OUTPUT_DIR / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    values = matrix.values
    size = max(6, min(16, len(matrix) * 0.6))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def save_bar_chart(df: pd.DataFrame, path: Path, title: str) -> bool:
    mpl_dir = OUTPUT_DIR / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    ax.barh(df["feature"], df["global_importance"], color="#3b6ea5")
    ax.set_title(title)
    ax.set_xlabel("Global Importance")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def classify_relationship(pearson: float, spearman: float, mutual_info: float) -> str:
    if abs(pearson) >= PEARSON_ABS_THR:
        return f"Strong {'positive' if pearson > 0 else 'negative'} linear"
    if abs(spearman) >= SPEARMAN_ABS_THR:
        return f"Strong {'positive' if spearman > 0 else 'negative'} monotonic"
    if mutual_info >= MUTUAL_INFO_THR:
        return "Nonlinear dependency"
    return "None"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load run feature-importance CSVs
    runs = collect_runs(RUNS_ROOT)
    feature_groups = group_runs_by_feature_set(runs)
    feature_sets = assign_feature_set_ids(feature_groups)

    expected_sets = list(range(1, 12))
    found_sets = sorted(feature_sets.keys())
    if found_sets != expected_sets:
        missing = [s for s in expected_sets if s not in found_sets]
        group_sizes = ", ".join(str(g["n_features"]) for g in feature_groups)
        raise ValueError(
            "Expected 11 feature sets (01..11). "
            f"Found: {found_sets}. Missing: {missing}. "
            f"Group sizes (n_features): [{group_sizes}]"
        )

    selected_tables: Dict[int, List[Tuple[str, float]]] = {}
    selected_runs: List[dict] = []
    for set_id in expected_sets:
        group = feature_sets[set_id]
        runs_in_group = group["runs"]
        xgb_run = select_best_run(runs_in_group, "xgb")
        rf_run = select_best_run(runs_in_group, "rf")
        fallback_run = select_best_run(runs_in_group, None)

        if set_id == 10:
            if USE_MODEL_FOR_SET10 == "xgb":
                run = xgb_run or fallback_run
                if run is None:
                    raise ValueError("Feature set 10 XGB run not found.")
                selected_tables[set_id] = load_importance_rows(run["importance_path"], TOP_K_PER_SET)
                used_fallback = xgb_run is None or run is not xgb_run
                model_label = run["model_type"] if used_fallback else "xgb"
                note_label = "selected (fallback)" if used_fallback else "selected"
                selected_runs.append(
                    {
                        "set_id": set_id,
                        "n_features": group["n_features"],
                        "model": model_label,
                        "run": run,
                        "note": note_label,
                    }
                )
            elif USE_MODEL_FOR_SET10 == "rf":
                run = rf_run or fallback_run
                if run is None:
                    raise ValueError("Feature set 10 RF run not found.")
                selected_tables[set_id] = load_importance_rows(run["importance_path"], TOP_K_PER_SET)
                used_fallback = rf_run is None or run is not rf_run
                model_label = run["model_type"] if used_fallback else "rf"
                note_label = "selected (fallback)" if used_fallback else "selected"
                selected_runs.append(
                    {
                        "set_id": set_id,
                        "n_features": group["n_features"],
                        "model": model_label,
                        "run": run,
                        "note": note_label,
                    }
                )
            elif USE_MODEL_FOR_SET10 == "avg":
                if xgb_run is None or rf_run is None:
                    raise ValueError("Feature set 10 requires both XGB and RF runs for avg.")
                xgb_rows = load_importance_rows(xgb_run["importance_path"], TOP_K_PER_SET)
                rf_rows = load_importance_rows(rf_run["importance_path"], TOP_K_PER_SET)
                selected_tables[set_id] = average_tables(xgb_rows, rf_rows, top_n=TOP_K_PER_SET)
                selected_runs.append(
                    {"set_id": set_id, "n_features": group["n_features"], "model": "xgb", "run": xgb_run, "note": "avg"}
                )
                selected_runs.append(
                    {"set_id": set_id, "n_features": group["n_features"], "model": "rf", "run": rf_run, "note": "avg"}
                )
            else:
                raise ValueError(f"Unknown USE_MODEL_FOR_SET10 option: {USE_MODEL_FOR_SET10}")
        else:
            run = xgb_run or fallback_run
            if run is None:
                raise ValueError(f"Feature set {set_id:02d} run not found.")
            selected_tables[set_id] = load_importance_rows(run["importance_path"], TOP_K_PER_SET)
            used_fallback = xgb_run is None or run is not xgb_run
            note_label = "selected (fallback)" if used_fallback else "selected"
            selected_runs.append(
                {
                    "set_id": set_id,
                    "n_features": group["n_features"],
                    "model": run["model_type"] or "unknown",
                    "run": run,
                    "note": note_label,
                }
            )

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
        if feature in EXCLUDE_FEATURES:
            continue
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

    top_png = OUTPUT_DIR / f"global_importance_top{TOP_N}.png"
    top_fig = save_bar_chart(df_top, top_png, f"Top {TOP_N} Global Importance")

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

    consistent_features = df_global[df_global["freq_count"] >= 5]["feature"].tolist()
    dataset_path, searched, present_features, mapping_rows, missing_features, dataset_select_mode = select_dataset(selected_features)

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

    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")
    mi, mi_method = compute_mi_matrix(df)
    mi_norm_method = "none"
    if MI_NORMALIZE:
        mi, mi_norm_method = normalize_mi_matrix(mi)

    pearson_path = OUTPUT_DIR / "correlation_pearson.csv"
    spearman_path = OUTPUT_DIR / "correlation_spearman.csv"
    mi_path = OUTPUT_DIR / "correlation_mutual_info.csv"
    pearson.to_csv(pearson_path)
    spearman.to_csv(spearman_path)
    mi.to_csv(mi_path)

    pearson_png = OUTPUT_DIR / "correlation_pearson.png"
    spearman_png = OUTPUT_DIR / "correlation_spearman.png"
    mi_png = OUTPUT_DIR / "correlation_mutual_info.png"
    pearson_fig = save_heatmap(pearson, pearson_png, "Pearson Correlation", vmin=-1, vmax=1, cmap="coolwarm")
    spearman_fig = save_heatmap(spearman, spearman_png, "Spearman Correlation", vmin=-1, vmax=1, cmap="coolwarm")
    mi_vmax = 1 if MI_NORMALIZE and mi_norm_method != "none" else None
    mi_fig = save_heatmap(mi, mi_png, "Mutual Information", vmin=0, vmax=mi_vmax, cmap="viridis")

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
        {
            "feature_a": a,
            "feature_b": b,
            "pearson": p,
            "spearman": s,
            "mutual_info": m,
            "relationship_type": classify_relationship(p, s, m),
        }
        for _, a, b, p, s, m in pairs
    ]
    df_pairs = pd.DataFrame(pair_rows)
    df_pairs.to_csv(OUTPUT_DIR / "high_corr_pairs.csv", index=False)

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
        ["Feature", "Freq Count", "Frequency", "Avg Importance", "Global Importance"], top_table_rows
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
                r["relationship_type"],
            ]
            for r in report_pairs
        ]
        pair_table_md = format_markdown_table(
            ["Feature A", "Feature B", "Pearson", "Spearman", "Mutual Info", "Relationship Type"],
            pair_table_rows,
        )
    else:
        pair_table_md = "No pairs exceeded thresholds."

    try:
        dataset_rel = dataset_path.relative_to(REPO_ROOT)
    except ValueError:
        dataset_rel = dataset_path

    try:
        runs_rel = RUNS_ROOT.relative_to(REPO_ROOT)
    except ValueError:
        runs_rel = RUNS_ROOT

    mapping_table_rows = [
        [
            row["original"],
            row["resolved"] or "—",
            row["status"],
            row["candidates"] or "—",
        ]
        for row in mapping_rows
    ]
    mapping_table_md = format_markdown_table(
        ["Original", "Resolved Column", "Status", "Candidates"], mapping_table_rows
    )

    consistent_rows = [[feat] for feat in consistent_features] or [["None"]]
    consistent_table_md = format_markdown_table(["Feature (>=5/11)"], consistent_rows)

    mi_method_label = "scikit-learn (symmetrized)" if mi_method == "sklearn" else f"binned histogram (bins={MI_BINS})"
    if MI_NORMALIZE and mi_norm_method != "none":
        mi_method_label += f", normalized by {mi_norm_method}"

    selection_table_rows = []
    for entry in selected_runs:
        run = entry["run"]
        try:
            run_rel = run["run_dir"].relative_to(REPO_ROOT)
        except ValueError:
            run_rel = run["run_dir"]
        r2_val = run.get("test_r2")
        r2_str = f"{r2_val:.4f}" if r2_val is not None else "—"
        selection_table_rows.append(
            [
                f"{entry['set_id']:02d}",
                str(entry["n_features"]),
                entry["model"],
                str(run_rel),
                r2_str,
                run.get("timestamp") or "—",
                entry["note"],
            ]
        )
    selection_table_md = format_markdown_table(
        ["Set", "n_features", "Model", "Run", "test_R2", "Timestamp", "Note"],
        selection_table_rows,
    )

    report_lines = [
        "# Global Feature Importance + Correlation Report",
        "",
        "## Quick Summary",
        f"- Dataset: `{dataset_rel}`",
        f"- Selected features (after mapping): {len(present_features)}",
        f"- Missing after mapping: {len(missing_features)}",
        f"- Feature set 10 handling: `{USE_MODEL_FOR_SET10}`",
        f"- MI estimator: {mi_method_label}",
        f"- Dataset selection: `{dataset_select_mode}`",
        f"- Runs root: `{runs_rel}`",
        "",
        "## Feature Set Sources",
        "Runs are grouped by identical feature lists, ordered by feature count (ties by timestamp/signature).",
        "Per set, we pick the best test_R2 run per model type when available (fallback to most recent).",
        selection_table_md,
        "",
        "## Method (casual)",
        f"We grab the top-{TOP_K_PER_SET} `feature_importance.csv` rows for each of the 11 feature sets, then compute:",
        "",
        rf"- Frequency: \( f = \frac{{\#\text{{sets with feature in top-{TOP_K_PER_SET}}}}}{11} \)",
        r"- Avg importance: \( \overline{I} \) over the sets where the feature appears",
        r"- Global importance: \( G = f \times \overline{I} \)",
        "",
        f"Feature selection for correlation uses `{selection_desc}`.",
        f"Correlations are computed on `{dataset_rel}` (rows: {after_rows}, dropped NaNs: {dropped_rows}).",
        "Note: Output column names retain `avg_shap`/`shap_values` for backwards compatibility.",
        "",
        "## Top Features by Global Importance",
        top_table_md,
        f"![](global_importance_top{TOP_N}.png)" if top_fig else "- Top-importance chart not generated (matplotlib missing).",
        "",
        "## Consistent Features (>=5 of 11)",
        consistent_table_md,
        "",
        "## Feature Column Mapping",
        "Some features appear with a leading family letter (e.g., `NDVI` -> `F_NDVI`).",
        mapping_table_md,
        "",
        "## Correlation Thresholds",
        f"- Pearson |r| >= {PEARSON_ABS_THR}",
        f"- Spearman |rho| >= {SPEARMAN_ABS_THR}",
        f"- Mutual information >= {MUTUAL_INFO_THR}",
        "- MI normalized by max off-diagonal value (range ~[0, 1])" if MI_NORMALIZE and mi_norm_method != "none" else "",
        "",
        "## Correlation Heatmaps",
        "These are for the selected features (after mapping).",
        f"![](correlation_pearson.png)" if pearson_fig else "- Pearson heatmap not generated (matplotlib missing).",
        f"![](correlation_spearman.png)" if spearman_fig else "- Spearman heatmap not generated (matplotlib missing).",
        f"![](correlation_mutual_info.png)" if mi_fig else "- MI heatmap not generated (matplotlib missing).",
        "",
        "## Top Correlated Pairs (max 30)",
        "Sorted by max(|Pearson|, |Spearman|, MI).",
        pair_table_md,
        "",
        "## Missing Features After Mapping",
        f"- {', '.join(missing_features) if missing_features else 'None'}",
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
