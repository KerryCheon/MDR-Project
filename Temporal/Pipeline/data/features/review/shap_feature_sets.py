import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
import shap

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[5]
REVIEW_DIR = THIS_FILE.parent
SELECTED_FEATURES_PATH = REVIEW_DIR / "SELECTED_FEATURES.md"
REPORT_PATH = REVIEW_DIR / "SHAP_ANALYSIS.md"
PLOTS_DIR = REVIEW_DIR / "shap_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_PATHS = {
    "base": {
        "train": PROJECT_ROOT / "Temporal/Pipeline/data/splits/base/train_base.csv",
        "val": PROJECT_ROOT / "Temporal/Pipeline/data/splits/base/val_base.csv",
        "test": PROJECT_ROOT / "Temporal/Pipeline/data/splits/base/test_base.csv",
    },
    "base_no_met": {
        "train": PROJECT_ROOT / "Temporal/Pipeline/data/splits/base_no_met/train_base_no_met.csv",
        "val": PROJECT_ROOT / "Temporal/Pipeline/data/splits/base_no_met/val_base_no_met.csv",
        "test": PROJECT_ROOT / "Temporal/Pipeline/data/splits/base_no_met/test_base_no_met.csv",
    },
    "derived_all": {
        "train": PROJECT_ROOT / "Temporal/Pipeline/data/splits/derived_all/train_derived_all.csv",
        "val": PROJECT_ROOT / "Temporal/Pipeline/data/splits/derived_all/val_derived_all.csv",
        "test": PROJECT_ROOT / "Temporal/Pipeline/data/splits/derived_all/test_derived_all.csv",
    },
    "derived": {
        "train": PROJECT_ROOT / "Temporal/Pipeline/data/splits/derived/train_derived.csv",
        "val": PROJECT_ROOT / "Temporal/Pipeline/data/splits/derived/val_derived.csv",
        "test": PROJECT_ROOT / "Temporal/Pipeline/data/splits/derived/test_derived.csv",
    },
    "derived_new": {
        "train": PROJECT_ROOT / "Temporal/Pipeline/data/splits/derived_new/train_derived_new.csv",
        "val": PROJECT_ROOT / "Temporal/Pipeline/data/splits/derived_new/val_derived_new.csv",
        "test": PROJECT_ROOT / "Temporal/Pipeline/data/splits/derived_new/test_derived_new.csv",
    },
}


def parse_selected_features(path: Path) -> dict[int, list[dict]]:
    text = path.read_text()
    lines = text.splitlines()

    feature_sets: dict[int, list[dict]] = {}
    current_set: int | None = None
    in_table = False

    for line in lines:
        if line.startswith("## Feature set"):
            m = re.match(r"## Feature set\s+(\d+)", line)
            if m:
                current_set = int(m.group(1))
                feature_sets[current_set] = []
            in_table = False
            continue

        if current_set is None:
            continue

        if line.strip().startswith("Model runs:"):
            in_table = True
            continue

        if in_table:
            if not line.strip().startswith("|"):
                if line.strip() == "":
                    in_table = False
                continue

            if "---" in line:
                continue

            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) < 5:
                continue

            version, run_folder, r2_str = parts[0], parts[1], parts[2]
            run_folder = run_folder.strip("`")

            try:
                r2 = float(r2_str)
            except Exception:
                r2 = None

            feature_sets[current_set].append(
                {
                    "version": version,
                    "run_folder": run_folder,
                    "r2": r2,
                }
            )

    return feature_sets


def select_best_runs(feature_sets: dict[int, list[dict]]) -> dict[int, dict]:
    best = {}
    for fs, runs in feature_sets.items():
        valid = [r for r in runs if isinstance(r.get("r2"), float)]
        if not valid:
            continue
        best_run = max(valid, key=lambda r: r["r2"])
        best[fs] = best_run
    return best


def load_run_metadata(run_folder: Path) -> dict:
    meta_path = run_folder / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing run_metadata.json: {meta_path}")
    return json.loads(meta_path.read_text())


def load_splits(split_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = SPLIT_PATHS[split_type]
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["val"])
    test_df = pd.read_csv(paths["test"])
    return train_df, val_df, test_df


def eval_metrics(y_true, y_pred) -> dict:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, y_pred),
    }


def shap_summary_and_bar(shap_values, X_eval, out_prefix: Path, max_display: int = 20):
    plt.figure()
    shap.summary_plot(shap_values, X_eval, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_summary.png", dpi=180)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_eval, show=False, plot_type="bar", max_display=max_display)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bar.png", dpi=180)
    plt.close()

def shap_dependence_plots(
    shap_values: np.ndarray,
    X_eval: pd.DataFrame,
    features: list[str],
    out_prefix: Path,
    sample_size: int = 2000,
    random_state: int = 42,
):
    if len(X_eval) == 0:
        return []

    if len(X_eval) > sample_size:
        X_sample = X_eval.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X_eval.copy()

    # Align SHAP values to sampled rows
    shap_sample = shap_values[X_sample.index.to_numpy(), :]

    paths = []
    for feat in features:
        plt.figure()
        shap.dependence_plot(
            feat,
            shap_sample,
            X_sample,
            interaction_index="auto",
            show=False,
        )
        plt.tight_layout()
        out_path = f"{out_prefix}_{feat}_dependence.png"
        plt.savefig(out_path, dpi=180)
        plt.close()
        paths.append(out_path)
    return paths

def compute_shap_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean = np.mean(shap_values, axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean,
    }).sort_values("mean_abs_shap", ascending=False)
    s = df["mean_abs_shap"].sum()
    df["mean_abs_shap_norm"] = df["mean_abs_shap"] / (s if s != 0 else 1)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def add_rain_impulse_features(d: pd.DataFrame) -> pd.DataFrame:
    RAIN_COL = "precip_mm"
    RAIN_THR = 4.0
    K = 7
    WEIGHTS = np.array([1.0, 0.6, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0], dtype=float)

    d = d.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values(["station_id", "date"]).reset_index(drop=True)

    d["_rain_event"] = (d[RAIN_COL] >= RAIN_THR).astype(int)

    for k in range(0, K + 1):
        d[f"_ev_lag{k}"] = d.groupby("station_id")["_rain_event"].shift(k)
        d[f"_mm_lag{k}"] = d.groupby("station_id")[RAIN_COL].shift(k)

    ev_cols = [f"_ev_lag{k}" for k in range(0, K + 1)]
    mm_cols = [f"_mm_lag{k}" for k in range(0, K + 1)]

    ev_mat = d[ev_cols].fillna(0).to_numpy(dtype=float)
    mm_mat = d[mm_cols].fillna(0).to_numpy(dtype=float)

    d["rain_event_impulse_0_7"] = (ev_mat * WEIGHTS).sum(axis=1)
    d["rain_mm_impulse_0_7"] = (mm_mat * WEIGHTS).sum(axis=1)

    def _days_since_event(group: pd.DataFrame) -> pd.Series:
        ev = group["_rain_event"].to_numpy()
        out = np.full(len(ev), np.nan, dtype=float)
        last_idx = None
        for i in range(len(ev)):
            if ev[i] == 1:
                last_idx = i
                out[i] = 0.0
            else:
                if last_idx is not None:
                    out[i] = float(i - last_idx)
        return pd.Series(out, index=group.index)

    d["days_since_rain_event"] = (
        d.groupby("station_id", group_keys=False)
         .apply(_days_since_event)
         .clip(upper=30)
    )

    drop_cols = ["_rain_event"] + ev_cols + mm_cols
    d.drop(columns=drop_cols, inplace=True, errors="ignore")

    return d

feature_sets = parse_selected_features(SELECTED_FEATURES_PATH)
best_runs = select_best_runs(feature_sets)

SPLIT_TYPE_BY_RUN = {
    "Models/Temporal/v1/v1.0/mdr_ts_v2_3_20260108_115420": "base_no_met",
    "Models/Temporal/v1/v1.2/mdr_ts_v1_2_20251223_113808": "base",
    "Models/Temporal/v2/v2.1/mdr_ts_v2_1_20260106_211015": "base",
    "Models/Temporal/v1/v1.1/mdr_ts_v1_1_20251223_113030": "base",
    "Models/Temporal/v1/v1.0/mdr_ts_v1_0_20251223_105722": "base",
    "Models/Temporal/v2/v2.3/mdr_ts_v2_2_20260107_170214": "derived",
    "Models/Temporal/v3/v3.3/mdr_ts_v3_3_20260109_113817": "derived_all",
    "Models/Temporal/v3/v3.1/mdr_ts_v3_1_20260108_151234": "derived_all",
    "Models/Temporal/v8/v8.1/mdr_ts_v8_1_20260121_105425": "derived_new",
    "Models/Temporal/v7/v7.1/mdr_ts_v7_1_20260117_161048": "derived_new",
    "Models/Temporal/v3/v3.1/mdr_ts_v3_1_20260108_144926": "derived_all",
}

MODEL_OVERRIDE_BY_RUN = {
    "Models/Temporal/v7/v7.1/mdr_ts_v7_1_20260117_161048": {
        "rf_config": {
            "n_estimators": 800,
            "max_depth": 18,
            "min_samples_leaf": 10,
            "min_samples_split": 20,
            "max_features": 0.5,
            "bootstrap": True,
            "n_jobs": -1,
            "random_state": 42,
        }
    }
}

report_lines: list[str] = []
report_lines.append("# SHAP Analysis for Selected Feature Sets")
report_lines.append("")
report_lines.append("This report retrains the best-performing run (by test R² in `SELECTED_FEATURES.md`) for each feature set and computes SHAP on the full test split.")
report_lines.append("")
report_lines.append("**Notes:**")
report_lines.append("- SHAP is computed on the full test set for each run.")
report_lines.append("- Relationship (dependence) plots use a random sample from the test set to keep runtime reasonable.")
report_lines.append("- For the v7.1 run, both XGBoost and RandomForest models are included (as used in the original notebook).")
report_lines.append("- Plots are saved under `Temporal/Pipeline/data/features/review/shap_plots/`.")
report_lines.append("")


def flush_report() -> None:
    REPORT_PATH.write_text("\n".join(report_lines))

for fs in sorted(best_runs.keys()):
    run_info = best_runs[fs]
    run_folder_str = run_info["run_folder"]
    run_folder = PROJECT_ROOT / run_folder_str
    if run_folder_str not in SPLIT_TYPE_BY_RUN:
        raise ValueError(f"Missing split mapping for run: {run_folder_str}")

    split_type = SPLIT_TYPE_BY_RUN[run_folder_str]
    meta = load_run_metadata(run_folder)

    features = meta["features"]
    target = meta["target"]

    train_df, val_df, test_df = load_splits(split_type)

    missing_cols = sorted(list(set(features + [target]) - set(train_df.columns)))
    if missing_cols:
        rain_features = {
            "days_since_rain_event",
            "rain_event_impulse_0_7",
            "rain_mm_impulse_0_7",
        }
        if set(missing_cols).issubset(rain_features):
            train_df = add_rain_impulse_features(train_df)
            val_df = add_rain_impulse_features(val_df)
            test_df = add_rain_impulse_features(test_df)
            missing_cols = sorted(list(set(features + [target]) - set(train_df.columns)))
        if missing_cols:
            raise ValueError(f"Missing columns in train split for run {run_folder_str}: {missing_cols}")

    for d in (train_df, val_df, test_df):
        d.replace([np.inf, -np.inf], np.nan, inplace=True)

    X_train = train_df[features].copy()
    y_train = train_df[target].copy()
    X_val = val_df[features].copy()
    y_val = val_df[target].copy()
    X_test = test_df[features].copy()
    y_test = test_df[target].copy()

    report_lines.append(f"## Feature set {fs:02d}")
    report_lines.append("")
    report_lines.append(f"Best run: `{run_folder_str}`")
    report_lines.append("")

    models_to_run = ["xgb"]
    overrides = MODEL_OVERRIDE_BY_RUN.get(run_folder_str, {})
    if "rf_config" in overrides:
        models_to_run.append("rf")

    for model_name in models_to_run:
        if model_name == "xgb":
            model_config = meta.get("model_config") or meta.get("best_params")
            if model_config is None:
                raise ValueError(f"No model config found for XGB run: {run_folder_str}")
            model = XGBRegressor(**model_config)
            model.fit(X_train, y_train)
            yhat_train = model.predict(X_train)
            yhat_val = model.predict(X_val)
            yhat_test = model.predict(X_test)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test).values
            shap_df = compute_shap_importance(shap_values, list(X_test.columns))

            plot_prefix = PLOTS_DIR / f"feature_set_{fs:02d}_xgb"
            shap_summary_and_bar(shap_values, X_test, plot_prefix)

            top_features = shap_df["feature"].head(3).tolist()
            dep_prefix = PLOTS_DIR / f"feature_set_{fs:02d}_xgb"
            dep_paths = shap_dependence_plots(
                shap_values,
                X_test,
                top_features,
                dep_prefix,
            )

        elif model_name == "rf":
            rf_config = overrides["rf_config"]
            rf = RandomForestRegressor(**rf_config, oob_score=True)
            model = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", rf),
            ])
            model.fit(X_train, y_train)
            yhat_train = model.predict(X_train)
            yhat_val = model.predict(X_val)
            yhat_test = model.predict(X_test)

            # SHAP on imputed data with the RF estimator
            imputer = model.named_steps["imputer"]
            rf_model = model.named_steps["model"]
            X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer(X_test_imp).values
            shap_df = compute_shap_importance(shap_values, list(X_test.columns))

            plot_prefix = PLOTS_DIR / f"feature_set_{fs:02d}_rf"
            shap_summary_and_bar(shap_values, X_test_imp, plot_prefix)

            top_features = shap_df["feature"].head(3).tolist()
            dep_prefix = PLOTS_DIR / f"feature_set_{fs:02d}_rf"
            dep_paths = shap_dependence_plots(
                shap_values,
                X_test_imp,
                top_features,
                dep_prefix,
            )

        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        metrics_train = eval_metrics(y_train, yhat_train)
        metrics_val = eval_metrics(y_val, yhat_val)
        metrics_test = eval_metrics(y_test, yhat_test)

        report_lines.append(f"### Model: {model_name.upper()}")
        report_lines.append("")
        report_lines.append("**Metrics (retrained):**")
        report_lines.append("")
        report_lines.append("| Split | R² | RMSE | MAE |")
        report_lines.append("| --- | --- | --- | --- |")
        report_lines.append(
            f"| Train | {metrics_train['r2']:.4f} | {metrics_train['rmse']:.4f} | {metrics_train['mae']:.4f} |"
        )
        report_lines.append(
            f"| Val | {metrics_val['r2']:.4f} | {metrics_val['rmse']:.4f} | {metrics_val['mae']:.4f} |"
        )
        report_lines.append(
            f"| Test | {metrics_test['r2']:.4f} | {metrics_test['rmse']:.4f} | {metrics_test['mae']:.4f} |"
        )
        report_lines.append("")

        report_lines.append("**Top features (mean |SHAP|, normalized):**")
        report_lines.append("")
        report_lines.append("| Rank | Feature | Mean |SHAP| | Share |")
        report_lines.append("| --- | --- | --- | --- |")
        for _, row in shap_df.head(10).iterrows():
            report_lines.append(
                f"| {int(row['rank'])} | {row['feature']} | {row['mean_abs_shap']:.6f} | {row['mean_abs_shap_norm']:.4f} |"
            )
        report_lines.append("")

        report_lines.append(f"![Feature set {fs:02d} {model_name.upper()} SHAP summary](shap_plots/feature_set_{fs:02d}_{model_name}_summary.png)")
        report_lines.append("")
        report_lines.append(f"![Feature set {fs:02d} {model_name.upper()} SHAP bar](shap_plots/feature_set_{fs:02d}_{model_name}_bar.png)")
        report_lines.append("")

        if dep_paths:
            report_lines.append("**Relationships (dependence plots):**")
            report_lines.append("")
            for p in dep_paths:
                # convert absolute path to relative
                rel = Path(p).relative_to(REVIEW_DIR)
                report_lines.append(f"![Feature set {fs:02d} {model_name.upper()} dependence]({rel.as_posix()})")
                report_lines.append("")
        flush_report()

    report_lines.append("---")
    report_lines.append("")

REPORT_PATH.write_text("\n".join(report_lines))
print(f"Report written to: {REPORT_PATH}")
