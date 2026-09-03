"""Router-only salvage on the canonical derived_8.4_ece_v3 split (1.1).

Same protocol as derived_8.4-ece-router-salvage-1.0, with two changes:
  1. Single ECE input: data/splits/derived_8.4_ece_v3/test.csv (150 rows,
     30-day warmup scaffold, strict native-NaN SMAP, MODIS NDVI fallback).
  2. Two single-regime global baselines reported as reference rows
     (policy=direct): Global_Single_54 (54 backbone features) and
     Global_Single_50 (50 V0 features).

Protocol (ECE strictly unseen):
  - Routers fit on WA trainval only (14,608 rows, 7 stations).
  - Experts and baselines (XGBRegressor, formal-eval exact_params, cpu) fit
    on WA trainval only. ECE targets are NEVER used for fitting.
  - Routing policies are inference-time label overrides applied to the SAME
    frozen experts. ECE is used for evaluation only. margin_fallback falls
    back to the Global_Single_54 expert.

Policies per family on v3:
  as_routed, c0_only, c1_only, gapi_transplant, dynamic_transplant,
  seasonal, margin_fallback; baselines use direct.

Resume: per-seed checkpoints under checkpoints/; reruns skip completed seeds
unless --no-resume is passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
FORMAL_DIR = PROJECT_ROOT / "notebooks/experiment/derived_8.4-formal-eval-2.0-ece"
sys.path.insert(0, str(FORMAL_DIR))
from eval_formal.routers import (  # noqa: E402
    Backbone54Router,
    DynamicClusterRouter,
    SeasonalBinaryRouter,
    UnivariateGAPIRouter,
    V0FullRouter,
)

TARGET_DEFAULT = "soil_moisture_5cm"
CHECKPOINT_DIR = EXP_DIR / "checkpoints"

# Series persisted per row for the ≤5-line station charts: the story set
# (as_routed, c0_only, margin_fallback) for regime families + direct baselines.
CHART_POLICIES = ("as_routed", "c0_only", "margin_fallback", "direct")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"r2": float("nan"), "rmse": float("nan"), "ubrmse": float("nan"),
                "bias": float("nan"), "mae": float("nan"), "pearson": float("nan")}
    bias = float(np.mean(y_pred - y_true))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    ubrmse = float(np.sqrt(max(0.0, rmse ** 2 - bias ** 2)))
    out = {"r2": float(r2_score(y_true, y_pred)), "rmse": rmse, "ubrmse": ubrmse,
           "bias": bias, "mae": float(mean_absolute_error(y_true, y_pred))}
    if len(y_true) > 1 and np.std(y_true) > 1e-9 and np.std(y_pred) > 1e-9:
        out["pearson"] = float(pearsonr(y_true, y_pred)[0])
    else:
        out["pearson"] = float("nan")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="1 seed (42) + 100 trees for a fast CPU check.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed override.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore per-seed checkpoints and refit everything.")
    return parser.parse_args(argv)


def load_configuration() -> dict:
    with (EXP_DIR / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_v0_features(project_root: Path) -> list[str]:
    path = project_root / "data/splits/derived_8.4/dataset_metadata.py"
    spec = importlib.util.spec_from_file_location("derived_84_metadata", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metadata module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.OVERALL_SELECTED_FEATURES_V0)


def _load_backbone() -> list[str]:
    with (FORMAL_DIR / "config.yaml").open(encoding="utf-8") as handle:
        formal = yaml.safe_load(handle)
    return list(formal["shared_backbone_54"])


def _prepare(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["month"] = out["date"].dt.month.astype(int)
    out["year"] = out["date"].dt.year.astype(int)
    return out


def load_data(config: dict) -> dict:
    target = config["target_column"]
    train_dir = PROJECT_ROOT / config["datasets"]["training"]
    train = pd.concat([
        pd.read_csv(train_dir / "train.csv", low_memory=False),
        pd.read_csv(train_dir / "val.csv", low_memory=False),
    ], ignore_index=True)
    v3 = pd.read_csv(PROJECT_ROOT / config["datasets"]["ece_v3"] / "test.csv",
                     low_memory=False)
    # Canonical v3 schema guards (mirror tests/test_ece_v3_split.py).
    # NOTE: raw v3 already carries a `year` column (499 cols total);
    # `_prepare` adds `month` (-> 500 cols), so assert BEFORE preparing.
    assert len(v3) == 150, f"Expected 150 v3 rows, got {len(v3)}"
    assert len(v3.columns) == 499, f"Expected 499 raw v3 columns, got {len(v3.columns)}"
    trainval = _prepare(train, target)
    v3 = _prepare(v3, target)
    counts = v3["station_id"].value_counts()
    assert len(counts) == 5 and bool((counts == 30).all()), \
        f"Expected 5 stations x 30 rows, got {counts.to_dict()}"
    assert v3["date"].min() == pd.Timestamp("2026-07-20")
    assert v3["date"].max() == pd.Timestamp("2026-08-19")
    smap_val = [c for c in v3.columns if "SMAP" in c and not c.endswith("_mask")]
    assert bool((v3[smap_val] != 0.0).all().all()), "No spurious SMAP 0.0 allowed"
    return {"trainval": trainval, "v3": v3, "target": target}


def kmeans_margin(router, frame: pd.DataFrame) -> np.ndarray:
    """Distance gap |d0 - d1| under the router's own scaler/centroids."""
    values = frame.loc[:, router.features].copy().fillna(router.means)
    dists = router.kmeans.transform(router.scaler.transform(values))
    return np.abs(dists[:, 0] - dists[:, 1])


def fit_routers(trainval: pd.DataFrame, v0_features: list[str],
                backbone: list[str]) -> dict:
    routers = {
        "v0": V0FullRouter(v0_features, seed=42),
        "backbone": Backbone54Router(backbone, seed=42),
        "dynamic": DynamicClusterRouter(seed=42),
        "gapi": UnivariateGAPIRouter(),
        "seasonal": SeasonalBinaryRouter(),
    }
    for router in routers.values():
        router.fit(trainval)
    return routers


def fit_experts(trainval: pd.DataFrame, labels: np.ndarray, features: list[str],
                target: str, params: dict, seed: int) -> dict[int, XGBRegressor]:
    y = trainval[target].to_numpy(dtype=float)
    experts: dict[int, XGBRegressor] = {}
    for cluster in (0, 1):
        mask = labels == cluster
        if not mask.any():
            continue
        model = XGBRegressor(**params)
        model.fit(trainval.loc[mask, features], y[mask], verbose=False)
        experts[cluster] = model
    return experts


def predict_with_labels(experts: dict[int, XGBRegressor], frame: pd.DataFrame,
                        labels: np.ndarray, features: list[str],
                        fallback: float) -> np.ndarray:
    preds = np.zeros(len(frame), dtype=float)
    for cluster in (0, 1):
        mask = labels == cluster
        if not mask.any():
            continue
        expert = experts.get(cluster)
        if expert is None:
            preds[mask] = fallback
        else:
            preds[mask] = np.asarray(
                expert.predict(frame.loc[mask, features])).ravel()
    return preds


def run_seed(seed: int, data: dict, routers: dict, v0_features: list[str],
             backbone: list[str], config: dict, params: dict,
             wa_thresholds: dict[str, float]
             ) -> tuple[list[dict], list[dict], dict, list[dict]]:
    trainval, frame = data["trainval"], data["v3"]
    target = data["target"]
    y = frame[target].to_numpy(dtype=float)
    fallback = float(trainval[target].mean())
    families = {"Clustering_V0_Full_k2": "v0", "Clustering_Backbone54_k2": "backbone"}
    baseline_features = {"Global_Single_54": backbone, "Global_Single_50": v0_features}
    station_ids = frame["station_id"].to_numpy()
    dates = frame["date"].dt.strftime("%Y-%m-%d").to_numpy()

    seed_params = dict(params)
    seed_params["random_state"] = seed

    # Frozen experts and baselines, fit on WA trainval only.
    family_experts, family_labels = {}, {}
    for family, key in families.items():
        labels = np.asarray(routers[key].predict(trainval)).ravel().astype(int)
        family_labels[family] = labels
        family_experts[family] = fit_experts(trainval, labels, backbone, target,
                                             seed_params, seed)
    baselines = {}
    for name, features in baseline_features.items():
        model = XGBRegressor(**seed_params)
        model.fit(trainval[features], trainval[target].to_numpy(dtype=float),
                  verbose=False)
        baselines[name] = (model, features)
    global_pred_54 = np.asarray(
        baselines["Global_Single_54"][0].predict(frame[backbone])).ravel()

    label_sets = {
        "as_routed": None,  # per-family static labels
        "c0_only": np.zeros(len(frame), dtype=int),
        "c1_only": np.ones(len(frame), dtype=int),
        "gapi_transplant": np.asarray(routers["gapi"].predict(frame)).ravel().astype(int),
        "dynamic_transplant": np.asarray(routers["dynamic"].predict(frame)).ravel().astype(int),
        "seasonal": np.asarray(routers["seasonal"].predict(frame)).ravel().astype(int),
    }

    records, station_records, audit, pred_rows = [], [], {"seed": seed, "policies": {}}, []
    per_station = frame["station_id"].to_numpy()

    def _record(family: str, policy: str, preds: np.ndarray, extra: dict) -> None:
        metrics = compute_metrics(y, preds)
        records.append({"family": family, "ece_input": "v3",
                        "policy": policy, "seed": seed, **metrics})
        for station in sorted(frame["station_id"].unique()):
            mask = per_station == station
            sm = compute_metrics(y[mask], preds[mask])
            station_records.append({"family": family, "ece_input": "v3",
                                    "policy": policy, "seed": seed,
                                    "station": station,
                                    "n": int(mask.sum()), **sm})
        audit["policies"].setdefault(family, {})[policy] = extra
        if policy in CHART_POLICIES:
            for i in range(len(frame)):
                pred_rows.append({"seed": seed, "station_id": station_ids[i],
                                  "date": dates[i], "y_true": float(y[i]),
                                  "family": family, "policy": policy,
                                  "y_pred": float(preds[i])})

    for family, key in families.items():
        as_routed = np.asarray(routers[key].predict(frame)).ravel().astype(int)
        margins = kmeans_margin(routers[key], frame)
        experts = family_experts[family]
        for policy in config["policies"]:
            if policy == "direct":
                continue  # baselines only
            if policy == "as_routed":
                labels = as_routed
                preds = predict_with_labels(experts, frame, labels, backbone, fallback)
                extra = {"c0_share": float((labels == 0).mean())}
            elif policy == "margin_fallback":
                thresh = wa_thresholds[key]
                confident = margins >= thresh
                routed = predict_with_labels(experts, frame, as_routed, backbone, fallback)
                preds = np.where(confident, routed, global_pred_54)
                labels = np.where(confident, as_routed, -1)
                extra = {"c0_share": float((as_routed[confident] == 0).mean())
                         if confident.any() else float("nan"),
                         "fallback_share": float((~confident).mean()),
                         "wa_margin_p5": float(thresh)}
            else:
                labels = label_sets[policy]
                if policy in ("gapi_transplant", "dynamic_transplant", "seasonal"):
                    # Transplanted labels index the SAME frozen experts;
                    # cluster semantics differ by construction (documented).
                    pass
                preds = predict_with_labels(experts, frame, labels, backbone, fallback)
                extra = {"c0_share": float((labels == 0).mean())}
            _record(family, policy, preds, extra)

    for name, (model, features) in baselines.items():
        preds = np.asarray(model.predict(frame[features])).ravel()
        _record(name, "direct", preds, {"n_features": len(features)})
    return records, station_records, audit, pred_rows


def checkpoint_path(seed: int) -> Path:
    return CHECKPOINT_DIR / f"seed_{seed}.json"


def main(argv: list[str] | None = None) -> None:
    """Run the salvage. Pass argv=[] when calling from a notebook kernel."""
    args = parse_args(argv)
    config = load_configuration()
    seeds = [int(v) for v in args.seeds.split(",")] if args.seeds else list(config["seeds"])
    params = dict(config["model_params"])
    if args.smoke:
        seeds = [42]
        params["n_estimators"] = 100

    data = load_data(config)
    v0_features = _load_v0_features(PROJECT_ROOT)
    backbone = _load_backbone()
    routers = fit_routers(data["trainval"], v0_features, backbone)

    # WA-only ambiguity thresholds (5th percentile of static margins).
    pct = float(config["margin_fallback"]["wa_percentile"])
    wa_thresholds = {
        "v0": float(np.percentile(kmeans_margin(routers["v0"], data["trainval"]), pct)),
        "backbone": float(np.percentile(kmeans_margin(routers["backbone"], data["trainval"]), pct)),
    }
    print(f"WA margin p{pct:g}: v0={wa_thresholds['v0']:.4f} "
          f"backbone={wa_thresholds['backbone']:.4f}")
    print(f"train_rows={len(data['trainval'])} ece_v3_rows={len(data['v3'])} "
          f"families={config['families']} baselines={config['baselines']}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    all_records, all_stations, all_preds, audits = [], [], [], []
    for seed in seeds:
        ckpt = checkpoint_path(seed)
        if ckpt.exists() and not args.no_resume and not args.smoke:
            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            if "predictions" not in payload:
                print(f"seed={seed} checkpoint predates predictions; refitting")
            else:
                all_records.extend(payload["records"])
                all_stations.extend(payload["stations"])
                all_preds.extend(payload["predictions"])
                audits.append(payload["audit"])
                print(f"seed={seed} resumed ({len(payload['records'])} rows)")
                continue
        started = time.time()
        records, stations, audit, pred_rows = run_seed(
            seed, data, routers, v0_features, backbone, config, params, wa_thresholds)
        ckpt.write_text(json.dumps({"records": records, "stations": stations,
                                    "audit": audit, "predictions": pred_rows},
                                   indent=2), encoding="utf-8")
        all_records.extend(records)
        all_stations.extend(stations)
        all_preds.extend(pred_rows)
        audits.append(audit)
        print(f"seed={seed} complete in {time.time() - started:.1f}s")

    seed_metrics = pd.DataFrame(all_records)
    station_metrics = pd.DataFrame(all_stations)
    summary = seed_metrics.groupby(["family", "ece_input", "policy"], sort=False).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), bias_mean=("bias", "mean"),
        ubrmse_mean=("ubrmse", "mean"), r2_mean=("r2", "mean"),
        pearson_mean=("pearson", "mean")).reset_index()
    base = summary.query("policy == 'as_routed'").set_index(
        ["family", "ece_input"])["rmse_mean"].to_dict()

    def _delta(row) -> float:
        # Baselines have no as_routed row; compare them directly by RMSE.
        return float(row["rmse_mean"] - base[(row["family"], row["ece_input"])]) \
            if (row["family"], row["ece_input"]) in base else float("nan")

    summary["rmse_change_vs_as_routed"] = summary.apply(_delta, axis=1)

    seed_metrics.to_csv(EXP_DIR / "seed_metrics.csv", index=False)
    station_metrics.to_csv(EXP_DIR / "station_metrics.csv", index=False)
    summary.to_csv(EXP_DIR / "summary.csv", index=False)
    pd.DataFrame(all_preds).to_csv(EXP_DIR / "predictions_v3.csv", index=False)
    with (EXP_DIR / "routing_audit.json").open("w", encoding="utf-8") as handle:
        json.dump({"wa_thresholds": wa_thresholds, "seeds": audits}, handle, indent=2)

    print("\nROUTER SALVAGE SUMMARY (mean over seeds)")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nSTATION RMSE (mean over seeds)")
    pivot = station_metrics.groupby(["family", "ece_input", "policy", "station"],
                                    sort=False)["rmse"].mean().unstack("station")
    print(pivot.to_string(float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
