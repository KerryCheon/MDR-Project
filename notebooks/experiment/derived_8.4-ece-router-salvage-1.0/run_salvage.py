"""Router-only salvage for Clustering_V0_Full_k2 / Clustering_Backbone54_k2 on ECE.

Protocol (ECE strictly unseen):
  - Routers fit on WA trainval only (14,608 rows, 7 stations).
  - Experts (XGBRegressor, formal-eval exact_params, cpu) fit on WA trainval
    cluster subsets only. ECE targets are NEVER used for fitting.
  - Routing policies are inference-time label overrides applied to the SAME
    frozen experts. ECE is used for evaluation only.

Policies per family x ECE input (zero-filled vs native-missing):
  as_routed, c0_only, c1_only, gapi_transplant, dynamic_transplant,
  seasonal, margin_fallback (ambiguous static rows -> Global_Single expert).

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
    zero = pd.read_csv(PROJECT_ROOT / config["datasets"]["ece_zero_filled"] / "test.csv",
                       low_memory=False)
    native = pd.read_csv(PROJECT_ROOT / config["datasets"]["ece_native_missing"] / "test.csv",
                         low_memory=False)
    trainval = _prepare(train, target)
    zero = _prepare(zero, target)
    native = _prepare(native, target)
    keys = ["station_id", "date", target]
    if not zero[keys].equals(native[keys]):
        raise ValueError("Zero-filled and native-missing ECE rows are not aligned.")
    return {"trainval": trainval, "zero": zero, "native": native, "target": target}


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
             wa_thresholds: dict[str, float]) -> tuple[list[dict], list[dict], dict]:
    trainval, zero, native = data["trainval"], data["zero"], data["native"]
    target = data["target"]
    y_zero = zero[target].to_numpy(dtype=float)
    fallback = float(trainval[target].mean())
    families = {"Clustering_V0_Full_k2": "v0", "Clustering_Backbone54_k2": "backbone"}

    seed_params = dict(params)
    seed_params["random_state"] = seed

    # Frozen experts, fit on WA trainval only.
    family_experts, family_labels = {}, {}
    for family, key in families.items():
        labels = np.asarray(routers[key].predict(trainval)).ravel().astype(int)
        family_labels[family] = labels
        family_experts[family] = fit_experts(trainval, labels, backbone, target,
                                             seed_params, seed)
    global_model = XGBRegressor(**seed_params)
    global_model.fit(trainval[backbone], trainval[target].to_numpy(dtype=float),
                     verbose=False)

    label_sets = {
        "as_routed": None,  # per-family static labels
        "c0_only": np.zeros(len(zero), dtype=int),
        "c1_only": np.ones(len(zero), dtype=int),
        "gapi_transplant": np.asarray(routers["gapi"].predict(zero)).ravel().astype(int),
        "dynamic_transplant": np.asarray(routers["dynamic"].predict(zero)).ravel().astype(int),
        "seasonal": np.asarray(routers["seasonal"].predict(zero)).ravel().astype(int),
    }

    records, station_records, audit = [], [], {"seed": seed, "policies": {}}
    for input_name, frame in (("zero", zero), ("native", native)):
        y = y_zero  # aligned frames share targets
        global_pred = np.asarray(global_model.predict(frame[backbone])).ravel()
        for family, key in families.items():
            as_routed = np.asarray(routers[key].predict(frame)).ravel().astype(int)
            margins = kmeans_margin(routers[key], frame)
            experts = family_experts[family]
            for policy in config["policies"]:
                if policy == "as_routed":
                    labels = as_routed
                    preds = predict_with_labels(experts, frame, labels, backbone, fallback)
                    extra = {"c0_share": float((labels == 0).mean())}
                elif policy == "margin_fallback":
                    thresh = wa_thresholds[key]
                    confident = margins >= thresh
                    routed = predict_with_labels(experts, frame, as_routed, backbone, fallback)
                    preds = np.where(confident, routed, global_pred)
                    labels = np.where(confident, as_routed, -1)
                    extra = {"c0_share": float((as_routed[confident] == 0).mean())
                             if confident.any() else float("nan"),
                             "fallback_share": float((~confident).mean()),
                             "wa_margin_p5": float(thresh)}
                else:
                    labels = label_sets[policy] if policy != "as_routed" else as_routed
                    if policy in ("gapi_transplant", "dynamic_transplant", "seasonal"):
                        # Transplanted labels index the SAME frozen experts;
                        # cluster semantics differ by construction (documented).
                        pass
                    preds = predict_with_labels(experts, frame, labels, backbone, fallback)
                    extra = {"c0_share": float((labels == 0).mean())}
                metrics = compute_metrics(y, preds)
                records.append({"family": family, "ece_input": input_name,
                                "policy": policy, "seed": seed, **metrics})
                per_station = frame["station_id"].to_numpy()
                for station in sorted(frame["station_id"].unique()):
                    mask = per_station == station
                    sm = compute_metrics(y[mask], preds[mask])
                    station_records.append({"family": family, "ece_input": input_name,
                                            "policy": policy, "seed": seed,
                                            "station": station,
                                            "n": int(mask.sum()), **sm})
                audit["policies"].setdefault(family, {}).setdefault(input_name, {})[policy] = extra
    return records, station_records, audit


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
    print(f"train_rows={len(data['trainval'])} ece_rows={len(data['zero'])} "
          f"features=54 families={config['families']}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    all_records, all_stations, audits = [], [], []
    for seed in seeds:
        ckpt = checkpoint_path(seed)
        if ckpt.exists() and not args.no_resume and not args.smoke:
            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            all_records.extend(payload["records"])
            all_stations.extend(payload["stations"])
            audits.append(payload["audit"])
            print(f"seed={seed} resumed ({len(payload['records'])} rows)")
            continue
        started = time.time()
        records, stations, audit = run_seed(seed, data, routers, v0_features,
                                            backbone, config, params, wa_thresholds)
        ckpt.write_text(json.dumps({"records": records, "stations": stations,
                                    "audit": audit}, indent=2), encoding="utf-8")
        all_records.extend(records)
        all_stations.extend(stations)
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
        ["family", "ece_input"])["rmse_mean"]
    summary["rmse_change_vs_as_routed"] = summary.apply(
        lambda r: r["rmse_mean"] - base.loc[(r["family"], r["ece_input"])], axis=1)

    seed_metrics.to_csv(EXP_DIR / "seed_metrics.csv", index=False)
    station_metrics.to_csv(EXP_DIR / "station_metrics.csv", index=False)
    summary.to_csv(EXP_DIR / "summary.csv", index=False)
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
