"""Automatic missingness-aware MoE routing bandaid (2.0).

Follow-up to derived_8.4-ece-router-salvage-1.1 with two hard constraints:

  1. No target at deploy time: gate thresholds, margin cutoffs, and the softmax
     temperature are fit on WA data only (ECE targets are eval-only).
  2. No global-model fallback: every MoE prediction is a convex combo w0*E0 + w1*E1
     of the SAME two frozen regime experts. Hard routing is the special case
     (w in {0, 1}). Global_Single_54 is reported as a single-regime baseline
     reference row only (policy=direct) and is never used as a fallback.

Bandaid logic (per row, per family):
  - Availability gate (input-only): router-unreliable when the full SMAP
    router-feature block is native-missing OR the overall router miss rate
    exceeds tau. Gated rows use the SMAP-free G_API auxiliary router.
  - Ambiguity blending (MoE-internal): rows whose static KMeans margin
    |d0 - d1| falls below the WA percentile cutoff are borderline and get a
    soft blend instead of a hard argmin.

WA-only calibration (ECE never touched):
  - Margin cutoffs (p5) + median margins from WA trainval.
  - Softmax temperature T per family: grid-selected on WA val (calibration
    experts fit on WA train only, scored on WA val).
  - Gate justification: WA val with synthetic full-SMAP masking shows the
    auxiliary router winning exactly in the masked regime for the Backbone54
    family; for V0-Full the masked-val comparison is neutral (static-masked
    slightly ahead), so the V0 gate application extrapolates beyond its own WA
    evidence and is reported as such.

Protocol (ECE strictly unseen):
  - Routers fit on WA trainval only (14,608 rows, 7 stations).
  - Experts and the Global_Single_54 baseline (XGBRegressor, formal-eval
    exact_params, cpu) fit on WA trainval only. ECE targets NEVER used for
    fitting or for any routing decision.
  - c0_only is a MANUAL oracle ceiling (deployable=false), kept only to
    quantify the remaining gap.
  - Global_Single_54 (policy=direct) is a deployable single-regime reference
    row only; it never feeds any MoE routing decision.

Resume: per-seed checkpoints under checkpoints/ keyed by a config+code hash;
reruns refit on any config or code change unless --no-resume is passed.
Smoke runs (--smoke) never read or write checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
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

# Series persisted per row for the <=5-line station charts: the deployable story
# set (as_routed, auto_soft, auto_hard) + the manual oracle ceiling (c0_only).
CHART_POLICIES = ("as_routed", "auto_soft", "auto_hard", "c0_only")


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


def config_code_hash() -> str:
    """Hash of config + runner source so stale checkpoints never resume silently."""
    digest = hashlib.sha256()
    digest.update((EXP_DIR / "config.yaml").read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()[:16]


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
    train_raw = pd.read_csv(train_dir / "train.csv", low_memory=False)
    val_raw = pd.read_csv(train_dir / "val.csv", low_memory=False)
    trainval = pd.concat([train_raw, val_raw], ignore_index=True)
    v3 = pd.read_csv(PROJECT_ROOT / config["datasets"]["ece_v3"] / "test.csv",
                     low_memory=False)
    # Canonical v3 schema guards (mirror tests/test_ece_v3_split.py).
    assert len(v3) == 150, f"Expected 150 v3 rows, got {len(v3)}"
    assert len(v3.columns) == 499, f"Expected 499 raw v3 columns, got {len(v3.columns)}"
    train = _prepare(train_raw, target)
    val = _prepare(val_raw, target)
    trainval = _prepare(trainval, target)
    v3 = _prepare(v3, target)
    counts = v3["station_id"].value_counts()
    assert len(counts) == 5 and bool((counts == 30).all()), \
        f"Expected 5 stations x 30 rows, got {counts.to_dict()}"
    assert v3["date"].min() == pd.Timestamp("2026-07-20")
    assert v3["date"].max() == pd.Timestamp("2026-08-19")
    smap_val = [c for c in v3.columns if "SMAP" in c and not c.endswith("_mask")]
    assert bool((v3[smap_val] != 0.0).all().all()), "No spurious SMAP 0.0 allowed"
    return {"train": train, "val": val, "trainval": trainval, "v3": v3,
            "target": target}


def static_dists(router, frame: pd.DataFrame) -> np.ndarray:
    """Per-row KMeans centroid distances under the router's scaler (n, 2)."""
    values = frame.loc[:, router.features].copy().fillna(router.means)
    return np.asarray(router.kmeans.transform(router.scaler.transform(values)),
                      dtype=float)


def kmeans_margin(router, frame: pd.DataFrame) -> np.ndarray:
    return np.abs(static_dists(router, frame)[:, 0] - static_dists(router, frame)[:, 1])


def softmax_neg_dists(dists: np.ndarray, temperature: float) -> np.ndarray:
    t = float(temperature)
    assert t > 0, "temperature must be positive"
    z = -np.asarray(dists, dtype=float) / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def smap_router_features(router_features: list[str]) -> list[str]:
    return [f for f in router_features if "SMAP" in f]


def availability_gate(frame: pd.DataFrame, router_features: list[str],
                      tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Input-only reliability flag. No target used.

    Returns (gated, miss_rate, smap_miss_rate). Gated when the full SMAP
    router-feature block is native-missing OR the overall router miss rate
    exceeds tau.
    """
    feats = list(router_features)
    miss = frame.loc[:, feats].isna().to_numpy(dtype=float)
    miss_rate = miss.mean(axis=1)
    smap = smap_router_features(feats)
    if smap:
        smap_miss = frame.loc[:, smap].isna().to_numpy(dtype=float).mean(axis=1)
        smap_block_missing = smap_miss >= 1.0 - 1e-12
    else:
        smap_miss = np.zeros(len(frame))
        smap_block_missing = np.zeros(len(frame), dtype=bool)
    gated = smap_block_missing | (miss_rate > float(tau))
    return gated.astype(bool), miss_rate, smap_miss


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
                target: str, params: dict) -> dict[int, XGBRegressor]:
    """Fit one XGBRegressor per non-empty cluster. Randomness comes only from
    params["random_state"], set per-seed by the caller."""
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


def expert_matrix(experts: dict[int, XGBRegressor], frame: pd.DataFrame,
                  features: list[str], fallback: float) -> np.ndarray:
    n = len(frame)
    mat = np.zeros((n, 2), dtype=float)
    for cluster in (0, 1):
        expert = experts.get(cluster)
        if expert is None:
            mat[:, cluster] = fallback
        else:
            mat[:, cluster] = np.asarray(
                expert.predict(frame.loc[:, features])).ravel()
    return mat


def calibrate_wa(train: pd.DataFrame, val: pd.DataFrame, routers: dict,
                 v0_features: list[str], backbone: list[str], config: dict,
                 params: dict) -> dict:
    """WA-only calibration. ECE is never touched here.

    - Fits calibration experts on WA train only (seed 42), scores T grid on WA
      val (clean + synthetic full-SMAP-masked copies).
    - Picks per-family T minimizing clean-val RMSE; verifies the gate regime
      (masked val) favors the G_API auxiliary router.
    """
    target = config["target_column"]
    grid = [float(t) for t in config["blend"]["temperature_grid"]]
    cal_params = dict(params)
    cal_params["random_state"] = 42
    y_val = val[target].to_numpy(dtype=float)
    fallback = float(train[target].mean())
    families = {"Clustering_V0_Full_k2": "v0", "Clustering_Backbone54_k2": "backbone"}

    cal_experts, cal_labels = {}, {}
    for family, key in families.items():
        labels = np.asarray(routers[key].predict(train)).ravel().astype(int)
        cal_labels[family] = labels
        cal_experts[family] = fit_experts(train, labels, backbone, target,
                                          cal_params)

    calibration_table: list[dict] = []
    temperatures: dict[str, float] = {}
    for family, key in families.items():
        router = routers[key]
        dval = static_dists(router, val)
        mat = expert_matrix(cal_experts[family], val, backbone, fallback)
        aux = np.asarray(routers["gapi"].predict(val)).ravel().astype(int)
        static_labels = np.asarray(routers[key].predict(val)).ravel().astype(int)
        # T grid on clean val.
        best_t, best_rmse = grid[0], float("inf")
        for t in grid:
            w = softmax_neg_dists(dval, t)
            preds = w[:, 0] * mat[:, 0] + w[:, 1] * mat[:, 1]
            rmse = float(root_mean_squared_error(y_val, preds))
            calibration_table.append({"family": family, "setting": "clean_val",
                                      "policy": f"soft_static_T{t:g}",
                                      "rmse": rmse})
            if rmse < best_rmse:
                best_rmse, best_t = rmse, t
        temperatures[key] = float(best_t)
        # Synthetic full-SMAP-mask regime on WA val (input masking only;
        # WA val targets used for scoring — ECE still untouched).
        masked = val.copy()
        for c in smap_router_features(router.features):
            if c in masked.columns:
                masked[c] = np.nan
        mat_m = expert_matrix(cal_experts[family], masked, backbone, fallback)
        static_m = np.asarray(routers[key].predict(masked)).ravel().astype(int)
        aux_m = np.asarray(routers["gapi"].predict(masked)).ravel().astype(int)
        for name, preds in [
            ("static_hard_masked", mat_m[np.arange(len(masked)), static_m]),
            ("aux_hard_masked", mat_m[np.arange(len(masked)), aux_m]),
        ]:
            calibration_table.append({"family": family, "setting": "smap_masked_val",
                                      "policy": name,
                                      "rmse": float(root_mean_squared_error(y_val, preds))})
        # Masked + clean reference rows for the hard routers.
        for name, preds in [
            ("static_hard_clean", mat[np.arange(len(val)), static_labels]),
            ("aux_hard_clean", mat[np.arange(len(val)), aux]),
        ]:
            calibration_table.append({"family": family, "setting": "clean_val",
                                      "policy": name,
                                      "rmse": float(root_mean_squared_error(y_val, preds))})
    return {"temperatures": temperatures, "table": calibration_table}


def run_seed(seed: int, data: dict, routers: dict, v0_features: list[str],
             backbone: list[str], config: dict, params: dict,
             wa_thresholds: dict[str, float], wa_medians: dict[str, float],
             temperatures: dict[str, float], tau: float
             ) -> tuple[list[dict], list[dict], dict, list[dict]]:
    trainval, frame = data["trainval"], data["v3"]
    target = data["target"]
    y = frame[target].to_numpy(dtype=float)
    fallback = float(trainval[target].mean())
    families = {"Clustering_V0_Full_k2": "v0", "Clustering_Backbone54_k2": "backbone"}
    station_ids = frame["station_id"].to_numpy()
    dates = frame["date"].dt.strftime("%Y-%m-%d").to_numpy()

    seed_params = dict(params)
    seed_params["random_state"] = seed

    # Frozen experts, fit on WA trainval only.
    # NOTE: routers are fixed at seed 42 by design (same as 1.1 and the formal
    # eval, whose delta additions are tied to seed-42 cluster labels); the
    # 5-seed std therefore measures expert-fit variance only, not end-to-end
    # routing variance.
    family_experts = {}
    for family, key in families.items():
        labels = np.asarray(routers[key].predict(trainval)).ravel().astype(int)
        family_experts[family] = fit_experts(trainval, labels, backbone, target,
                                             seed_params)

    # Single-regime global baseline reference, fit on WA trainval only
    # (same seed_params / exact_params as the experts). Reference row only:
    # its predictions never feed any MoE routing decision.
    baseline_features = {"Global_Single_54": backbone}
    baselines: dict[str, tuple[XGBRegressor, list[str]]] = {}
    for name, features in baseline_features.items():
        model = XGBRegressor(**seed_params)
        model.fit(trainval[features], trainval[target].to_numpy(dtype=float),
                  verbose=False)
        baselines[name] = (model, features)

    aux_labels = np.asarray(routers["gapi"].predict(frame)).ravel().astype(int)

    records, station_records, audit, pred_rows = [], [], {"seed": seed, "policies": {}}, []
    per_station = frame["station_id"].to_numpy()

    def _record(family: str, policy: str, preds: np.ndarray,
                weights: np.ndarray, extra: dict) -> None:
        metrics = compute_metrics(y, preds)
        deployable = policy in config["deployable"]
        records.append({"family": family, "ece_input": "v3",
                        "policy": policy, "seed": seed,
                        "deployable": deployable, **metrics})
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
                                  "y_pred": float(preds[i]),
                                  "w0": float(weights[i, 0])})

    def _record_direct(family: str, policy: str, preds: np.ndarray,
                       extra: dict) -> None:
        # Baseline reference rows (policy=direct): pooled + per-station metrics
        # only. No MoE weights exist, and baselines stay out of
        # predictions_v3.csv so the <=5-line chart budget is untouched.
        metrics = compute_metrics(y, preds)
        deployable = policy in config["deployable"]
        records.append({"family": family, "ece_input": "v3",
                        "policy": policy, "seed": seed,
                        "deployable": deployable, **metrics})
        for station in sorted(frame["station_id"].unique()):
            mask = per_station == station
            sm = compute_metrics(y[mask], preds[mask])
            station_records.append({"family": family, "ece_input": "v3",
                                    "policy": policy, "seed": seed,
                                    "station": station,
                                    "n": int(mask.sum()), **sm})
        audit["policies"].setdefault(family, {})[policy] = extra

    for family, key in families.items():
        router = routers[key]
        experts = family_experts[family]
        mat = expert_matrix(experts, frame, backbone, fallback)
        dists = static_dists(router, frame)
        margins = np.abs(dists[:, 0] - dists[:, 1])
        static_labels = np.asarray(routers[key].predict(frame)).ravel().astype(int)
        gated, miss_rate, smap_miss = availability_gate(frame, router.features, tau)
        thresh = float(wa_thresholds[key])
        med = float(wa_medians[key])
        temp = float(temperatures[key])
        # Diagnostic temperature: grid max (not WA-selected). At T=2.0 the
        # aux-anchored blend actually blends (w_aux ~ 0.86/0.14), exercising
        # the soft path that the WA-selected T=0.25 leaves near-hard.
        temp_diag = float(max(config["blend"]["temperature_grid"]))
        ambiguous = margins < thresh
        w_static_soft = softmax_neg_dists(dists, temp)
        # Aux-anchored pseudo-distances for gated rows: aux label gets
        # distance 0, the other gets the WA median margin (data-driven scale,
        # no hand-tuned constant), then the same WA-tuned softmax.
        pseudo = np.where((aux_labels == 0)[:, None],
                          np.column_stack([np.zeros(len(frame)),
                                           np.full(len(frame), med)]),
                          np.column_stack([np.full(len(frame), med),
                                           np.zeros(len(frame))]))
        w_aux_soft = softmax_neg_dists(pseudo, temp)
        w_aux_soft_diag = softmax_neg_dists(pseudo, temp_diag)
        w_static_soft_diag = softmax_neg_dists(dists, temp_diag)

        hard_static_preds = mat[np.arange(len(frame)), static_labels]
        soft_static_preds = w_static_soft[:, 0] * mat[:, 0] + w_static_soft[:, 1] * mat[:, 1]

        auto_hard_labels = np.where(gated, aux_labels, static_labels)
        auto_hard_preds = mat[np.arange(len(frame)), auto_hard_labels]

        w_auto_soft = np.where(gated[:, None], w_aux_soft,
                               np.where(ambiguous[:, None], w_static_soft,
                                        np.column_stack([static_labels == 0,
                                                         static_labels == 1]).astype(float)))
        auto_soft_preds = w_auto_soft[:, 0] * mat[:, 0] + w_auto_soft[:, 1] * mat[:, 1]
        w_auto_soft_t2 = np.where(gated[:, None], w_aux_soft_diag,
                                  np.where(ambiguous[:, None], w_static_soft_diag,
                                           np.column_stack([static_labels == 0,
                                                            static_labels == 1]).astype(float)))
        auto_soft_t2_preds = (w_auto_soft_t2[:, 0] * mat[:, 0]
                              + w_auto_soft_t2[:, 1] * mat[:, 1])

        w_auto_equal = np.where((gated | ambiguous)[:, None], 0.5,
                                np.column_stack([static_labels == 0,
                                                 static_labels == 1]).astype(float))
        auto_equal_preds = w_auto_equal[:, 0] * mat[:, 0] + w_auto_equal[:, 1] * mat[:, 1]

        c0_preds = mat[:, 0]

        hard_onehot = np.column_stack([static_labels == 0, static_labels == 1]).astype(float)
        w_hard_aux = np.column_stack([auto_hard_labels == 0, auto_hard_labels == 1]).astype(float)
        w_c0 = np.column_stack([np.ones(len(frame)), np.zeros(len(frame))])

        policy_outputs = {
            "as_routed": (hard_static_preds, hard_onehot),
            "soft_static": (soft_static_preds, w_static_soft),
            "auto_hard": (auto_hard_preds, w_hard_aux),
            "auto_soft": (auto_soft_preds, w_auto_soft),
            "auto_soft_T2": (auto_soft_t2_preds, w_auto_soft_t2),
            "auto_equal": (auto_equal_preds, w_auto_equal),
            "c0_only": (c0_preds, w_c0),
        }
        for policy in config["policies"]:
            preds, weights = policy_outputs[policy]
            c0_share = float((weights[:, 0] > 0.5).mean())
            extra = {
                "c0_share": c0_share,
                "deployable": bool(policy in config["deployable"]),
                "diagnostic": bool(policy == "auto_soft_T2"),
                "gate_share": float(gated.mean()),
                "ambiguous_share": float(ambiguous.mean()),
                "mean_miss_rate": float(miss_rate.mean()),
                "mean_smap_miss_rate": float(smap_miss.mean()),
                "mean_w0": float(weights[:, 0].mean()),
                "temperature": temp_diag if policy == "auto_soft_T2" else temp,
                "wa_margin_p5": thresh,
                "wa_margin_median": med,
            }
            _record(family, policy, preds, weights, extra)

    for name, (model, features) in baselines.items():
        preds = np.asarray(model.predict(frame[features])).ravel()
        _record_direct(name, "direct", preds,
                       {"n_features": len(features),
                        "deployable": True,
                        "reference_only": True})

    return records, station_records, audit, pred_rows


def checkpoint_path(seed: int) -> Path:
    return CHECKPOINT_DIR / f"seed_{seed}.json"


def main(argv: list[str] | None = None) -> None:
    """Run the automatic salvage. Pass argv=[] when calling from a notebook kernel."""
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

    # WA-only ambiguity cutoffs + median scales (5th percentile of static margins).
    pct = float(config["blend"]["wa_percentile"])
    wa_thresholds = {
        "v0": float(np.percentile(kmeans_margin(routers["v0"], data["trainval"]), pct)),
        "backbone": float(np.percentile(kmeans_margin(routers["backbone"], data["trainval"]), pct)),
    }
    wa_medians = {
        "v0": float(np.median(kmeans_margin(routers["v0"], data["trainval"]))),
        "backbone": float(np.median(kmeans_margin(routers["backbone"], data["trainval"]))),
    }
    tau = float(config["gate"]["tau_miss_rate"])
    print(f"WA margin p{pct:g}: v0={wa_thresholds['v0']:.4f} "
          f"backbone={wa_thresholds['backbone']:.4f}")
    print(f"WA margin median: v0={wa_medians['v0']:.4f} "
          f"backbone={wa_medians['backbone']:.4f}")

    # WA-only temperature + gate calibration (ECE never touched).
    calibration = calibrate_wa(data["train"], data["val"], routers, v0_features,
                               backbone, config, params)
    temperatures = calibration["temperatures"]
    print(f"WA-tuned temperatures: {temperatures}")
    print("WA calibration (train-fit / val-scored,incl. synthetic SMAP masking):")
    for row in calibration["table"]:
        print(f"  {row['family']} {row['setting']} {row['policy']}: rmse={row['rmse']:.6f}")

    print(f"train_rows={len(data['trainval'])} ece_v3_rows={len(data['v3'])} "
          f"families={config['families']} policies={config['policies']} "
          f"baselines={config.get('baselines', [])}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run_hash = config_code_hash()
    print(f"config+code hash: {run_hash}")
    all_records, all_stations, all_preds, audits = [], [], [], []
    for seed in seeds:
        ckpt = checkpoint_path(seed)
        # Smoke runs never read or write checkpoints: their 100-tree fits must
        # not poison (or reuse) full-run checkpoints.
        if ckpt.exists() and not args.no_resume and not args.smoke:
            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            if payload.get("config_code_hash") != run_hash:
                print(f"seed={seed} checkpoint hash mismatch "
                      f"({payload.get('config_code_hash')} != {run_hash}); refitting")
            else:
                all_records.extend(payload["records"])
                all_stations.extend(payload["stations"])
                all_preds.extend(payload["predictions"])
                audits.append(payload["audit"])
                print(f"seed={seed} resumed ({len(payload['records'])} rows)")
                continue
        started = time.time()
        records, stations, audit, pred_rows = run_seed(
            seed, data, routers, v0_features, backbone, config, params,
            wa_thresholds, wa_medians, temperatures, tau)
        if not args.smoke:
            ckpt.write_text(json.dumps({"records": records, "stations": stations,
                                        "audit": audit, "predictions": pred_rows,
                                        "calibration": calibration,
                                        "wa_thresholds": wa_thresholds,
                                        "wa_medians": wa_medians,
                                        "temperatures": temperatures,
                                        "config_code_hash": run_hash},
                                       indent=2), encoding="utf-8")
        all_records.extend(records)
        all_stations.extend(stations)
        all_preds.extend(pred_rows)
        audits.append(audit)
        print(f"seed={seed} complete in {time.time() - started:.1f}s")

    seed_metrics = pd.DataFrame(all_records)
    station_metrics = pd.DataFrame(all_stations)
    summary = seed_metrics.groupby(["family", "ece_input", "policy", "deployable"],
                                   sort=False).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), bias_mean=("bias", "mean"),
        ubrmse_mean=("ubrmse", "mean"), r2_mean=("r2", "mean"),
        pearson_mean=("pearson", "mean")).reset_index()
    base = summary.query("policy == 'as_routed'").set_index(
        ["family", "ece_input"])["rmse_mean"].to_dict()

    def _delta(row) -> float:
        return float(row["rmse_mean"] - base[(row["family"], row["ece_input"])]) \
            if (row["family"], row["ece_input"]) in base else float("nan")

    summary["rmse_change_vs_as_routed"] = summary.apply(_delta, axis=1)
    oracle = summary.query("policy == 'c0_only'").set_index(
        ["family", "ece_input"])["rmse_mean"].to_dict()

    def _gap(row) -> float:
        key = (row["family"], row["ece_input"])
        return float(row["rmse_mean"] - oracle[key]) if key in oracle else float("nan")

    summary["rmse_gap_vs_oracle_c0"] = summary.apply(_gap, axis=1)

    seed_metrics.to_csv(EXP_DIR / "seed_metrics.csv", index=False)
    station_metrics.to_csv(EXP_DIR / "station_metrics.csv", index=False)
    summary.to_csv(EXP_DIR / "summary.csv", index=False)
    pd.DataFrame(all_preds).to_csv(EXP_DIR / "predictions_v3.csv", index=False)
    pd.DataFrame(calibration["table"]).to_csv(EXP_DIR / "wa_calibration.csv", index=False)
    with (EXP_DIR / "routing_audit.json").open("w", encoding="utf-8") as handle:
        json.dump({"wa_thresholds": wa_thresholds, "wa_medians": wa_medians,
                   "temperatures": temperatures, "tau_miss_rate": tau,
                   "config_code_hash": run_hash,
                   "calibration": calibration["table"],
                   "seeds": audits}, handle, indent=2)

    print("\nAUTO SALVAGE SUMMARY (mean over seeds; deployable=true needs no target)")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nSTATION RMSE (mean over seeds)")
    pivot = station_metrics.groupby(["family", "ece_input", "policy", "station"],
                                    sort=False)["rmse"].mean().unstack("station")
    print(pivot.to_string(float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
