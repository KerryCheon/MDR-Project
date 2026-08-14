#!/usr/bin/env python3
"""Val-selected delta protocol for derived_8.4-formal-eval-1.0.

The historical per-regime delta feature selection (derived_8.4-eval-1.1) ranked
candidate-pool features by ``gain_rank + |spearman corr(feature, per-cluster TEST
residual)|`` — the residual term uses the 2023-2025 test period, which a reviewer
can legitimately scrutinize. This script re-runs the same selection with an honest
temporal holdout:

  1. Router fit on TRAIN only (2017-2020); labels for train + val (2021-2022).
  2. Backbone (0,0) experts fit on TRAIN only -> predictions on VAL.
  3. Candidate-pool evidence rebuilt on val: residual association recomputed from VAL
     residuals (V0-50 global backbone fit on train only, mirroring fs20's use of the
     V0 baseline); gain scores refit on TRAIN only (500-tree proxy). The MI ranking
     and seed-frequency priors are reused from fs20's candidate_pool.csv (trainval-based
     prior evidence, not test-period selection).
  4. Per-cluster delta rankings with VAL residuals (identical ranking formula to eval-1.1).
  5. 9-point (c0, c1 in {0, 5, 10}) delta grid evaluated on VAL (experts fit on TRAIN
     only, seed 42); winner = best val pooled R2 (tie-break RMSE, then (c0, c1)
     lexicographic).

Outputs (in the experiment dir): val_selected_deltas.json (per-strategy rankings,
grid rows, winners, additions), val_grid_summary.csv, val_pool.csv.

Usage:
    python select_deltas_val.py                  # full run (CUDA by default)
    python select_deltas_val.py --smoke          # n_estimators=100, CPU (data_version -1 artifacts)
    python select_deltas_val.py --device cpu     # CPU run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from eval_formal.data import load_experiment_data  # noqa: E402
from eval_formal.evaluator import compute_metrics  # noqa: E402
from eval_formal.routers import get_router  # noqa: E402

PROJECT_ROOT = EXP_DIR.parents[2]


def build_val_pool(data, config, gain_scores: dict[str, float],
                   residual_scores: dict[str, float]) -> list[str]:
    """Rebuild the fs20 candidate pool with val-based evidence.

    Identical formula to derived_8.4-feature-selection-2.0/fs20/search.py
    ``_candidate_pool`` (support = sum of MI/seed/gain/residual top-memberships,
    consensus rank = mean of the four ranks) but with:
      - gain_scores   from a TRAIN-only proxy fit (this script),
      - residual_scores from VAL-period residuals (this script),
      - mi_rank + seed_frequency priors reused from fs20's candidate_pool.csv
        (trainval-based prior evidence, not test-period selection).
    """
    source_order = data.source_order
    pool_size = int(config["val_selection"]["candidate_pool_size"])
    mi_k = int(config["val_selection"].get("canonical_mi_k", 300))

    fs20_pool_path = PROJECT_ROOT / Path(config["val_selection"]["fs20_candidate_pool"])
    if not fs20_pool_path.exists():
        raise FileNotFoundError(f"fs20 candidate pool not found: {fs20_pool_path}")
    fs20 = pd.read_csv(fs20_pool_path)
    fs20_map = {row["feature"]: row for _, row in fs20.iterrows()}

    def rank_scores(scores: dict[str, float], order: list[str]) -> dict[str, int]:
        ordered = sorted(order, key=lambda f: (-scores.get(f, 0.0), f))
        return {feature: index + 1 for index, feature in enumerate(ordered)}

    mi_rank: dict[str, float] = {}
    seed_frequency: dict[str, float] = {}
    for feature in source_order:
        prior = fs20_map.get(feature)
        if prior is not None:
            mi_rank[feature] = float(prior["mi_rank"]) if pd.notna(prior["mi_rank"]) \
                else len(source_order) + 1
            seed_frequency[feature] = float(prior["seed_frequency"])
        else:
            mi_rank[feature] = len(source_order) + 1
            seed_frequency[feature] = 0.0
    seed_rank = rank_scores(seed_frequency, source_order)
    gain_rank = rank_scores(gain_scores, source_order)
    residual_rank = rank_scores(residual_scores, source_order)
    gain_top = {f for f, r in gain_rank.items() if r <= pool_size}
    residual_top = {f for f, r in residual_rank.items() if r <= pool_size}

    records = []
    for feature in source_order:
        support = (
            int(mi_rank[feature] <= mi_k)
            + int(seed_frequency[feature] > 0)
            + int(feature in gain_top)
            + int(feature in residual_top)
        )
        records.append({
            "feature": feature,
            "support": support,
            "consensus_rank": float(np.mean([mi_rank[feature], seed_rank[feature],
                                            gain_rank[feature], residual_rank[feature]])),
            "mi_rank": mi_rank[feature],
            "seed_frequency": seed_frequency[feature],
            "gain_rank": gain_rank[feature],
            "residual_rank": residual_rank[feature],
            "gain": gain_scores.get(feature, 0.0),
            "residual_association": residual_scores.get(feature, 0.0),
        })
    evidence = pd.DataFrame(records).sort_values(
        ["support", "consensus_rank", "feature"], ascending=[False, True, True]
    )
    multi_source = evidence.loc[evidence["support"] >= 2]
    selected = multi_source.head(pool_size)
    if len(selected) < pool_size:
        selected = pd.concat([
            selected,
            evidence.loc[~evidence["feature"].isin(selected["feature"])].head(
                pool_size - len(selected)
            ),
        ])
    selected.to_csv(EXP_DIR / "val_pool.csv", index=False)
    return selected["feature"].tolist()


def delta_rankings_val(data, config, strategy: str, pool: list[str],
                       gain_scores: dict[str, float],
                       params: dict, router_seed: int) -> tuple[dict[str, list[str]], dict]:
    """Per-cluster delta rankings on VAL residuals (eval-1.1 formula, val residuals)."""
    train = data.train
    val = data.val
    backbone = data.shared_backbone_54
    external = [f for f in pool if f not in set(backbone)]

    router = get_router(strategy, data.v0_features, backbone_54=backbone,
                        seed=router_seed, device=params["device"])
    router.fit(train)
    labels_train = np.asarray(router.predict(train)).ravel().astype(int)
    labels_val = np.asarray(router.predict(val)).ravel().astype(int)

    y_train = train[data.target].to_numpy(dtype=float)
    y_val = val[data.target].to_numpy(dtype=float)

    # Backbone (0,0) experts fit on TRAIN only -> VAL predictions -> residuals.
    predictions_val = np.zeros(len(val), dtype=float)
    for cluster in (0, 1):
        mask_train = labels_train == cluster
        mask_val = labels_val == cluster
        expert = XGBRegressor(**params)
        expert.fit(train.loc[mask_train, backbone], y_train[mask_train], verbose=False)
        predictions_val[mask_val] = np.asarray(
            expert.predict(val.loc[mask_val, backbone])).ravel()

    rankings: dict[str, list[str]] = {}
    for cluster in (0, 1):
        mask_val = labels_val == cluster
        residual = pd.Series(
            y_val[mask_val] - predictions_val[mask_val], index=val.index[mask_val]
        )
        correlations: dict[str, float] = {}
        for feat in external:
            corr = val.loc[mask_val, feat].corr(residual, method="spearman")
            correlations[feat] = float(abs(corr)) if pd.notna(corr) else 0.0
        gain_sorted = sorted(external, key=lambda f: gain_scores.get(f, 0.0), reverse=True)
        gain_rank = {f: i for i, f in enumerate(gain_sorted)}
        corr_sorted = sorted(external, key=lambda f: correlations[f], reverse=True)
        corr_rank = {f: i for i, f in enumerate(corr_sorted)}
        ranked = sorted(external, key=lambda f: (gain_rank[f] + corr_rank[f], f))
        rankings[str(cluster)] = ranked[: int(config["val_selection"]["max_additions"])]
    return rankings, {"labels_val": labels_val, "y_val": y_val}


def evaluate_val_grid(data, config, strategy: str, additions_rankings: dict[str, list[str]],
                      params: dict, router_seed: int) -> pd.DataFrame:
    """9-point delta grid evaluated on VAL (experts fit on TRAIN only, seed 42)."""
    train = data.train
    val = data.val
    backbone = data.shared_backbone_54

    router = get_router(strategy, data.v0_features, backbone_54=backbone,
                        seed=router_seed, device=params["device"])
    router.fit(train)
    labels_train = np.asarray(router.predict(train)).ravel().astype(int)
    labels_val = np.asarray(router.predict(val)).ravel().astype(int)

    y_train = train[data.target].to_numpy(dtype=float)
    y_val = val[data.target].to_numpy(dtype=float)

    rows = []
    for c0 in config["val_selection"]["delta_addition_counts"]:
        for c1 in config["val_selection"]["delta_addition_counts"]:
            additions = {
                "0": additions_rankings["0"][:c0],
                "1": additions_rankings["1"][:c1],
            }
            predictions = np.zeros(len(val), dtype=float)
            for cluster in (0, 1):
                features = [*backbone, *additions[str(cluster)]]
                mask_train = labels_train == cluster
                mask_val = labels_val == cluster
                if not mask_train.any():
                    predictions[mask_val] = float(np.mean(y_train))
                    continue
                expert = XGBRegressor(**params)
                expert.fit(train.loc[mask_train, features], y_train[mask_train], verbose=False)
                predictions[mask_val] = np.asarray(
                    expert.predict(val.loc[mask_val, features])).ravel()
            m = compute_metrics(y_val, predictions)
            rows.append({
                "strategy_name": strategy,
                "cluster_0_count": c0,
                "cluster_1_count": c1,
                "val_r2": m["r2"],
                "val_rmse": m["rmse"],
                "val_mae": m["mae"],
                "val_bias": m["bias"],
                "n_val": int(len(val)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="n_estimators=100 + CPU (never reused by the real run).")
    parser.add_argument("--device", default=None, help="Override model device (e.g. cpu).")
    args = parser.parse_args()

    with open(EXP_DIR / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data = load_experiment_data(PROJECT_ROOT, config)

    params = dict(config["model"]["proxy_params"])
    if args.smoke:
        params["n_estimators"] = 100
        params["device"] = "cpu"
    if args.device:
        params["device"] = args.device
    params["n_jobs"] = 1
    params["random_state"] = int(config["model"]["router_seed"])

    router_seed = int(config["model"]["router_seed"])

    print("=" * 70, flush=True)
    print("Val-selected delta protocol (derived_8.4-formal-eval-1.0)", flush=True)
    print("=" * 70, flush=True)
    print(f"[Data] Train={len(data.train)} Val={len(data.val)} Test={len(data.test)}", flush=True)
    print(f"[Model] n_estimators={params['n_estimators']} device={params['device']}", flush=True)

    # 1. Train-only gain proxy (all features, 500 trees) — replaces the trainval proxy.
    print("[Gain] Fitting train-only gain proxy...", flush=True)
    proxy = XGBRegressor(**params)
    proxy.fit(data.train.loc[:, data.feature_columns],
              data.train[data.target].to_numpy(dtype=float), verbose=False)
    gain_scores = dict(zip(data.feature_columns,
                           proxy.feature_importances_.astype(float)))

    # 2. Val residuals from a V0-50 global backbone fit on TRAIN only (mirrors fs20's
    #    residual_association, which used the V0 baseline on the test period).
    print("[Residuals] V0 backbone on train -> val residuals...", flush=True)
    v0_params = dict(params)
    v0_params["random_state"] = router_seed
    v0_expert = XGBRegressor(**v0_params)
    v0_expert.fit(data.train.loc[:, data.v0_features],
                  data.train[data.target].to_numpy(dtype=float), verbose=False)
    v0_pred_val = np.asarray(v0_expert.predict(data.val.loc[:, data.v0_features])).ravel()
    residual = pd.Series(
        data.val[data.target].to_numpy(dtype=float) - v0_pred_val, index=data.val.index
    )
    residual_scores: dict[str, float] = {}
    for feature in data.feature_columns:
        corr = data.val[feature].corr(residual, method="spearman")
        residual_scores[feature] = float(abs(corr)) if pd.notna(corr) else 0.0

    # 3. Candidate pool rebuilt on val evidence.
    pool = build_val_pool(data, config, gain_scores, residual_scores)
    print(f"[Pool] Val-rebuilt candidate pool: {len(pool)} features", flush=True)

    # 4-5. Per-strategy rankings + 9-point val grid + val winner.
    grid_rows: list[dict] = []
    output: dict = {
        "pool": pool,
        "gain_scores": gain_scores,
        "n_estimators": int(params["n_estimators"]),
        "device": str(params["device"]),
        "router_seed": router_seed,
        "strategies": {},
    }
    for strategy in config["val_selection"]["strategies"]:
        print(f"[{strategy}] Ranking deltas on val residuals...", flush=True)
        rankings, _ = delta_rankings_val(data, config, strategy, pool, gain_scores,
                                         params, router_seed)
        print(f"[{strategy}] Evaluating 9-point grid on val...", flush=True)
        grid = evaluate_val_grid(data, config, strategy, rankings, params, router_seed)
        grid_rows.append(grid)
        best = grid.sort_values(["val_r2", "val_rmse", "cluster_0_count", "cluster_1_count"],
                                ascending=[False, True, True, True]).iloc[0]
        winner = (int(best["cluster_0_count"]), int(best["cluster_1_count"]))
        output["strategies"][strategy] = {
            "winner": winner,
            "additions": {k: list(v) for k, v in rankings.items()},
            "val_winner_r2": float(best["val_r2"]),
            "val_winner_rmse": float(best["val_rmse"]),
        }
        print(f"[{strategy}] winner (c0={winner[0]}, c1={winner[1]}) "
              f"val_r2={best['val_r2']:.5f}", flush=True)

    df_grid = pd.concat(grid_rows, ignore_index=True)
    df_grid.to_csv(EXP_DIR / "val_grid_summary.csv", index=False)
    with open(EXP_DIR / config["val_selection"]["output_json"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[Artifacts] Wrote {EXP_DIR / 'val_selected_deltas.json'}, "
          f"{EXP_DIR / 'val_grid_summary.csv'}, {EXP_DIR / 'val_pool.csv'}", flush=True)


if __name__ == "__main__":
    main()
