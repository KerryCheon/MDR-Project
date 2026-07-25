"""Generate all-station, all-month, temporal, and input-sufficiency evidence."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from fs21.artifacts import (
    atomic_write_csv,
    atomic_write_json,
    completion_is_valid,
    invalidate_completion,
    write_completion,
)
from fs21.diagnostics import (
    candidate_diagnostic_ids,
    correlation_components,
    feature_family_summary,
    fit_window_ledger,
    joint_component_importance,
    residual_distribution,
    seasonal_climatology,
    station_input_diagnostics,
    station_sufficiency_classification,
    write_metric_tables,
)
from fs21.global_pipeline import build_context


REQUIRED = [
    "metrics_by_overall.csv",
    "metrics_by_year.csv",
    "metrics_by_month.csv",
    "metrics_by_station.csv",
    "metrics_by_station_year.csv",
    "station_sufficiency_classification.csv",
    "selected_vs_all_station_intervals.csv",
    "station_input_diagnostics.csv",
    "station_month_residual_distribution.csv",
    "station_month_climatology.csv",
    "fit_window_predictions.csv.gz",
    "fit_window_metrics.csv",
    "correlation_components.json",
    "correlation_edges.csv",
    "correlation_joint_permutation.csv",
    "feature_family_summary.csv",
    "transition_month_highlights.csv",
    "diagnostic_manifest.json",
]


def run(*, device: str, workers: int, smoke: bool) -> None:
    context = build_context(device=device, workers=workers, smoke=smoke)
    stage = context.artifact_root / "stages" / "10_station_temporal_diagnostics"
    if completion_is_valid(stage, REQUIRED):
        return
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    candidate_features_path = (
        context.artifact_root / "stages" / "09_consensus" / "candidate_features.json"
    )
    candidate_features = json.loads(
        candidate_features_path.read_text(encoding="utf-8")
    )
    candidate_ids = candidate_diagnostic_ids(candidate_features)
    ledger = pd.read_csv(
        context.artifact_root / "stages" / "06_candidate_oof" / "oof_predictions.csv.gz"
    )
    tables = write_metric_tables(
        ledger,
        candidate_ids,
        stage,
        variance_epsilon=float(
            context.config["decision"]["r2_zero_variance_epsilon"]
        ),
    )
    classifications, intervals = station_sufficiency_classification(
        ledger,
        tables,
        candidate_ids,
        replicates=int(context.config["bootstrap"]["replicates"]),
        seed=int(context.config["bootstrap"]["seed"]),
    )
    atomic_write_csv(
        classifications, stage / "station_sufficiency_classification.csv"
    )
    atomic_write_csv(intervals, stage / "selected_vs_all_station_intervals.csv")

    final_features = list(candidate_features["benchmark_challenger_features"])
    input_diagnostics = station_input_diagnostics(
        context.frame,
        final_features,
        context.config,
    )
    atomic_write_csv(input_diagnostics, stage / "station_input_diagnostics.csv")
    atomic_write_csv(
        residual_distribution(ledger, candidate_ids),
        stage / "station_month_residual_distribution.csv",
    )
    atomic_write_csv(
        seasonal_climatology(context.frame, context.config),
        stage / "station_month_climatology.csv",
    )

    fit_ledger = fit_window_ledger(
        context.frame,
        {
            "V0": list(context.controls["V0"]),
            "challenger": final_features,
        },
        context.config,
        device=context.device,
        smoke=context.smoke,
    )
    atomic_write_csv(fit_ledger, stage / "fit_window_predictions.csv.gz")
    fit_rows = []
    for candidate in sorted(fit_ledger["candidate"].unique()):
        metric = write_metric_tables(
            fit_ledger,
            {candidate: candidate},
            stage / "fit_window_detail" / candidate,
            variance_epsilon=float(
                context.config["decision"]["r2_zero_variance_epsilon"]
            ),
        )["overall"]
        row = metric.iloc[0].to_dict()
        row["candidate"] = candidate
        fit_rows.append(row)
    atomic_write_csv(pd.DataFrame(fit_rows), stage / "fit_window_metrics.csv")

    latest_origin = max(int(value) for value in context.config["folds"]["outer_origins"])
    correlation_training = context.frame.loc[context.frame["_year"] < latest_origin]
    components, edges = correlation_components(
        correlation_training,
        final_features,
        threshold=float(context.config["diagnostics"]["correlation_threshold"]),
    )
    atomic_write_json(
        stage / "correlation_components.json",
        {
            "threshold": float(
                context.config["diagnostics"]["correlation_threshold"]
            ),
            "training_years": sorted(correlation_training["_year"].unique().tolist()),
            "components": components,
            "diagnostic_only": True,
        },
    )
    if edges.empty:
        edges = pd.DataFrame(columns=["feature_a", "feature_b", "spearman"])
    atomic_write_csv(edges, stage / "correlation_edges.csv")
    component_importance = joint_component_importance(
        context.frame,
        final_features,
        components,
        context.config,
        origin=latest_origin,
        device=context.device,
        smoke=context.smoke,
    )
    if component_importance.empty:
        component_importance = pd.DataFrame(
            columns=[
                "component_id",
                "features",
                "active_features",
                "component_size",
                "baseline_rmse",
                "joint_permutation_rmse",
                "importance_delta_rmse",
                "diagnostic_only",
            ]
        )
    atomic_write_csv(
        component_importance, stage / "correlation_joint_permutation.csv"
    )
    atomic_write_csv(
        feature_family_summary(final_features), stage / "feature_family_summary.csv"
    )

    transition_months = set(
        int(value) for value in context.config["diagnostics"]["transition_months"]
    )
    month_table = tables["month"].copy()
    month_table["transition_month"] = month_table["month"].isin(transition_months)
    transition = month_table.loc[month_table["transition_month"]].copy()
    if not transition.empty:
        threshold = month_table.groupby("candidate")["RMSE"].transform(
            lambda values: values.quantile(0.9)
        )
        month_table["high_error_month"] = month_table["RMSE"] >= threshold
        transition = month_table.loc[
            month_table["transition_month"] & month_table["high_error_month"]
        ]
    atomic_write_csv(transition, stage / "transition_month_highlights.csv")
    atomic_write_json(
        stage / "diagnostic_manifest.json",
        {
            "candidate_ids": candidate_ids,
            "stations_in_scope": sorted(context.frame["station_id"].unique().tolist()),
            "station_pruning": False,
            "months_reported": list(range(1, 13)),
            "new_inputs_added": False,
            "fit_window_is_promotion_input": False,
            "correlation_diagnostic_can_change_selection": False,
        },
    )
    write_completion(stage, REQUIRED)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(device=args.device, workers=args.workers, smoke=args.smoke)
    print(json.dumps({"status": "complete"}, indent=2))


if __name__ == "__main__":
    main()

