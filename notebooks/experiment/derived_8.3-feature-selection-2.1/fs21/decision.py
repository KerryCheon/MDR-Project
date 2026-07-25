"""Predeclared global, beta, MoE, and benchmark decision rules."""

from __future__ import annotations

from collections.abc import Iterable


def global_candidate_eligible(summary: dict) -> tuple[bool, list[str]]:
    comparison = summary["bootstrap"]["comparisons"]
    failures = []
    if comparison["combined_primary_rmse"]["ci_upper"] >= 0.0:
        failures.append("paired_primary_upper_ci_not_below_zero")
    if comparison["forward_time_rmse"]["delta"] > 0.0:
        failures.append("forward_time_point_regression")
    if comparison["station_time_rmse"]["delta"] > 0.0:
        failures.append("station_time_point_regression")
    for metric, reason in (
        ("p90_station_year_rmse", "p90_station_year_guard_failed"),
        ("worst_station_rmse", "worst_station_guard_failed"),
        ("p90_month_rmse", "p90_month_guard_failed"),
    ):
        if comparison[metric]["delta"] > comparison[metric][
            "bootstrap_standard_error"
        ]:
            failures.append(reason)
    if not bool(summary.get("coverage_matches_v0", False)):
        failures.append("oof_coverage_mismatch")
    if not bool(summary.get("promotable", True)):
        failures.append("diagnostic_only_candidate")
    return not failures, failures


def choose_global_candidate(candidate_summaries: Iterable[dict]) -> dict:
    candidates = []
    for raw in candidate_summaries:
        summary = dict(raw)
        eligible, failures = global_candidate_eligible(summary)
        summary["eligible"] = eligible
        summary["failure_reasons"] = failures
        candidates.append(summary)
    if not candidates:
        raise ValueError("global decision received no candidates")
    best_failed = min(
        candidates,
        key=lambda row: (row["combined_primary_rmse"], row["candidate"]),
    )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        return {
            "global_gate_passed": False,
            "winner": "V0",
            "winner_summary": None,
            "best_failed_candidate": best_failed["candidate"],
            "best_failed_summary": best_failed,
            "candidate_summaries": candidates,
            "selection_rule": "automatic_v0_fallback",
        }
    minimum = min(eligible, key=lambda row: row["combined_primary_rmse"])
    threshold = (
        minimum["combined_primary_rmse"]
        + minimum["bootstrap"]["candidate_primary_bootstrap_standard_error"]
    )
    within_one_se = [
        row for row in eligible if row["combined_primary_rmse"] <= threshold
    ]
    path_order = {"station_time": 0, "forward_time": 1}
    winner = min(
        within_one_se,
        key=lambda row: (
            int(row["actual_count"]),
            -float(row.get("selection_stability", 0.0)),
            0 if row.get("list_form") == "selected_k" else 1,
            path_order.get(row.get("path_source"), 99),
            row["candidate"],
        ),
    )
    return {
        "global_gate_passed": True,
        "winner": winner["candidate"],
        "winner_summary": winner,
        "best_failed_candidate": None,
        "best_failed_summary": None,
        "candidate_summaries": candidates,
        "minimum_risk_candidate": minimum["candidate"],
        "one_standard_error_threshold": threshold,
        "selection_rule": "minimum_then_one_standard_error",
    }


def choose_beta(beta_comparison: dict) -> dict:
    comparison = beta_comparison["comparisons"]
    choose_recent = (
        comparison["combined_primary_rmse"]["ci_upper"] < 0.0
        and comparison["forward_time_rmse"]["delta"] <= 0.0
        and comparison["station_time_rmse"]["delta"] <= 0.0
    )
    return {
        "selected_beta": 0.2 if choose_recent else 0.0,
        "beta_0_2_selected": choose_recent,
        "comparison": beta_comparison,
        "rule": (
            "paired_upper_ci_below_zero_and_no_family_regression"
            if choose_recent
            else "default_beta_0_0"
        ),
    }


def choose_moe(
    *,
    global_gate_passed: bool,
    single_global_id: str,
    moe_summary: dict | None,
) -> dict:
    if not global_gate_passed:
        return {
            "moe_promoted": False,
            "winner": single_global_id,
            "benchmark_sota_eligible": False,
            "reason": "global_gate_failed",
        }
    if moe_summary is None:
        return {
            "moe_promoted": False,
            "winner": single_global_id,
            "benchmark_sota_eligible": True,
            "reason": "no_eligible_moe_candidate",
        }
    eligible, failures = global_candidate_eligible(moe_summary)
    if eligible:
        return {
            "moe_promoted": True,
            "winner": moe_summary["candidate"],
            "benchmark_sota_eligible": True,
            "reason": "moe_passed_paired_and_robustness_gates",
            "comparison": moe_summary,
        }
    return {
        "moe_promoted": False,
        "winner": single_global_id,
        "benchmark_sota_eligible": True,
        "reason": "moe_development_gate_failed",
        "failure_reasons": failures,
        "comparison": moe_summary,
    }


def benchmark_sota_verdict(
    *,
    global_gate_passed: bool,
    challenger_development_eligible: bool,
    r2: float,
    rmse: float,
    historical_r2: float,
    historical_rmse: float,
    r2_margin: float,
    paired_primary_ci_upper: float,
    worst_station_delta: float,
    worst_station_standard_error: float,
    p90_month_delta: float,
    p90_month_standard_error: float,
    alignment_verified: bool,
) -> dict:
    checks = {
        "global_development_gate_passed": bool(global_gate_passed),
        "challenger_development_gate_passed": bool(challenger_development_eligible),
        "r2_margin_passed": float(r2) >= float(historical_r2) + float(r2_margin),
        "rmse_passed": float(rmse) < float(historical_rmse),
        "paired_primary_passed": float(paired_primary_ci_upper) < 0.0,
        "worst_station_guard_passed": (
            float(worst_station_delta) <= float(worst_station_standard_error)
        ),
        "p90_month_guard_passed": (
            float(p90_month_delta) <= float(p90_month_standard_error)
        ),
        "historical_alignment_verified": bool(alignment_verified),
    }
    return {
        "benchmark_sota_eligible": all(checks.values()),
        "checks": checks,
        "r2_threshold": float(historical_r2) + float(r2_margin),
        "claim_scope": "project_derived_8.3_2023_2025_benchmark",
        "retrospective_test": True,
        "benchmark_reused": True,
        "unbiased_sota_eligible": False,
        "unbiased_generalization_claim_eligible": False,
        "ece_external_confirmation_pending": True,
    }

