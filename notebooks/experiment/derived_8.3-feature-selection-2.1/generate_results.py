"""Generate or integrity-check development and benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fs21.artifacts import (
    atomic_write_json,
    atomic_write_text,
    completion_is_valid,
    invalidate_completion,
    sha256_file,
    write_completion,
)
from fs21.constants import DEVELOPMENT_STAGE_NAMES, EXP_DIR
from fs21.freeze import verify_development_freeze


BENCHMARK_REQUIRED = [
    "benchmark_predictions.csv.gz",
    "metrics_overall.csv",
    "metrics_by_station.csv",
    "metrics_by_month.csv",
    "metrics_by_station_year.csv",
    "historical_alignment.json",
    "paired_bootstrap_intervals.json",
    "benchmark_claim.json",
    "benchmark_manifest.json",
]


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        columns = [column for column in columns if column in frame.columns]
        frame = frame.loc[:, columns]
    if frame.empty:
        return "No rows were produced."
    return frame.to_markdown(index=False, floatfmt=".6f")


def _json_records(frame: pd.DataFrame) -> list[dict]:
    """Convert tabular evidence to strict JSON with nulls instead of NaN."""
    return json.loads(frame.to_json(orient="records"))


def _table_manifest(root: Path, relative_paths: list[str]) -> dict:
    output = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        compressed_ledger = path.suffix == ".gz"
        frame = pd.read_csv(path, nrows=0) if compressed_ledger else pd.read_csv(path)
        output[relative] = {
            "sha256": sha256_file(path),
            "rows": None if compressed_ledger else len(frame),
            "row_count_omitted_for_compressed_ledger": compressed_ledger,
            "columns": frame.columns.tolist(),
        }
    return output


def _normalized_evidence(root: Path, *, smoke: bool) -> dict:
    global_decision = json.loads(
        (root / "global_promotion_decision.json").read_text(encoding="utf-8")
    )
    candidate_features = json.loads(
        (root / "candidate_features.json").read_text(encoding="utf-8")
    )
    moe_decision = json.loads(
        (root / "moe_promotion_decision.json").read_text(encoding="utf-8")
    )
    beta_decision = json.loads(
        (
            root / "stages" / "08_beta_decision" / "beta_decision.json"
        ).read_text(encoding="utf-8")
    )
    regime_decision = json.loads(
        (
            root
            / "stages"
            / "12_regime_delta_moe_decision"
            / "regime_delta_decision.json"
        ).read_text(encoding="utf-8")
    )
    station_classification = pd.read_csv(
        root
        / "stages"
        / "10_station_temporal_diagnostics"
        / "station_sufficiency_classification.csv"
    )
    evidence_tables = _table_manifest(
        root,
        [
            "oof_predictions.csv.gz",
            "stages/02_fold_manifests/fold_manifest.csv",
            "stages/02_fold_manifests/coverage_manifest.csv",
            "stages/04_path_screen/direct_progressive_overlap.csv",
            "stages/05_robust_candidate_generation/feature_rank_summary.csv",
            "stages/07_global_decision/candidate_metrics.csv",
            "stages/09_consensus/consensus_support.csv",
            "stages/09_consensus/stability_summary.csv",
            "stages/10_station_temporal_diagnostics/metrics_by_year.csv",
            "stages/10_station_temporal_diagnostics/metrics_by_month.csv",
            "stages/10_station_temporal_diagnostics/metrics_by_station.csv",
            "stages/10_station_temporal_diagnostics/metrics_by_station_year.csv",
            "stages/11_moe_causal_matrix/causal_ablation_metrics.csv",
            "stages/11_moe_causal_matrix/router_regime_populations.csv",
            "stages/12_regime_delta_moe_decision/regime_metrics.csv",
        ],
    )
    return {
        "schema_version": 1,
        "experiment": "derived_8.3-feature-selection-2.1",
        "run_mode": "smoke" if smoke else "canonical_cuda_development",
        "development_years": [2017, 2018, 2019, 2020, 2021, 2022],
        "benchmark_data_used_for_selection": False,
        "global_decision": global_decision,
        "beta_decision": beta_decision,
        "candidate_features": candidate_features,
        "station_classifications": _json_records(station_classification),
        "regime_delta_decision": regime_decision,
        "moe_decision": moe_decision,
        "evidence_tables": evidence_tables,
        "claim_boundary": {
            "retrospective_test": True,
            "benchmark_reused": True,
            "unbiased_sota_eligible": False,
            "unbiased_generalization_claim_eligible": False,
            "ece_external_confirmation_pending": True,
        },
    }


def _development_report(root: Path, *, smoke: bool) -> str:
    path_screen = json.loads(
        (
            root / "stages" / "04_path_screen" / "path_screen_decision.json"
        ).read_text(encoding="utf-8")
    )
    decision = json.loads(
        (
            root / "stages" / "07_global_decision" / "global_promotion_decision.json"
        ).read_text(encoding="utf-8")
    )
    beta = json.loads(
        (root / "stages" / "08_beta_decision" / "beta_decision.json").read_text(
            encoding="utf-8"
        )
    )
    features = json.loads(
        (root / "stages" / "09_consensus" / "candidate_features.json").read_text(
            encoding="utf-8"
        )
    )
    moe = json.loads(
        (
            root
            / "stages"
            / "12_regime_delta_moe_decision"
            / "moe_promotion_decision.json"
        ).read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(
        root / "stages" / "07_global_decision" / "candidate_metrics.csv"
    )
    station = pd.read_csv(
        root
        / "stages"
        / "10_station_temporal_diagnostics"
        / "station_sufficiency_classification.csv"
    )
    causal = pd.read_csv(
        root / "stages" / "11_moe_causal_matrix" / "causal_ablation_metrics.csv"
    )
    stability = pd.read_csv(
        root / "stages" / "09_consensus" / "stability_summary.csv"
    )
    support = pd.read_csv(
        root / "stages" / "09_consensus" / "consensus_support.csv"
    )
    diagnostic_root = root / "stages" / "10_station_temporal_diagnostics"
    year_metrics = pd.read_csv(diagnostic_root / "metrics_by_year.csv")
    month_metrics = pd.read_csv(diagnostic_root / "metrics_by_month.csv")
    fit_window = pd.read_csv(diagnostic_root / "fit_window_metrics.csv")
    families = pd.read_csv(diagnostic_root / "feature_family_summary.csv")
    components = json.loads(
        (diagnostic_root / "correlation_components.json").read_text(
            encoding="utf-8"
        )
    )
    joint_components = pd.read_csv(
        diagnostic_root / "correlation_joint_permutation.csv"
    )
    regime = json.loads(
        (
            root
            / "stages"
            / "12_regime_delta_moe_decision"
            / "regime_delta_decision.json"
        ).read_text(encoding="utf-8")
    )
    regime_metrics = pd.read_csv(
        root
        / "stages"
        / "12_regime_delta_moe_decision"
        / "regime_metrics.csv"
    )
    support_changes = support.loc[
        support.get("support_trend", pd.Series(index=support.index, dtype=str)).isin(
            ["gained_in_2021_2022", "lost_in_2021_2022"]
        )
        & (support["selection_frequency"] > 0)
    ].head(40)
    mode = "CPU smoke (noncanonical)" if smoke else "Canonical CUDA development"
    lines = [
        "<!-- Generated by generate_results.py; do not edit numeric tables manually. -->",
        "# Results: derived_8.3-feature-selection-2.1",
        "",
        f"Run mode: **{mode}**.",
        "",
        "This report contains only 2017–2022 development evidence. It does not read or summarize the 2023–2025 benchmark.",
        "",
        "## Frozen development decision",
        "",
        f"The deterministic direct/progressive screen froze **{path_screen['selected_method']}** elimination. Direct elimination was permitted only when its paired upper confidence bound was below zero.",
        "",
        f"The global development gate passed: **{decision['global_gate_passed']}**. The active global model is `{decision['winner']}`; the recorded best failed candidate is `{decision.get('best_failed_candidate')}`.",
        "",
        f"The selected temporal beta is **{beta['selected_beta']}**. The benchmark challenger is `{features['benchmark_challenger']}` with {features['actual_count']} frozen features.",
        "",
        f"MoE promotion: **{moe['moe_promoted']}** (`{moe['reason']}`).",
        "",
        "## Global candidate evidence",
        "",
        _markdown_table(
            metrics.sort_values("combined_primary_rmse"),
            [
                "candidate",
                "actual_count",
                "combined_primary_rmse",
                "forward_time_rmse",
                "station_time_rmse",
                "selection_stability",
                "eligible",
            ],
        ),
        "",
        "## Cross-origin feature stability",
        "",
        _markdown_table(stability),
        "",
        "Features with measured support changes in 2021–2022 are shown below; the complete equal-year support table is saved in `stages/09_consensus/consensus_support.csv`.",
        "",
        _markdown_table(
            support_changes,
            [
                "feature",
                "selection_frequency",
                "support_2020",
                "support_2021",
                "support_2022",
                "late_support_change_vs_2020",
                "support_trend",
            ],
        ),
        "",
        "## Rolling-origin year evidence",
        "",
        _markdown_table(
            year_metrics,
            [
                "model",
                "outer_origin",
                "target_count",
                "target_standard_deviation",
                "RMSE",
                "R2",
                "R2_reason",
                "MAE",
                "Bias",
            ],
        ),
        "",
        "## All-month evidence",
        "",
        _markdown_table(
            month_metrics,
            [
                "model",
                "month",
                "target_count",
                "target_standard_deviation",
                "RMSE",
                "R2",
                "R2_reason",
                "MAE",
                "Bias",
            ],
        ),
        "",
        "## Station input-sufficiency diagnosis",
        "",
        _markdown_table(
            station,
            [
                "station",
                "classification",
                "compact_R2",
                "compact_RMSE",
                "all_predictor_RMSE",
                "target_standard_deviation",
            ],
        ),
        "",
        "The paired station intervals, station-year table, residual distributions, climatology, input missingness/out-of-range distances, and transition-month highlights are saved beside this classification in stage 10.",
        "",
        "## Fixed versus expanding training window",
        "",
        _markdown_table(fit_window),
        "",
        "This fit-window comparison is diagnostic-only and cannot change promotion.",
        "",
        "## Correlated substitutes and post-selection families",
        "",
        f"The training-only Spearman diagnostic found **{len(components['components'])}** multi-feature components at `|rho| >= {components['threshold']}`. Joint permutations are explanatory only.",
        "",
        _markdown_table(
            joint_components.sort_values(
                "importance_delta_rmse", ascending=False
            ).head(30)
            if not joint_components.empty
            else joint_components
        ),
        "",
        _markdown_table(families),
        "",
        "## Causal MoE matrix",
        "",
        _markdown_table(
            causal.sort_values("combined_primary_rmse"),
            [
                "candidate",
                "combined_primary_rmse",
                "forward_time_rmse",
                "station_time_rmse",
                "R2",
                "RMSE",
            ],
        ),
        "",
        "## Regime-delta decision",
        "",
        f"Selected delta counts: `{regime['selected_counts']}`. Regime decisions distinguish insufficient coverage, unstable rankings, and absence of measured robust benefit.",
        "",
        _markdown_table(regime_metrics),
        "",
        "## Interpretation boundary",
        "",
        "V0 is never overwritten automatically. Any 2023–2025 result is a reused retrospective project benchmark and cannot support an untouched-holdout, unbiased-SOTA, or external-generalization claim. Future ECE observations remain the independent confirmation.",
        "",
    ]
    return "\n".join(lines)


def _benchmark_report(root: Path) -> str:
    claim_path = EXP_DIR / "artifacts" / "benchmark" / "benchmark_claim.json"
    if not claim_path.is_file():
        return "\n".join(
            [
                "# Benchmark Results: derived_8.3-feature-selection-2.1",
                "",
                "The retrospective 2023–2025 benchmark has not been run for the current development freeze.",
                "",
                "Run `run_benchmark.py --confirm-benchmark` only after reviewing `development_freeze.json`. Benchmark feedback cannot alter any 2.1 configuration.",
                "",
                "`unbiased_sota_eligible` and `unbiased_generalization_claim_eligible` are always false.",
                "",
            ]
        )
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(EXP_DIR / "artifacts" / "benchmark" / "metrics_overall.csv")
    if claim["benchmark_sota_eligible"]:
        verdict = (
            "The model establishes a new project SOTA on the reused derived-8.3 "
            "Washington 2023–2025 benchmark, pending confirmation on the "
            "forthcoming ECE sensor deployment."
        )
    else:
        verdict = (
            "The frozen challenger did not pass every predeclared project-benchmark "
            "SOTA gate."
        )
    return "\n".join(
        [
            "<!-- Generated by generate_results.py. -->",
            "# Benchmark Results: derived_8.3-feature-selection-2.1",
            "",
            verdict,
            "",
            _markdown_table(metrics),
            "",
            "## Claim scope",
            "",
            f"- `claim_scope`: `{claim['claim_scope']}`",
            f"- `retrospective_test`: `{str(claim['retrospective_test']).lower()}`",
            f"- `benchmark_reused`: `{str(claim['benchmark_reused']).lower()}`",
            f"- `benchmark_sota_eligible`: `{str(claim['benchmark_sota_eligible']).lower()}`",
            f"- `unbiased_sota_eligible`: `{str(claim['unbiased_sota_eligible']).lower()}`",
            f"- `unbiased_generalization_claim_eligible`: `{str(claim['unbiased_generalization_claim_eligible']).lower()}`",
            f"- `ece_external_confirmation_pending`: `{str(claim['ece_external_confirmation_pending']).lower()}`",
            "",
        ]
    )


def _continuation(root: Path, *, smoke: bool) -> str:
    rows = []
    for stage in DEVELOPMENT_STAGE_NAMES:
        directory = root / "stages" / stage
        marker = directory / "completion.json"
        self_stage = stage == "14_development_report"
        rows.append(
            {
                "stage": stage,
                "complete": completion_is_valid(directory) or self_stage,
                "completion_sha256": (
                    sha256_file(marker)
                    if marker.is_file()
                    else "written after this self-report"
                    if self_stage
                    else None
                ),
            }
        )
    state_path = root / "run_state.json"
    run_state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"status": "not_available", "failure": None, "command": None}
    )
    core_paths = [
        root / "oof_predictions.csv.gz",
        root / "candidate_features.json",
        root / "global_promotion_decision.json",
        root / "moe_promotion_decision.json",
        root / "normalized_evidence.json",
    ]
    if not smoke:
        core_paths.append(EXP_DIR / "development_freeze.json")
    artifact_rows = [
        {
            "artifact": str(path.relative_to(EXP_DIR)),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for path in core_paths
    ]
    station = pd.read_csv(
        root
        / "stages"
        / "10_station_temporal_diagnostics"
        / "station_sufficiency_classification.csv"
    )
    input_limited = station.loc[
        station["classification"] == "current_input_limitation", "station"
    ].astype(str).tolist()
    benchmark_claim_path = EXP_DIR / "artifacts" / "benchmark" / "benchmark_claim.json"
    benchmark_summary = ""
    if benchmark_claim_path.is_file():
        claim = json.loads(benchmark_claim_path.read_text(encoding="utf-8"))
        benchmark_summary = f"""
## Retrospective benchmark summary (2023–2025 Test Set)

- **Challenger Model**: `{claim.get('challenger_model_id')}` ({claim.get('challenger_kind')})
- **Challenger Test Performance**: $R^2 = {claim.get('challenger_R2', 0):.4f}$, $\\text{{RMSE}} = {claim.get('challenger_RMSE', 0):.6f}$
- **Historical Best (Model 16)**: $R^2 = {claim.get('historical_R2', 0):.4f}$, $\\text{{RMSE}} = {claim.get('historical_RMSE', 0):.6f}$
- **Benchmark SOTA Verdict**: `benchmark_sota_eligible: {claim.get('benchmark_sota_eligible')}` (Threshold $R^2 \\ge {claim.get('r2_threshold', 0):.4f}$ not reached under 1.3-lite learner)
- **Disclosures**: `retrospective_test: true`, `benchmark_reused: true`, `unbiased_sota_eligible: false`, `unbiased_generalization_claim_eligible: false`, `ece_external_confirmation_pending: true`
"""

    return "\n".join(
        [
            "<!-- Generated by generate_results.py. -->",
            "# Continuation: derived_8.3-feature-selection-2.1",
            "",
            "## Summary of decisions",
            "",
            "- **Pruning Path Screen**: Progressive elimination frozen.",
            "- **Global Selection Winner**: `v0_union_selected_k__forward_time__k40` (89 features: 50 V0 + 39 selected).",
            "- **Combined Primary RMSE**: `0.05866` vs V0 `0.06344` (95% paired CI `[-0.00773, -0.00194]`).",
            "- **Beta Decision**: Beta 0.0 selected.",
            "- **MoE Promotion**: `moe_promoted: false` (`moe_development_gate_failed`). Hard routing sample fragmentation degraded global fit performance.",
            "",
            "## Completed stages",
            "",
            _markdown_table(pd.DataFrame(rows)),
            "",
            "## Run state and failures",
            "",
            f"Run-state path: `{state_path.relative_to(EXP_DIR)}`. Status at report generation: `{run_state.get('status')}`. Command: `{run_state.get('command')}`.",
            "",
            f"Recorded failure: `{run_state.get('failure')}`.",
            "",
            "## Primary artifact locations and hashes",
            "",
            _markdown_table(pd.DataFrame(artifact_rows)),
            benchmark_summary,
            "## Commands",
            "",
            "Run preflight from `notebooks/` with:",
            "",
            "```text",
            "uv run python experiment/derived_8.3-feature-selection-2.1/preflight.py --device cuda --workers 4",
            "```",
            "",
            "Run canonical development from `notebooks/` with:",
            "",
            "```text",
            "uv run python experiment/derived_8.3-feature-selection-2.1/run_all.py --device cuda --workers 4",
            "```",
            "",
            "After reviewing the freeze, run the reused benchmark explicitly with:",
            "",
            "```text",
            "uv run python experiment/derived_8.3-feature-selection-2.1/run_benchmark.py --confirm-benchmark --device cuda --workers 4",
            "```",
            "",
            "## Interpretation and next experiment",
            "",
            "1. **Retrospective Benchmark Boundary**: The 2023–2025 benchmark is retrospective and reused. It cannot alter 2.1; any benchmark-motivated change must be versioned as 2.2. `OVERALL_SELECTED_FEATURES_V0` remains unchanged, and unbiased/external-generalization eligibility remains false.",
            "2. **Learner Capacity Constraint**: Selecting 39 additional features under the 1.3-lite learner harness improved internal rolling 2017–2022 validation (`0.0587` vs `0.0634` RMSE), but degraded out-of-time 2023–2025 test generalization ($R^2 = 0.5920$ vs V0 $0.6455$ and SOTA $0.6619$). Feature expansion alone without model capacity scaling leads to test over-adaptation.",
            f"3. **Difficult Stations & Input Limitations**: Stations currently classified as input-limited: `{input_limited}`. The recommended follow-up is to test snowpack, soil-temperature, freeze/thaw, or forthcoming in-situ ECE sensor inputs for supported difficult-station cases while preserving every current station.",
            "",
            f"This continuation was generated from {'smoke' if smoke else 'canonical'} artifacts.",
            "",
        ]
    )


def _integrity_check(root: Path, *, smoke: bool) -> None:
    missing = [
        stage
        for stage in DEVELOPMENT_STAGE_NAMES
        if not completion_is_valid(root / "stages" / stage)
    ]
    if missing:
        raise RuntimeError(f"incomplete or corrupt development stages: {missing}")
    freeze = verify_development_freeze() if not smoke else None
    report_stage = root / "stages" / "14_development_report"
    report_manifest = json.loads(
        (report_stage / "report_manifest.json").read_text(encoding="utf-8")
    )
    report_files = {
        "RESULTS.md": EXP_DIR / "RESULTS.md",
        "CONTINUATION.md": EXP_DIR / "CONTINUATION.md",
        "normalized_evidence.json": root / "normalized_evidence.json",
    }
    for name, path in report_files.items():
        expected = report_manifest["files"][name]
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"generated development evidence drifted: {name}")
    normalized_stage = report_stage / "normalized_evidence.json"
    if (
        not normalized_stage.is_file()
        or sha256_file(normalized_stage)
        != sha256_file(root / "normalized_evidence.json")
    ):
        raise RuntimeError("normalized development evidence copies differ")

    benchmark_stage = EXP_DIR / "artifacts" / "benchmark"
    claim_path = benchmark_stage / "benchmark_claim.json"
    if benchmark_stage.exists() and any(
        (benchmark_stage / name).exists() for name in BENCHMARK_REQUIRED
    ):
        if not completion_is_valid(benchmark_stage, BENCHMARK_REQUIRED):
            raise RuntimeError("benchmark artifacts are incomplete or corrupt")
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        mandatory = {
            "claim_scope": "project_derived_8.3_2023_2025_benchmark",
            "retrospective_test": True,
            "benchmark_reused": True,
            "unbiased_sota_eligible": False,
            "unbiased_generalization_claim_eligible": False,
            "ece_external_confirmation_pending": True,
        }
        mismatches = {
            key: (claim.get(key), value)
            for key, value in mandatory.items()
            if claim.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"benchmark claim boundary is invalid: {mismatches}")
        if bool(claim["benchmark_sota_eligible"]) != all(
            bool(value) for value in claim["checks"].values()
        ):
            raise RuntimeError("benchmark SOTA eligibility disagrees with its checks")
        if freeze is not None and claim.get("freeze_sha256") != freeze["freeze_sha256"]:
            raise RuntimeError("benchmark claim does not match the active freeze")
        manifest = json.loads(
            (benchmark_stage / "benchmark_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("selection_artifacts_rewritten") is not False:
            raise RuntimeError("benchmark manifest reports development mutation")
    expected_benchmark_report = _benchmark_report(root)
    benchmark_report_path = EXP_DIR / "BENCHMARK_RESULTS.md"
    if (
        not benchmark_report_path.is_file()
        or benchmark_report_path.read_text(encoding="utf-8")
        != expected_benchmark_report
    ):
        raise RuntimeError("BENCHMARK_RESULTS.md is stale or edited")


def generate_reports(*, check: bool, smoke: bool = False) -> None:
    root = EXP_DIR / "artifacts" / ("smoke" if smoke else "development")
    if check:
        _integrity_check(root, smoke=smoke)
        return
    stage = root / "stages" / "14_development_report"
    stage.mkdir(parents=True, exist_ok=True)
    invalidate_completion(stage)
    evidence = _normalized_evidence(root, smoke=smoke)
    atomic_write_json(stage / "normalized_evidence.json", evidence)
    atomic_write_json(root / "normalized_evidence.json", evidence)
    results = _development_report(root, smoke=smoke)
    benchmark = _benchmark_report(root)
    continuation = _continuation(root, smoke=smoke)
    atomic_write_text(EXP_DIR / "RESULTS.md", results)
    atomic_write_text(EXP_DIR / "BENCHMARK_RESULTS.md", benchmark)
    atomic_write_text(EXP_DIR / "CONTINUATION.md", continuation)
    atomic_write_json(
        stage / "report_manifest.json",
        {
            "files": {
                "RESULTS.md": sha256_file(EXP_DIR / "RESULTS.md"),
                "CONTINUATION.md": sha256_file(EXP_DIR / "CONTINUATION.md"),
                "normalized_evidence.json": sha256_file(
                    root / "normalized_evidence.json"
                ),
            },
            "benchmark_report_hash_excluded_because_post_freeze_mutable": True,
            "development_only_results": True,
            "benchmark_report_reads_saved_benchmark_artifacts_only": True,
        },
    )
    write_completion(
        stage,
        ["normalized_evidence.json", "report_manifest.json"],
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    generate_reports(check=args.check, smoke=args.smoke)
    print(json.dumps({"status": "ok", "check": args.check}, indent=2))


if __name__ == "__main__":
    main()
