"""Build the reproducible 1.1 report notebook through the trusted ``nb`` CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
NOTEBOOK = EXP_DIR / "derived_8.4-ece-model-salvage-1.1.ipynb"


def markdown(source: str) -> tuple[str, str]:
    return "@@markdown", source.strip()


def code(source: str) -> tuple[str, str]:
    return "@@code", source.strip()


def add_batch(cells: list[tuple[str, str]]) -> None:
    payload_parts: list[str] = []
    for kind, source in cells:
        payload_parts.extend([kind, source, ""])
    subprocess.run(
        ["nb", "cell", "add", str(NOTEBOOK), "--source", "-"],
        input="\n".join(payload_parts),
        text=True,
        check=True,
    )


def main() -> None:
    subprocess.run(
        ["nb", "create", str(NOTEBOOK), "--kernel", "python3", "--uv", "--force"],
        check=True,
    )
    title = """# `derived_8.4-ece-model-salvage-1.1`

This report evaluates six retrained routing/model families at 60 selected features and the global model at 40, 50, 60, and 69 selected features after removing every SMAP-derived input. All selector, router, and model fitting is restricted to the seven Washington training stations; ECE targets are held out for spatial evaluation only."""
    subprocess.run(
        ["nb", "cell", "update", str(NOTEBOOK), "--cell-index", "0", "--type", "markdown", "--source", "-"],
        input=title,
        text=True,
        check=True,
    )

    add_batch([
        markdown("""## Reproducible setup

The next cell loads the tracked runner and generated artifacts so the notebook remains a reporting layer rather than a second implementation of the experiment."""),
        code("""from pathlib import Path
import importlib.util
import json
import sys
import pandas as pd

EXP_DIR = Path("experiment/derived_8.4-ece-model-salvage-1.1").resolve()
if not (EXP_DIR / "run_model_salvage.py").exists():
    EXP_DIR = Path.cwd().resolve()
spec = importlib.util.spec_from_file_location("model_salvage_11", EXP_DIR / "run_model_salvage.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
config = runner.load_configuration()
predictions = pd.read_csv(EXP_DIR / "predictions.csv", low_memory=False)
seed_metrics = pd.read_csv(EXP_DIR / "seed_metrics.csv", low_memory=False)
summary = pd.read_csv(EXP_DIR / "summary.csv", low_memory=False)
audit = pd.read_csv(EXP_DIR / "routing_audit.csv", low_memory=False)
print(f"Experiment: {config['experiment']['name']}")
print(f"Models: {len(config['models'])}; feature sizes: {config['selected_feature_sizes']}; seeds: {config['seeds']}")
print(f"Prediction rows: {len(predictions):,}")"""),
        markdown("""## Feature-selection provenance

The local selector is a copied and adapted MI → ElasticNet → stability → wrapper pipeline. Its accepted wrapper objective uses the WA 2023–2025 evaluation period as a documented manual-selection analogue. One evidence-ranked wrapper result is normalized into nested 40/50/60/69 feature manifests, with no specialist or delta additions."""),
        code("""print("REPORT_BEGIN::SELECTION")
selection_path = EXP_DIR / "feature_selection_artifacts" / "selected_features_by_size.json"
selection = json.loads(selection_path.read_text(encoding="utf-8"))
selected_by_size = selection["feature_sizes"]
candidate_pool = pd.read_csv(EXP_DIR / "feature_selection_artifacts" / "candidate_pool.csv")
selection_table = pd.DataFrame([
    {
        "status": selection.get("status"),
        "selected_features": int(size),
        "smap_features": sum("smap" in feature.lower() for feature in record["features"]),
        "candidate_pool": len(candidate_pool),
        "delta_additions": "none",
        "selection_period": "WA 2023–2025",
        "fit_scope": "WA only",
    }
    for size, record in sorted(selected_by_size.items(), key=lambda item: int(item[0]))
])
print(selection_table.to_markdown(index=False))
for size, record in sorted(selected_by_size.items(), key=lambda item: int(item[0])):
    print()
    print(f"### Selected feature manifest ({size} features)")
    print()
    print("```text")
    print("\\n".join(record["features"]))
    print("```")
print("REPORT_END::SELECTION")"""),
    ])

    add_batch([
        markdown("""## Input and feature audit

This audit records the exact training/evaluation populations, the nested selected feature manifests, and the lineage-specific router inputs after SMAP filtering."""),
        code("""print("REPORT_BEGIN::INPUT_AUDIT")
with (EXP_DIR / "input_audit.json").open(encoding="utf-8") as handle:
    print(json.dumps(json.load(handle), indent=2, sort_keys=True))
print("REPORT_END::INPUT_AUDIT")

print("REPORT_BEGIN::FEATURE_AUDIT")
with (EXP_DIR / "feature_manifest.json").open(encoding="utf-8") as handle:
    feature_manifest = json.load(handle)
feature_rows = []
for name, value in feature_manifest.items():
    if name == "selected_model_feature_sets":
        for size, selected_value in sorted(value.items(), key=lambda item: int(item[0])):
            feature_rows.append({
                "component": f"selected_model_{size}",
                "parent_count": "selector",
                "dropped_smap": 0,
                "effective_count": len(selected_value["effective"]),
                "effective_features": ";".join(selected_value["effective"]),
            })
    elif isinstance(value, dict) and "effective" in value:
        feature_rows.append({
            "component": name,
            "parent_count": len(value["parent"]) if isinstance(value["parent"], list) else value["parent"],
            "dropped_smap": len(value["dropped"]) if isinstance(value["dropped"], list) else 0,
            "effective_count": len(value["effective"]),
            "effective_features": ";".join(value["effective"]),
        })
feature_table = pd.DataFrame(feature_rows)
print(feature_table.to_markdown(index=False))
print("REPORT_END::FEATURE_AUDIT")"""),
        markdown("""## Router audit

Regime shares are shown for WA trainval, the held-out WA temporal test, and the unseen ECE sensor set. Router seed remains fixed at 42."""),
        code("""print("REPORT_BEGIN::ROUTER_AUDIT")
print(audit.to_markdown(index=False, floatfmt=".4f"))
print("REPORT_END::ROUTER_AUDIT")"""),
        markdown("""## Ten-variant metrics

RMSE is the primary error metric; Pearson and first-difference Pearson correlation assess whether predictions follow the target level and temporal trend."""),
        code("""pooled = summary[summary["scope"] == "__pooled__"].copy()
columns = ["model_id", "dataset", "window", "n_seeds", "rmse_mean", "rmse_std", "mae_mean", "bias_mean", "ubrmse_mean", "r2_mean", "pearson_mean", "target_std_mean", "prediction_std_mean", "diff_pearson_mean"]
print("REPORT_BEGIN::METRICS")
print(pooled[columns].sort_values(["dataset", "window", "rmse_mean"]).to_markdown(index=False, floatfmt=".6f"))
print("REPORT_END::METRICS")"""),
    ])

    add_batch([
        markdown("""## Comparison with original SMAP-trained models

The original formal-evaluation results remain reference-only and are never used for 1.1 fitting. To keep this section compact, each model/split row reports the mean over the common three learner seeds; the seed-level paired artifact remains available for audit."""),
        code("""comparison = pd.read_csv(EXP_DIR / "reference_comparison.csv", low_memory=False)
print("REPORT_BEGIN::REFERENCE_COMPARISON")
if comparison.empty:
    print("No reference rows available.")
else:
    reference_mean = runner.summarize_reference_comparison(
        comparison, expected_seeds=config["seeds"]
    )
    columns = ["model_id", "dataset", "window", "n_seeds", "rmse_no_smap", "rmse_original", "rmse_delta_no_smap_minus_original", "pearson_no_smap", "pearson_original"]
    print(reference_mean[columns].to_markdown(index=False, floatfmt=".6f"))
print("REPORT_END::REFERENCE_COMPARISON")"""),
        markdown("""## Comparison of 1.1 selected features with 1.0 no-SMAP models

This same-seed comparison isolates the feature-selection change from the original SMAP-removal change. It is populated when the completed 1.0 prediction artifacts are available."""),
        code("""comparison_10 = pd.read_csv(EXP_DIR / "salvage_1_1_vs_1_0_summary.csv", low_memory=False)
print("REPORT_BEGIN::SALVAGE_1_1_VS_1_0")
if comparison_10.empty:
    print("No completed 1.0 paired prediction artifacts available yet.")
else:
    columns = ["model_id", "old_model_id", "dataset", "window", "n_seeds", "change_rmse_mean", "change_mae_mean", "change_bias_mean", "change_pearson_mean", "change_diff_pearson_mean"]
    print(comparison_10[columns].sort_values(["dataset", "window", "change_rmse_mean"]).to_markdown(index=False, floatfmt=".6f"))
print("REPORT_END::SALVAGE_1_1_VS_1_0")"""),
        markdown("""## Before vs after the new feature-selection round

This global-only comparison treats the 1.0 no-SMAP global model as the before-selection baseline and the four nested 1.1 global feature sizes as after-selection variants. It reports pooled ECE benefit, pooled WA degradation, and changes in level and first-difference trend correlation over the common seeds."""),
        code("""feature_selection_effect = pd.read_csv(EXP_DIR / "feature_selection_round_effect_summary.csv", low_memory=False)
print("REPORT_BEGIN::FEATURE_SELECTION_ROUND")
if feature_selection_effect.empty:
    print("Feature-selection comparison is not available yet.")
else:
    table_rows = []
    for feature_size in sorted(feature_selection_effect["feature_size"].unique()):
        ece_row = feature_selection_effect[
            (feature_selection_effect["feature_size"] == feature_size)
            & (feature_selection_effect["split"] == "ECE spatial")
        ]
        wa_row = feature_selection_effect[
            (feature_selection_effect["feature_size"] == feature_size)
            & (feature_selection_effect["split"] == "WA temporal")
        ]
        if len(ece_row) != 1 or len(wa_row) != 1:
            raise ValueError(f"Expected one ECE and one WA row for {feature_size} features.")
        ece_row = ece_row.iloc[0]
        wa_row = wa_row.iloc[0]
        table_rows.append({
            "features": int(feature_size),
            "n_seeds": int(ece_row["n_seeds"]),
            "ECE before RMSE": ece_row["rmse_before_mean"],
            "ECE after RMSE": ece_row["rmse_after_mean"],
            "ECE benefit": ece_row["rmse_effect_mean"],
            "ECE benefit %": ece_row["rmse_effect_pct_mean"],
            "ECE ΔPearson": ece_row["pearson_change_mean"],
            "ECE Δdiff Pearson": ece_row["diff_pearson_change_mean"],
            "WA before RMSE": wa_row["rmse_before_mean"],
            "WA after RMSE": wa_row["rmse_after_mean"],
            "WA degradation": wa_row["rmse_effect_mean"],
            "WA degradation %": wa_row["rmse_effect_pct_mean"],
            "WA ΔPearson": wa_row["pearson_change_mean"],
            "WA Δdiff Pearson": wa_row["diff_pearson_change_mean"],
        })
    print(pd.DataFrame(table_rows).to_markdown(index=False, floatfmt=".6f"))
    print("INTERPRETATION_BEGIN")
    print(runner.interpret_feature_selection_round(feature_selection_effect, expected_seeds=(42, 7, 13)))
    print("INTERPRETATION_END")
print("REPORT_END::FEATURE_SELECTION_ROUND")"""),
        markdown("""## Effect of Removing SMAP: ECE Benefit vs WA Degradation

This paired summary compares 1.1 against the original SMAP-trained references. ECE benefit is original RMSE minus no-SMAP RMSE; WA degradation is no-SMAP RMSE minus original RMSE. Global feature-count variants are listed separately. The effect chart uses a shared fixed y-axis of −0.04 to 0.04 RMSE."""),
        code("""effect_summary = pd.read_csv(EXP_DIR / "old_vs_new_effect_summary.csv", low_memory=False)
effect_columns = ["model_id", "split", "n_seeds", "rmse_original_mean", "rmse_no_smap_mean", "effect_rmse_mean", "effect_rmse_std", "effect_rmse_pct_mean", "improved_seeds", "worsened_seeds", "pearson_change_mean", "diff_pearson_change_mean"]
print("REPORT_BEGIN::OLD_NEW_EFFECT")
if effect_summary.empty:
    print("No paired old-vs-new rows available.")
else:
    print(effect_summary[effect_columns].sort_values(["split", "effect_rmse_mean"], ascending=[True, False]).to_markdown(index=False, floatfmt=".6f"))
effect_figure = runner.make_old_vs_new_effect_figure(effect_summary, EXP_DIR / "figures")
print(f"FIGURE::{effect_figure.name}")
print("REPORT_END::OLD_NEW_EFFECT")"""),
    ])

    add_batch([
        markdown("""## SMAP-invariance check

Seed-42 ECE rows are evaluated once with native SMAP values and once with all SMAP columns replaced by zero; a no-SMAP model must produce identical regimes and predictions."""),
        code("""invariance = pd.read_csv(EXP_DIR / "smap_invariance.csv", low_memory=False)
print("REPORT_BEGIN::SMAP_INVARIANCE")
print(invariance.to_markdown(index=False, floatfmt=".12f"))
print("REPORT_END::SMAP_INVARIANCE")"""),
        markdown("""## Global model version comparison

Each ECE station receives one seven-line chart comparing the original global model, the 1.0 no-SMAP global model, the 1.1 global models at 40/50/60/69 features, and ground truth. Predictions are aligned by station/date and averaged over the common seeds `[42, 7, 13]`. Every ECE line chart uses the same fixed y-axis of 0.00 to 0.25 soil-moisture units."""),
        code("""print("REPORT_BEGIN::GLOBAL_VERSION")
version_data = runner.load_data(config)
version_status = runner.global_version_source_status(version_data, config, (42, 7, 13))
print(json.dumps(version_status, indent=2, sort_keys=True))
if version_status["ready"]:
    version_paths = runner.make_global_version_charts(version_data, config, EXP_DIR / "figures", (42, 7, 13))
    for path in version_paths:
        print(f"FIGURE::{path.name}")
else:
    print("Global-version figures deferred until all old, 1.0, and 1.1 common-seed artifacts are complete.")
print("REPORT_END::GLOBAL_VERSION")"""),
        markdown("""## Trend figures

The next cell generates three figure suites per ECE station: architecture gates, alternative regime gates, and all four global feature sizes. Each chart is limited to five lines including the observed target and uses the common fixed y-axis of 0.00 to 0.25 soil-moisture units."""),
        code("""figure_paths = runner.make_trend_figures(predictions, EXP_DIR / "figures")
print("REPORT_BEGIN::FIGURES")
for path in figure_paths:
    print(f"- {path.name}")
print("REPORT_END::FIGURES")"""),
        markdown("""## Completion

All tables above are sourced from executed cells, and the linked figures are generated during this notebook execution."""),
        code("""print("Notebook complete — all report sections above are generated from experiment artifacts.")"""),
    ])
    print(f"Built {NOTEBOOK}")


if __name__ == "__main__":
    main()
