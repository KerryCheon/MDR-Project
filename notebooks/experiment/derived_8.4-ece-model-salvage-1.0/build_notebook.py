"""Build the report notebook using the trusted ``nb`` CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
NOTEBOOK = EXP_DIR / "derived_8.4-ece-model-salvage-1.0.ipynb"


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
    title = """# `derived_8.4-ece-model-salvage-1.0`

This report evaluates seven retrained routing/model families after removing every SMAP-derived input. All fitting is restricted to the seven Washington training stations, while ECE targets are held out for spatial evaluation only."""
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
import sys
import pandas as pd

EXP_DIR = Path("experiment/derived_8.4-ece-model-salvage-1.0").resolve()
if not (EXP_DIR / "run_model_salvage.py").exists():
    EXP_DIR = Path.cwd().resolve()
spec = importlib.util.spec_from_file_location("model_salvage", EXP_DIR / "run_model_salvage.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
config = runner.load_configuration()
predictions = pd.read_csv(EXP_DIR / "predictions.csv", low_memory=False)
seed_metrics = pd.read_csv(EXP_DIR / "seed_metrics.csv", low_memory=False)
summary = pd.read_csv(EXP_DIR / "summary.csv", low_memory=False)
audit = pd.read_csv(EXP_DIR / "routing_audit.csv", low_memory=False)
print(f"Experiment: {config['experiment']['name']}")
print(f"Models: {len(config['models'])}; seeds: {config['seeds']}")
print(f"Prediction rows: {len(predictions):,}")"""),
        markdown("""## Input and feature audit

This audit records the exact training/evaluation populations and confirms the effective feature counts after SMAP removal."""),
        code("""import json

print("REPORT_BEGIN::INPUT_AUDIT")
with (EXP_DIR / "input_audit.json").open(encoding="utf-8") as handle:
    print(json.dumps(json.load(handle), indent=2, sort_keys=True))
print("REPORT_END::INPUT_AUDIT")

print("REPORT_BEGIN::FEATURE_AUDIT")
with (EXP_DIR / "feature_manifest.json").open(encoding="utf-8") as handle:
    feature_manifest = json.load(handle)
feature_table = pd.DataFrame([
    {"component": name, "parent_count": len(value["parent"]), "dropped_smap": len(value["dropped"]), "effective_count": len(value["effective"]), "effective_features": ";".join(value["effective"])}
    for name, value in feature_manifest.items()
    if isinstance(value, dict) and "parent" in value
])
print(feature_table.to_markdown(index=False))
print("REPORT_END::FEATURE_AUDIT")"""),
    ])

    add_batch([
        markdown("""## Router audit

Regime shares are shown for WA trainval, the held-out WA temporal test, and the unseen ECE sensor set."""),
        code("""print("REPORT_BEGIN::ROUTER_AUDIT")
print(audit.to_markdown(index=False, floatfmt=".4f"))
print("REPORT_END::ROUTER_AUDIT")"""),
        markdown("""## Seven-model metrics

RMSE is the primary error metric; Pearson and first-difference Pearson correlation assess whether predictions follow the target level and temporal trend."""),
        code("""pooled = summary[summary["scope"] == "__pooled__"].copy()
columns = ["model_id", "dataset", "window", "n_seeds", "rmse_mean", "rmse_std", "mae_mean", "bias_mean", "ubrmse_mean", "r2_mean", "pearson_mean", "diff_pearson_mean"]
print("REPORT_BEGIN::METRICS")
print(pooled[columns].sort_values(["dataset", "window", "rmse_mean"]).to_markdown(index=False, floatfmt=".6f"))
print("REPORT_END::METRICS")"""),
        markdown("""## Original-model comparison

The following table compares same-seed no-SMAP results with the original SMAP-trained reference runs; reference rows are never used for fitting."""),
        code("""comparison = pd.read_csv(EXP_DIR / "reference_comparison.csv", low_memory=False)
print("REPORT_BEGIN::REFERENCE_COMPARISON")
if comparison.empty:
    print("No reference rows available.")
else:
    columns = ["model_id", "seed", "dataset", "rmse_no_smap", "rmse_original", "rmse_delta_no_smap_minus_original", "pearson_no_smap", "pearson_original"]
    print(comparison[columns].to_markdown(index=False, floatfmt=".6f"))
print("REPORT_END::REFERENCE_COMPARISON")"""),
        markdown("""## Effect of Removing SMAP: ECE Benefit vs WA Degradation

This paired five-seed summary uses the original model only as a reference. ECE benefit is defined as original RMSE minus no-SMAP RMSE, while WA degradation is defined as no-SMAP RMSE minus original RMSE; positive values therefore have the stated interpretation in each panel."""),
        code("""effect_summary = pd.read_csv(EXP_DIR / "old_vs_new_effect_summary.csv", low_memory=False)
effect_columns = [
    "model_id", "split", "n_seeds", "rmse_original_mean", "rmse_original_std",
    "rmse_no_smap_mean", "rmse_no_smap_std", "effect_rmse_mean", "effect_rmse_std",
    "effect_rmse_pct_mean", "effect_rmse_pct_std", "improved_seeds", "worsened_seeds",
    "pearson_change_mean", "diff_pearson_change_mean",
]
print("REPORT_BEGIN::OLD_NEW_EFFECT")
if effect_summary.empty:
    print("No paired old-vs-new rows available.")
else:
    print(effect_summary[effect_columns].sort_values(["split", "effect_rmse_mean"], ascending=[True, False]).to_markdown(index=False, floatfmt=".6f"))
    print("Trend correlation is Pearson change; first-difference Pearson change is unavailable because the original reference summaries do not contain first-difference predictions.")
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
        markdown("""## Trend figures

The next cell generates two five-line figure suites per ECE station: architecture gates and the three alternative regime gates, each against observed target values."""),
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
