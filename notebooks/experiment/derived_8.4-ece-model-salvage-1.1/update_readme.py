"""Regenerate the experiment README from executed notebook stdout."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
NOTEBOOK = EXP_DIR / "derived_8.4-ece-model-salvage-1.1.ipynb"
README = EXP_DIR / "README.md"


def notebook_outputs() -> str:
    result = subprocess.run(
        ["nb", "read", str(NOTEBOOK), "--limit", "1000000"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def extract_sections(text: str) -> dict[str, str]:
    outputs = re.findall(r"@@output .*?\n```text\n(.*?)\n```", text, flags=re.DOTALL)
    sections: dict[str, str] = {}
    combined_output = "\n".join(outputs)
    for match in re.finditer(
        r"REPORT_BEGIN::([A-Z0-9_]+)\n(.*?)\nREPORT_END::\1",
        combined_output,
        flags=re.DOTALL,
    ):
        sections[match.group(1)] = match.group(2).strip()
    return sections


def _figure_names(section: str, marker: str = "FIGURE::") -> list[str]:
    names = [
        line.removeprefix(marker).strip()
        for line in section.splitlines()
        if line.startswith(marker)
    ]
    for name in names:
        if not name.endswith(".png") or Path(name).name != name:
            raise RuntimeError(f"Invalid generated figure filename: {name}")
        if not (EXP_DIR / "figures" / name).exists():
            raise FileNotFoundError(EXP_DIR / "figures" / name)
    return names


def main() -> None:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)
    sections = extract_sections(notebook_outputs())
    required = {
        "SELECTION", "INPUT_AUDIT", "FEATURE_AUDIT", "ROUTER_AUDIT", "METRICS",
        "REFERENCE_COMPARISON", "SALVAGE_1_1_VS_1_0", "OLD_NEW_EFFECT",
        "SMAP_INVARIANCE", "GLOBAL_VERSION", "FIGURES",
    }
    missing = sorted(required - sections.keys())
    if missing:
        raise RuntimeError(f"Notebook stdout is missing report sections: {missing}")

    effect_names = _figure_names(sections["OLD_NEW_EFFECT"])
    if effect_names != ["old_vs_new_rmse_effect.png"]:
        raise RuntimeError("OLD_NEW_EFFECT must contain the generated RMSE effect figure.")
    version_names = _figure_names(sections["GLOBAL_VERSION"])
    if len(version_names) != 5:
        raise RuntimeError(
            "GLOBAL_VERSION must contain exactly one chart for each of five ECE stations."
        )
    trend_names = [
        line.strip().removeprefix("-").strip()
        for line in sections["FIGURES"].splitlines()
        if line.strip()
    ]
    for name in trend_names:
        if not name.endswith(".png") or Path(name).name != name:
            raise RuntimeError(f"Invalid trend figure filename: {name}")
        if not (EXP_DIR / "figures" / name).exists():
            raise FileNotFoundError(EXP_DIR / "figures" / name)

    all_figures = version_names + trend_names
    figure_links = "\n\n".join(
        f"![{name}](figures/{name})" for name in all_figures
    )
    version_links = "\n\n".join(
        f"![{name}](figures/{name})" for name in version_names
    )
    content = f"""# Experiment: `derived_8.4-ece-model-salvage-1.1`

## Objective

This descendant of 1.0 reruns the local MI → ElasticNet → stability → wrapper feature-selection pipeline after filtering every SMAP-derived feature. It produces nested 40/50/60/69 feature manifests: six routing families use 60 features and the global family is evaluated at all four sizes. There are no delta or specialist additions. Selector, router, and model fitting use WA trainval only; ECE targets are evaluation-only.

## Model families and seeds

The experiment covers V0 KMeans, backbone KMeans, supervised trained gating, G_API gating, dynamic gating, seasonal gating, and the single-regime global model. The six routing families use 60 features; the global model uses 40, 50, 60, and 69 feature variants. Every variant uses learner seeds `[42, 7, 13]`; router seed is fixed at 42.

## Feature-selection provenance

{sections['SELECTION']}

## Input audit

```text
{sections['INPUT_AUDIT']}
```

## Feature audit

{sections['FEATURE_AUDIT']}

## Router audit

{sections['ROUTER_AUDIT']}

## Metrics

The temporal results include the complete 2023–2025 WA test period and the matched 2025-07-20 through 2025-08-19 summer window. Spatial results use the native-missing `derived_8.4_ece_v3` ECE test set. RMSE, MAE, bias, ubRMSE, R², Pearson, standard deviations, and first-difference Pearson are reported.

{sections['METRICS']}

## Comparison with original SMAP-trained models

Original formal-evaluation runs remain reference-only and are not retrained or used by the selector or models.

{sections['REFERENCE_COMPARISON']}

## Comparison of 1.1 selected features with 1.0 no-SMAP models

This same-seed comparison isolates the feature-selection change from the original SMAP-removal change.

{sections['SALVAGE_1_1_VS_1_0']}

## Effect of Removing SMAP: ECE Benefit vs WA Degradation

The paired summary uses original RMSE − no-SMAP RMSE for ECE benefit and no-SMAP RMSE − original RMSE for WA degradation. Positive values have the stated interpretation.

{sections['OLD_NEW_EFFECT']}

![old_vs_new_rmse_effect.png](figures/old_vs_new_rmse_effect.png)

## No-SMAP invariance

Seed-42 predictions were checked after replacing all ECE SMAP columns with zero.

{sections['SMAP_INVARIANCE']}

## Global model version comparison

The following charts align station/date keys and average predictions over the common seeds `[42, 7, 13]`. Each chart has exactly seven lines: original `Global_Single_54`, 1.0 `Global_Single_54_no_smap`, 1.1 global models at 40/50/60/69 features, and ground truth.

{sections['GLOBAL_VERSION']}

{version_links}

## Figures

All generated paths below use the notebook-relative `figures/<filename>` form. Trend charts contain no more than five lines.

{figure_links}

## Reproduction

```bash
cd notebooks/experiment/derived_8.4-ece-model-salvage-1.1
uv run --no-sync python run_feature_selection.py --stage all
uv run --no-sync python run_model_salvage.py --resume
uv run --no-sync python build_notebook.py
nb execute derived_8.4-ece-model-salvage-1.1.ipynb --uv --timeout 1800
uv run --no-sync python update_readme.py
```

The Slurm workflow separates feature selection and model/report execution on `gpu_debug`, with the second stage dependent on successful selection.
"""
    README.write_text(content, encoding="utf-8")
    print(f"[README] updated {README}")


if __name__ == "__main__":
    main()
