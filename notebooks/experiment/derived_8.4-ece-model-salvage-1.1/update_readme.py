"""Regenerate the experiment README from executed notebook stdout."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
NOTEBOOK = EXP_DIR / "derived_8.4-ece-model-salvage-1.1.ipynb"
README = EXP_DIR / "README.md"


def notebook_outputs() -> str:
    """Read executed stream outputs through nb without parsing display wrappers."""
    result = subprocess.run(
        ["nb", "read", str(NOTEBOOK), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    notebook = json.loads(result.stdout)
    streams: list[str] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") != "stream" or output.get("name") != "stdout":
                continue
            text = output.get("text", "")
            streams.append("".join(text) if isinstance(text, list) else str(text))
    return "\n".join(streams)


def extract_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for match in re.finditer(
        r"REPORT_BEGIN::([A-Z0-9_]+)\n(.*?)\nREPORT_END::\1",
        text,
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


def _without_figure_markers(section: str) -> str:
    """Remove notebook-internal figure discovery markers before embedding."""
    return "\n".join(
        line for line in section.splitlines() if not line.startswith("FIGURE::")
    ).strip()


def _table_blocks(text: str) -> list[list[str]]:
    """Return pipe-table blocks outside fenced code blocks."""
    blocks: list[list[str]] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        is_table_line = (
            not in_fence
            and line.startswith("|")
            and line.rstrip().endswith("|")
        )
        if is_table_line:
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _validate_table_blocks(text: str) -> None:
    """Validate GFM pipe-table shape without changing notebook-derived text."""
    for block_index, block in enumerate(_table_blocks(text), start=1):
        if len(block) < 2:
            raise RuntimeError(f"Markdown table {block_index} has no separator row.")
        expected_pipes = block[0].count("|")
        if expected_pipes < 3 or any(line.count("|") != expected_pipes for line in block):
            raise RuntimeError(f"Markdown table {block_index} has inconsistent pipe counts.")
        separator_cells = [cell.strip() for cell in block[1].strip("|").split("|")]
        if len(separator_cells) != expected_pipes - 1 or any(
            not re.fullmatch(r":?-{3,}:?", cell) for cell in separator_cells
        ):
            raise RuntimeError(f"Markdown table {block_index} has an invalid separator row.")


def _validate_provenance_section(section: str) -> None:
    blocks = _table_blocks(section)
    if len(blocks) != 1:
        raise RuntimeError("SELECTION must contain exactly one Markdown table.")
    table = blocks[0]
    headers = [cell.strip() for cell in table[0].strip("|").split("|")]
    expected_headers = [
        "status", "selected_features", "smap_features", "candidate_pool",
        "delta_additions", "selection_period", "fit_scope",
    ]
    if headers != expected_headers:
        raise RuntimeError(f"Unexpected provenance table headers: {headers}")
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in table[2:]
    ]
    if len(rows) != 4:
        raise RuntimeError("Provenance table must contain exactly four feature-size rows.")
    if [int(row[1]) for row in rows] != [40, 50, 60, 69]:
        raise RuntimeError("Provenance table feature sizes are incomplete or unordered.")
    if any(row[3] != "69" or row[4].lower() != "none" for row in rows):
        raise RuntimeError("Provenance table contains non-scalar or unexpected values.")
    manifests = re.findall(
        r"### Selected feature manifest \((\d+) features\)\n\n```text\n(.*?)\n```",
        section,
        flags=re.DOTALL,
    )
    if [int(size) for size, _ in manifests] != [40, 50, 60, 69]:
        raise RuntimeError("SELECTION must contain four ordered fenced manifests.")
    for size, body in manifests:
        features = [line.strip() for line in body.splitlines() if line.strip()]
        if len(features) != int(size) or any("smap" in feature.lower() for feature in features):
            raise RuntimeError(f"Invalid {size}-feature selection manifest.")


def _validate_reference_section(section: str) -> None:
    """Require one mean-across-seeds row for each model and evaluation split."""
    blocks = _table_blocks(section)
    if len(blocks) != 1:
        raise RuntimeError("REFERENCE_COMPARISON must contain exactly one Markdown table.")
    table = blocks[0]
    headers = [cell.strip() for cell in table[0].strip("|").split("|")]
    expected_headers = [
        "model_id", "dataset", "window", "n_seeds", "rmse_no_smap",
        "rmse_original", "rmse_delta_no_smap_minus_original", "pearson_no_smap",
        "pearson_original",
    ]
    if headers != expected_headers:
        raise RuntimeError(f"Reference comparison must be seed-averaged: {headers}")
    rows = table[2:]
    if len(rows) != 20:
        raise RuntimeError(
            "Reference comparison must contain exactly 20 model/split mean rows."
        )
    parsed = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in rows
    ]
    if any(len(row) != len(expected_headers) or row[3] != "3" for row in parsed):
        raise RuntimeError("Reference comparison rows must report n_seeds=3.")
    keys = [(row[0], row[1], row[2]) for row in parsed]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Reference comparison contains duplicate model/split rows.")


def _validate_readme_content(content: str) -> None:
    if re.search(r"(?:REPORT_BEGIN|REPORT_END|FIGURE)::|INTERPRETATION_(?:BEGIN|END)", content):
        raise RuntimeError("README contains internal notebook report markers.")
    fence_count = sum(1 for line in content.splitlines() if line.strip().startswith("```"))
    if fence_count % 2:
        raise RuntimeError("README contains an unclosed fenced block.")
    _validate_table_blocks(content)
    links = re.findall(r"!\[[^]]*\]\((figures/[^)]+\.png)\)", content)
    if len(links) != len(set(links)):
        raise RuntimeError("README contains duplicate figure links.")
    for link in links:
        if not (EXP_DIR / link).exists():
            raise FileNotFoundError(EXP_DIR / link)


def main() -> None:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)
    sections = extract_sections(notebook_outputs())
    required = {
        "SELECTION", "INPUT_AUDIT", "FEATURE_AUDIT", "ROUTER_AUDIT", "METRICS",
        "REFERENCE_COMPARISON", "SALVAGE_1_1_VS_1_0", "FEATURE_SELECTION_ROUND", "OLD_NEW_EFFECT",
        "SMAP_INVARIANCE", "GLOBAL_VERSION", "FIGURES", "MULTIPANEL", "BEST_GLOBAL_VALIDATION",
    }
    missing = sorted(required - sections.keys())
    if missing:
        raise RuntimeError(f"Notebook stdout is missing report sections: {missing}")
    _validate_provenance_section(sections["SELECTION"])
    _validate_reference_section(sections["REFERENCE_COMPARISON"])

    effect_names = _figure_names(sections["OLD_NEW_EFFECT"])
    if effect_names != ["old_vs_new_rmse_effect.png"]:
        raise RuntimeError("OLD_NEW_EFFECT must contain the generated RMSE effect figure.")
    version_names = _figure_names(sections["GLOBAL_VERSION"])
    if len(version_names) != 5:
        raise RuntimeError(
            "GLOBAL_VERSION must contain exactly one chart for each of five ECE stations."
        )
    multipanel_names = _figure_names(sections["MULTIPANEL"])
    expected_multipanel_names = [
        "ece_all_sensors_architecture_multipanel.png",
        "ece_all_sensors_regime_multipanel.png",
        "ece_all_sensors_global_feature_sizes_multipanel.png",
    ]
    if multipanel_names != expected_multipanel_names:
        raise RuntimeError(
            "MULTIPANEL must contain one chart for each configured model group."
        )
    best_global_names = _figure_names(sections["BEST_GLOBAL_VALIDATION"])
    if best_global_names != ["ece_all_sensors_global_best_validation_multipanel.png"]:
        raise RuntimeError(
            "BEST_GLOBAL_VALIDATION must contain exactly one diagnostic chart."
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

    # Version charts are embedded in their dedicated section above. Keep the
    # general Figures section focused on trend/effect charts so each generated
    # figure has one README link and the section remains easy to audit.
    figure_links = "\n\n".join(
        f"![{name}](figures/{name})" for name in trend_names
    )
    version_links = "\n\n".join(
        f"![{name}](figures/{name})" for name in version_names
    )
    multipanel_links = "\n\n".join(
        f"![{name}](figures/{name})" for name in multipanel_names
    )
    best_global_links = "\n\n".join(
        f"![{name}](figures/{name})" for name in best_global_names
    )
    feature_selection_section = sections["FEATURE_SELECTION_ROUND"]
    feature_selection_section = feature_selection_section.replace(
        "INTERPRETATION_BEGIN", "### Interpretation"
    ).replace("INTERPRETATION_END", "")
    reference_section = _without_figure_markers(sections["REFERENCE_COMPARISON"])
    old_new_effect_section = _without_figure_markers(sections["OLD_NEW_EFFECT"])
    global_version_section = _without_figure_markers(sections["GLOBAL_VERSION"])
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

{reference_section}

## Comparison of 1.1 selected features with 1.0 no-SMAP models

This same-seed comparison isolates the feature-selection change from the original SMAP-removal change.

{sections['SALVAGE_1_1_VS_1_0']}

## Before vs after the new feature-selection round

This global-only comparison isolates the new nested selector from the 1.0 manually selected no-SMAP baseline. Positive ECE values are benefits; positive WA values are degradations. The trend columns are after-minus-before changes, so positive correlation changes indicate stronger agreement with the observed level or temporal direction.

{feature_selection_section}

## Effect of Removing SMAP: ECE Benefit vs WA Degradation

The paired summary uses original RMSE − no-SMAP RMSE for ECE benefit and no-SMAP RMSE − original RMSE for WA degradation. Positive values have the stated interpretation. The effect chart uses a shared fixed y-axis of −0.04 to 0.04 RMSE.

{old_new_effect_section}

![old_vs_new_rmse_effect.png](figures/old_vs_new_rmse_effect.png)

## No-SMAP invariance

Seed-42 predictions were checked after replacing all ECE SMAP columns with zero.

{sections['SMAP_INVARIANCE']}

## Global model version comparison

The following charts align station/date keys and average predictions over the common seeds `[42, 7, 13]`. Each chart has exactly seven lines: original `Global_Single_54`, 1.0 `Global_Single_54_no_smap`, 1.1 global models at 40/50/60/69 features, and ground truth. Every ECE line chart uses the same fixed y-axis of 0.00 to 0.25 soil-moisture units.

{global_version_section}

{version_links}

## Multi-panel ECE sensor comparisons

Each image contains all five ECE sensors in separate panels for one model group: architecture, alternative regime gates, or global feature sizes. Every panel uses the common fixed y-axis of 0.00 to 0.25 soil-moisture units.

{multipanel_links}

## Original versus best 1.1 global validation

This focused diagnostic compares ground truth with the original `Global_Single_54` and the 1.1 global model selected by the lowest pooled ECE RMSE across the common seeds `[42, 7, 13]`. The bottom-right panel reports pooled ECE RMSE, MAE, Pearson correlation, and RMSE improvement. Similar trend shapes would support, but cannot by themselves prove, the hypothesis that SMAP availability drives the original-model consistency.

{best_global_links}

## Figures

All generated paths below use the notebook-relative `figures/<filename>` form. Trend charts contain no more than five lines and use the common fixed y-axis of 0.00 to 0.25 soil-moisture units.

{figure_links}

## Reproduction

```bash
cd notebooks/experiment/derived_8.4-ece-model-salvage-1.1
uv run --no-sync python run_feature_selection.py --stage all
uv run --no-sync python run_model_salvage.py
uv run --no-sync python build_notebook.py
nb execute derived_8.4-ece-model-salvage-1.1.ipynb --uv --timeout 1800
uv run --no-sync python update_readme.py
```

The Slurm workflow separates feature selection and model/report execution on `gpu_debug`, with the second stage dependent on successful selection.
"""
    _validate_readme_content(content)
    README.write_text(content, encoding="utf-8")
    print(f"[README] updated {README}")


if __name__ == "__main__":
    main()
