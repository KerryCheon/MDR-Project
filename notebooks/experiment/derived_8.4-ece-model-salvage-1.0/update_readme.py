"""Regenerate the experiment README from executed notebook stdout."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
NOTEBOOK = EXP_DIR / "derived_8.4-ece-model-salvage-1.0.ipynb"
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
        r"REPORT_BEGIN::([A-Z_]+)\n(.*?)\nREPORT_END::\1",
        combined_output,
        flags=re.DOTALL,
    ):
        sections[match.group(1)] = match.group(2).strip()
    return sections


def split_effect_report(section: str) -> tuple[str, str]:
    figure_lines = [line.removeprefix("FIGURE::").strip() for line in section.splitlines() if line.startswith("FIGURE::")]
    if len(figure_lines) != 1 or not figure_lines[0].endswith(".png") or Path(figure_lines[0]).name != figure_lines[0]:
        raise RuntimeError("OLD_NEW_EFFECT must contain exactly one generated PNG filename.")
    figure_name = figure_lines[0]
    figure_path = EXP_DIR / "figures" / figure_name
    if not figure_path.exists():
        raise FileNotFoundError(figure_path)
    report = "\n".join(line for line in section.splitlines() if not line.startswith("FIGURE::")).strip()
    return report, figure_name


def main() -> None:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)
    sections = extract_sections(notebook_outputs())
    required = {"INPUT_AUDIT", "FEATURE_AUDIT", "ROUTER_AUDIT", "METRICS", "REFERENCE_COMPARISON", "OLD_NEW_EFFECT", "SMAP_INVARIANCE", "FIGURES"}
    missing = sorted(required - sections.keys())
    if missing:
        raise RuntimeError(f"Notebook stdout is missing report sections: {missing}")

    figure_paths = [line.strip().removeprefix("-").strip() for line in sections["FIGURES"].splitlines() if line.strip()]
    figure_lines = "\n\n".join(f"![{path}](figures/{path})" for path in figure_paths)
    effect_report, effect_figure = split_effect_report(sections["OLD_NEW_EFFECT"])
    content = f"""# Experiment: `derived_8.4-ece-model-salvage-1.0`

## Objective

Retrain the current two-regime routing families and the 54-feature single-regime baseline after removing all SMAP-derived inputs. Models and routers are fitted strictly on the seven Washington training stations; the ECE target is used only for spatial evaluation.

## Model families

The experiment covers V0 KMeans, Backbone54 KMeans, supervised trained gating, G_API gating, dynamic gating, seasonal gating, and the `Global_Single_54` lineage. The effective global/model input count is 42 after filtering the parent 54-feature list; the V0 router also retains 42 inputs because its canonical 50-feature list contains eight SMAP-derived columns. The dynamic router retains `G_API` and `LST_modis`.

## Input audit

```text
{sections['INPUT_AUDIT']}
```

## Feature audit

{sections['FEATURE_AUDIT']}

## Router audit

{sections['ROUTER_AUDIT']}

## Metrics

The temporal results include the complete 2023–2025 WA test period and the matched 2025-07-20 through 2025-08-19 summer window. Spatial results use all rows in the native-missing `derived_8.4_ece_v3` ECE test set. RMSE is the primary error metric; Pearson and first-difference Pearson measure trend following.

{sections['METRICS']}

## Comparison with original models

Original SMAP-trained results are reference-only and are not used for fitting.

{sections['REFERENCE_COMPARISON']}

## Effect of Removing SMAP: ECE Benefit vs WA Degradation

The paired summary below reports the five-seed effect for each model on the pooled five-station ECE set and pooled seven-station WA temporal test. Positive ECE effect values are benefits (original RMSE − no-SMAP RMSE); positive WA effect values are degradations (no-SMAP RMSE − original RMSE). Pearson change is the available level/trend-correlation comparison. The original runs remain reference-only.

{effect_report}

![old_vs_new_rmse_effect.png](figures/{effect_figure})

## No-SMAP invariance

Seed-42 predictions were checked after replacing all ECE SMAP columns with zero.

{sections['SMAP_INVARIANCE']}

## Figures

Each figure contains observed values plus four model lines, staying within the five-line limit.

{figure_lines}

## Reproduction

```bash
cd notebooks
uv run --no-sync python experiment/derived_8.4-ece-model-salvage-1.0/run_model_salvage.py --smoke
sbatch experiment/derived_8.4-ece-model-salvage-1.0/run_slurm.sh
```

The Slurm job runs the full five-seed experiment, builds the notebook with `nb`, executes it with `nb execute --uv`, and regenerates this README from notebook stdout.
"""
    README.write_text(content, encoding="utf-8")
    print(f"[README] updated {README}")


if __name__ == "__main__":
    main()
