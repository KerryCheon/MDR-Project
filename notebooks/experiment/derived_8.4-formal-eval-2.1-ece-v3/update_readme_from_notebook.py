"""Generate README.md for derived_8.4-formal-eval-2.1-ece-v3 strictly from executed notebook output."""

import json
import re
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
nb_path = EXP_DIR / "derived_8.4-formal-eval-2.1-ece-v3.ipynb"
readme_path = EXP_DIR / "README.md"

if not nb_path.exists():
    raise FileNotFoundError(f"Notebook {nb_path} not found.")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Header-keyed stdout extraction: map each code cell's stdout to the nearest
# preceding markdown header, so adding/removing cells never silently empties
# README sections (legacy code used hardcoded outputs.get(3)/get(5)/...).
sections: dict[str, str] = {}
legacy: dict[int, str] = {}
takeaways_md = ""
last_header = ""
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        m = re.search(r"^#{1,3}\s*(.+)", text.strip(), re.MULTILINE)
        last_header = m.group(1).strip().lower() if m else text.strip()[:80].lower()
        if "key takeaways" in last_header and not takeaways_md:
            takeaways_md = text.strip()
    elif cell["cell_type"] == "code":
        text = ""
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                parts = out.get("text", [])
                text += "".join(parts) if isinstance(parts, list) else str(parts)
        text = text.strip()
        legacy[i] = text
        if last_header and text and last_header not in sections:
            sections[last_header] = text


def _get(*keywords: str, fallback_idx: int) -> str:
    for key, val in sections.items():
        if all(k.lower() in key for k in keywords):
            return val
    return legacy.get(fallback_idx, "")


configs_md = _get("configurations", fallback_idx=3)
temporal_md = _get("temporal results", fallback_idx=5)
temporal_pair_md = _get("temporal focused", fallback_idx=7)
temporal_boot_md = _get("temporal sample-level", fallback_idx=9)
spatial_md = _get("in-situ ece spatial results", fallback_idx=11)
station_md = _get("per-station breakdown", fallback_idx=13)
spatial_pair_md = _get("spatial focused pairwise", fallback_idx=15)
spatial_boot_md = _get("spatial sample-level", fallback_idx=17)
focused_md = _get("focused in-situ", fallback_idx=19)
delta_md = _get("delta-robustness", fallback_idx=23)
repl_md = _get("replication checks", fallback_idx=25)
# Conclusion body = takeaways markdown cell minus its own top-level header
# (README provides the ## header). Falls back to "" if the cell is missing.
takeaways_body = re.sub(r"^#+\s*.*\n", "", takeaways_md, count=1).strip()

readme_content = f"""# Experiment: `derived_8.4-formal-eval-2.1-ece-v3` — Formal Statistical Evaluation on In-Situ ECE Sensors (v3 Split & Salvaged Router)

## Objective

Publication-oriented formal statistical evaluation of the two-regime clustering models against global baselines and trained gating, evaluated under **in-situ spatial generalization** across 5 newly deployed sensor stations on the canonical **`derived_8.4_ece_v3`** dataset split (150 rows across 2026-07-20 to 2026-08-19 in Bellevue and Renton, WA).

All models and routers are trained **strictly on the 7 Washington State stations** (`derived_8.4` `trainval`, 14,608 rows). The in-situ dataset `derived_8.4_ece_v3` is **completely unseen** during training.

### Key Methodology Updates in v2.1:
1. **Canonical `derived_8.4_ece_v3` Evaluation:** Features native-NaN SMAP satellite channels reflecting true in-situ deployment conditions where SMAP is absent or degraded.
2. **Missingness-Aware MoE Router Salvage:** Incorporates the availability gate fix ($\\\\tau = 0.10$), automatically routing samples with missing SMAP features through the SMAP-free `Univariate_G_API_k2` router.
3. **Primary Metric Realism (RMSE):** In this 30-day late-summer dry-down window, soil moisture target variance is extremely small ($\\\\sigma_y \\\\in [0.003, 0.008]\\\\text{{ m}}^3/\\\\text{{m}}^3$). Modest residuals ($\\\\sim 0.04$ to $0.05$) unavoidably produce large negative $R^2$ values due to tiny denominators. Models are therefore **ranked primarily by RMSE (ascending, lower is better)**, with $R^2$, MAE, Bias, ubRMSE, and Pearson correlation ($r$) reported alongside.
4. **Trend Directionality via Pearson Correlation:** Evaluates whether model predictions faithfully reproduce the ground-truth drying curve.
5. **Time Series Line Charts (Strictly $\\\\le 5$ Lines per Chart):**
   - **Chart Suite 1 (Architecture Showdown, NO per-regime deltas):** Observed In-Situ Ground Truth, `Clustering_V0_Full_k2 c0=0,c1=0`, `Clustering_Backbone54_k2 c0=0,c1=0`, `Global_Single_54`, `Trained_Gating_k2 c0=0,c1=0`.
   - **Chart Suite 2 (Regime Benchmark Showdown):** Observed In-Situ Ground Truth, `Clustering_V0_Full_k2 c0=0, c1=0`, `Univariate_G_API_k2 c0=0, c1=0`, `Clustering_Dynamic_k2 c0=0, c1=0`, `Seasonal_Binary_k2 c0=0, c1=0`.

All tables below are copied verbatim from the stdout of the executed report notebook
(`derived_8.4-formal-eval-2.1-ece-v3.ipynb`, executed with `nb execute --uv` from `notebooks/`).

---

## Configurations (20)

{configs_md}

---

## Protocol

- **Training:** Models and routers trained strictly on the 7 Washington state stations from `derived_8.4` (`trainval`, 2017–2022, 14,608 rows). `derived_8.4_ece_v3` is **completely unseen** during training.
- **Temporal evaluation:** Evaluated on the frozen Washington test set (2023–2025, 6,620 rows, 7 WA stations), **30 random seeds** (seeds 42, 7, 13, ..., 2222; seed 42 included as exact replication anchor vs eval-1.1).
- **Spatial evaluation:** Evaluated on all 5 in-situ stations from `derived_8.4_ece_v3` (150 rows across 2026-07-20 to 2026-08-19 in Bellevue and Renton, WA), **30 random seeds**.
- **Missingness-Aware Router:** Availability gate ($\\tau=0.10$) detects missing SMAP channels in `derived_8.4_ece_v3` and falls back dynamically to the SMAP-free `Univariate_G_API_k2` router.
- **Primary Metric:** Models ranked primarily by **RMSE (m³/m³)** ascending (lower is better).
- **Statistics:** Seed-level (mean ± std, median, 95% t-CI, paired t-test, Wilcoxon signed-rank, % seeds A better), sample-level (paired cluster bootstrap over (station, date) blocks, percentile 95% CI + bootstrap p), Benjamini–Hochberg FDR, and per-station win counts + two-sided binomial sign test (n = 5; 5/5 → p ≈ 0.0625, 4/5 → p ≈ 0.3750).

---

## Temporal results (Washington test set, 2023–2025, 30 seeds)

{temporal_md}

### Focused temporal pairwise comparisons

{temporal_pair_md}

### Temporal sample-level bootstrap (paired cluster bootstrap over (station, month) blocks; seed-42 fits)

{temporal_boot_md}

---

## In-Situ ECE Spatial results (5 unseen stations, derived_8.4_ece_v3, 30 seeds)

{spatial_md}

### Per-station breakdown across 5 In-Situ ECE stations

{station_md}

### Focused Spatial pairwise tests (5 In-Situ ECE stations)

{spatial_pair_md}

### Spatial sample-level bootstrap (5 ECE stations)

{spatial_boot_md}

---

## Focused In-Situ ECE Spatial Comparison (No Delta Feature Selection)

A low-noise architectural comparison evaluating two-regime models strictly **without regime-specific feature selection**
(identical 54 global features) against single-regime global models and trained gating routers across the 5 ECE stations:

{focused_md}

---

## Delta-robustness (Temporal WA vs Spatial ECE)

{delta_md}

---

## Replication checks (seed 42 must reproduce the deterministic historical runs)

```
{repl_md}
```

---

## Publication Figures

### 1. Architecture Showdown Time Series (Observed + 2 No-Delta Clustering Regimes + Global + No-Delta Gating; $\\\\le 5$ lines; the two clustering no-delta lines overlap — numerically identical on ECE)
![Architecture Showdown Combined](spatial_ece_timeseries_architecture_combined.png)

### 2. Regime Benchmark Showdown Time Series (Observed + 4 Zero-Delta Regimes; $\\\\le 5$ lines)
![Regime Benchmark Showdown Combined](spatial_ece_timeseries_regime_benchmark_combined.png)

### 3. Error Distributions & Model Comparison
![Spatial ECE seed boxplot RMSE](spatial_seed_boxplot_rmse.png)
![Spatial ECE seed boxplot R2](spatial_seed_boxplot_r2.png)
![Spatial ECE seed boxplot Pearson](spatial_seed_boxplot_pearson.png)

### 4. Robustness Across Feature Selection Sources
![Delta robustness RMSE](delta_robustness_rmse.png)
![Delta robustness R2](delta_robustness_r2.png)

---

## Conclusion (No-Delta Regimes)

{takeaways_body}

---

## Execution Source
Generated deterministically from `{nb_path.name}` via `nb execute --uv`.
"""

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content.strip() + "\n")

print(f"Successfully generated {readme_path} from executed notebook.")
