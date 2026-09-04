# Implementation Plan: `derived_8.4-formal-eval-2.1-ece-v3`

## Goal Description

Create `derived_8.4-formal-eval-2.1-ece-v3` in `notebooks/experiment/` as the continuation of `derived_8.4-formal-eval-2.0-ece`, following the complete formal evaluation output protocol, to evaluate spatio-temporal soil moisture models on the newly deployed in-situ ECE sensor split (`derived_8.4_ece_v3`, 5 stations x 30 days = 150 rows, July 20 – August 19, 2026 in Bellevue and Renton, WA).

Key requirements and updates:
1. **Dataset Split:** Transition from legacy `derived_8.4-ece` to the canonical `derived_8.4_ece_v3` split (clean native-NaN SMAP, MODIS NDVI fallback, 30-day scaffold).
2. **MoE Routing Fix:** Integrate the input-only missingness-aware availability gate and auxiliary SMAP-free router from `derived_8.4-ece-router-salvage-2.0`, resolving the catastrophic SMAP missingness routing trap where dry-summer observations were wrongly routed to wet-mountain specialists.
3. **Primary Metric: RMSE:** Rank models primarily by RMSE (ascending, lower is better) while keeping $R^2$ on the side to prevent variance compression artifacts (short 30-day late-summer dry-down period with $\sigma_y \approx 0.003\text{--}0.008\,\text{m}^3/\text{m}^3$) from distorting model comparisons with extreme negative $R^2$.
4. **Time Series Line Charts (Strictly $\le 5$ Lines per Chart):**
    - **Chart Suite 1 (Architecture Showdown):** Replace `Baseline_V0_50` with salvaged `Clustering_V0_Full_k2` (5 lines):
      1. Observed In-Situ Ground Truth
      2. `Clustering_Backbone54_k2` (salvaged)
      3. `Clustering_V0_Full_k2` (salvaged)
      4. `Global_Single_54`
      5. `Trained_Gating_k2_c0_5_c1_10` (test-selected representative)
   - **Chart Suite 2 (Regime & Benchmark Showdown):** Compare salvaged `Clustering_V0_Full_k2` against heuristic and dynamic regimes (5 lines):
     1. Observed In-Situ Ground Truth
     2. `Clustering_V0_Full_k2` (salvaged)
     3. `Univariate_G_API_k2_c0_0_c1_0`
     4. `Clustering_Dynamic_k2_c0_0_c1_0`
     5. `Seasonal_Binary_k2_c0_0_c1_0`
5. **Pearson Correlation Analysis:** Calculate and report Pearson correlation ($r$) at both pooled and per-station levels to quantify whether model trajectories correctly track drying and wetting trends even when variance compression suppresses $R^2$.
6. **SLURM Batch Submission:** Provide and submit the job using `sbatch` on the TAMU ACES XPU accelerator partition (`pvc` with `--gres=gpu:pvc:1`).

---

## Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Data["1. Data Inputs"]
        WA["WA Reference Stations (derived_8.4)<br/>trainval (14,608 rows, 2017–2022)<br/>test (6,620 rows, 2023–2025)"]
        ECE["ECE In-Situ Sensors (derived_8.4_ece_v3)<br/>test.csv (150 rows, 5 stations x 30 days)<br/>SMAP features native-NaN"]
    end

    subgraph Models["2. Models & Routing"]
        Checkpoints["Frozen Experts (20 configs x 30 seeds)<br/>Reused from models/ symlink"]
        Router["Router with Salvage-2.0 Fix<br/>Availability Gate (tau=0.10, SMAP missing)"]
        AuxRouter["Auxiliary G_API Router<br/>(SMAP-free, fit on WA trainval)"]
        Router -->|SMAP available (WA test)| StaticKMeans["Static KMeans Routing"]
        Router -->|SMAP missing (ECE v3)| AuxRouter
    end

    subgraph Eval["3. Evaluation Engine"]
        RunTemp["run_temporal.py<br/>WA test (30 seeds)<br/>Deterministic verification"]
        RunSpat["run_spatial.py<br/>ECE v3 (30 seeds)<br/>RMSE-ranked + Pearson r"]
    end

    subgraph Output["4. Analysis & Artifacts"]
        DistDiag["analyze_cluster_distances.py<br/>OOD & Distance Diagnostics"]
        BuildNB["build_notebook.py<br/>Generates report notebook"]
        ExecuteNB["nb execute --uv<br/>Generates figures & stdout tables"]
        UpdateMD["update_readme_from_notebook.py<br/>Populates README.md strictly from stdout"]
    end

    WA --> Models
    ECE --> Models
    Models --> Eval
    Eval --> Output
```

---

## User Review Required

> [!IMPORTANT]
> **Primary Metric Pivot:** Models will be ranked by **RMSE** (ascending, lower is better) instead of $R^2$, directly reflecting the user's insight that small residuals over low-variance 30-day late-summer periods artificially magnify negative $R^2$. $R^2$, MAE, Bias, ubRMSE, and Pearson $r$ are retained on the side in all output tables.

> [!IMPORTANT]
> **Line Charts & Model Sets:**
> - **Chart 1 (Architecture Showdown):** `Baseline_V0_50` is replaced by `Clustering_V0_Full_k2`. Contains: Observed Ground Truth, `Clustering_Backbone54_k2` (salvaged), `Clustering_V0_Full_k2` (salvaged), `Global_Single_54`, `Trained_Gating_k2_c0_5_c1_10` (test-selected; exactly 5 lines).
> - **Chart 2 (Regime Benchmark Showdown):** Dedicated chart comparing salvaged `Clustering_V0_Full_k2` against `Univariate_G_API_k2 c0=0, c1=0`, `Clustering_Dynamic_k2 c0=0, c1=0`, and `Seasonal_Binary_k2 c0=0, c1=0` (exactly 5 lines).

> [!NOTE]
> **XPU / SLURM Partition:** The cluster is TAMU ACES, where "XPU" corresponds to Intel Data Center GPU Max 1550 (Ponte Vecchio) on the `pvc` partition (`#SBATCH --partition=pvc`, `#SBATCH --gres=gpu:pvc:1`).

---

## Proposed Changes

We will create the directory `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/` with all necessary scripts and modules.

```
notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/
├── config.yaml                          # Updated for ece_v3 split, routing fix, and metrics
├── .gitignore                           # Git ignore for artifacts, models, logs
├── setup_config.py                      # Setup script for generating configuration
├── eval_formal/                         # Modular evaluation package
│   ├── __init__.py
│   ├── configs.py                       # 20 pinned configurations metadata
│   ├── data.py                          # Loads derived_8.4 and derived_8.4_ece_v3 splits
│   ├── routers.py                       # Routers with salvage-2.0 availability gate
│   ├── evaluator.py                     # Evaluator supporting RMSE, R2, Pearson r, and soft/hard gating
│   ├── jobs.py                          # Multiprocessing worker execution manager
│   ├── stats.py                         # Hypothesis testing, Wilcoxon, FDR, Cluster Bootstrap
│   └── plots.py                         # Publication figures with strict MAX_LINES <= 5 guard
├── run_worker.py                        # Single-task worker CLI
├── run_temporal.py                      # 30-seed temporal evaluation on WA test
├── run_spatial.py                       # 30-seed spatial evaluation on ECE v3
├── analyze_cluster_distances.py         # Table 4 cluster distances and OOD diagnostics
├── build_notebook.py                    # Programmatic generator for the report notebook
├── update_readme_from_notebook.py       # Populates README.md strictly from notebook stdout
├── run_slurm.sh                         # SLURM submission script for ACES XPU (pvc partition)
├── val_selected_deltas.json             # Pinned validation deltas
├── models -> ../derived_8.4-formal-eval-2.0/models          # Symlink to frozen trained models
└── predictions -> ../derived_8.4-formal-eval-2.0/predictions # Symlink to temporal predictions
```

---

### Component 1: Configuration (`config.yaml` & `setup_config.py`)

#### [NEW] `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/config.yaml`
- Update `spatial_ece` dataset path from `derived_8.4-ece` to `data/splits/derived_8.4_ece_v3`.
- Configure `routing_fix`:
  - `enabled: true`
  - `tau_miss_rate: 0.10`
  - `smap_block_rule: "all_smap_router_features_nan"`
  - `aux_router: "Univariate_G_API"`
  - `temperature: 0.25`
  - `wa_percentile: 5`
- Primary ranking metric: `primary_metric: "rmse"`
- Chart lines restriction: `max_chart_lines: 5`
- Chart model sets:
  - `showdown_configs`: `["Clustering_Backbone54_k2_c0_10_c1_10", "Clustering_V0_Full_k2_c0_0_c1_10", "Global_Single_54", "Trained_Gating_k2_c0_5_c1_10"]` (with no-delta variants for architectural comparison)
  - `regime_configs`: `["Clustering_V0_Full_k2_c0_0_c1_0", "Univariate_G_API_k2_c0_0_c1_0", "Clustering_Dynamic_k2_c0_0_c1_0", "Seasonal_Binary_k2_c0_0_c1_0"]`

---

### Component 2: Routing Fix Implementation (`eval_formal/routers.py`)

#### [NEW] `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/eval_formal/routers.py`
Incorporate the missingness-aware availability gate from `derived_8.4-ece-router-salvage-2.0`:
- Detect whether router features contain missing SMAP columns or miss rate $> \tau$ ($\tau = 0.10$).
- When gated, fall back to the SMAP-free `UnivariateGAPIRouter` fitted on WA `trainval`.
- Provide `predict(frame)` (hard label: 0 or 1) and `predict_weights(frame)` (soft blend weights $w_0, w_1$ with WA-calibrated temperature $T=0.25$ and median margin pseudo-distances).
- Guarantee that clean WA test data evaluates identically to standard KMeans (gate inactive), preserving historical replication.

```python
def smap_router_features(router_features: list[str]) -> list[str]:
    return [f for f in router_features if "SMAP" in f]

def availability_gate(frame: pd.DataFrame, router_features: list[str], tau: float = 0.10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats = list(router_features)
    miss = frame.loc[:, feats].isna().to_numpy(dtype=float)
    miss_rate = miss.mean(axis=1)
    smap = smap_router_features(feats)
    if smap:
        smap_miss = frame.loc[:, smap].isna().to_numpy(dtype=float).mean(axis=1)
        smap_block_missing = smap_miss >= 1.0 - 1e-12
    else:
        smap_miss = np.zeros(len(frame))
        smap_block_missing = np.zeros(len(frame), dtype=bool)
    gated = smap_block_missing | (miss_rate > float(tau))
    return gated.astype(bool), miss_rate, smap_miss
```

---

### Component 3: Data Loading & Evaluation Logic (`eval_formal/data.py` & `eval_formal/evaluator.py`)

#### [NEW] `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/eval_formal/data.py`
- Load `data/splits/derived_8.4_ece_v3/test.csv` (150 rows, 5 stations x 30 days).
- Handle empty `train.csv` / `val.csv` in `derived_8.4_ece_v3` without error.
- Enforce schema validation (150 rows across Bellevue and Renton stations, July 20 – August 19, 2026).

#### [NEW] `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/eval_formal/evaluator.py`
- Ensure `compute_metrics` accurately calculates:
  - `rmse = root_mean_squared_error(y_true, y_pred)`
  - `r2 = r2_score(y_true, y_pred)`
  - `pearson = pearsonr(y_true, y_pred)[0]` (with guard for low variance)
  - `mae = mean_absolute_error(y_true, y_pred)`
  - `bias = np.mean(y_pred - y_true)`
  - `ubrmse = sqrt(max(0, rmse^2 - bias^2))`
- In `evaluate_spatial_ece()`:
  - Route through the salvaged router (availability gate + auxiliary SMAP-free router).
  - Load pre-trained expert models from `models/` symlink.
  - Return comprehensive pooled, per-station, yearly, and per-cluster metrics.

---

### Component 4: Publication Figures with <= 5 Lines (`eval_formal/plots.py`)

#### [NEW] `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/eval_formal/plots.py`
- Enforce hard assertion in all timeseries plotting functions:
  ```python
  MAX_LINES = 5
  assert len(ax.get_lines()) <= MAX_LINES, f"Expected at most {MAX_LINES} lines, got {len(ax.get_lines())}"
  ```
- Generate:
  1. **Architecture Showdown Timeseries (`spatial_ece_station_timeseries_architecture.png`):**
     Multi-panel timeseries (5 stations) with at most 5 lines:
     - Line 1: Observed Ground Truth (black, solid with marker dots)
     - Line 2: `Clustering_Backbone54_k2` (salvaged, blue)
     - Line 3: `Clustering_V0_Full_k2` (salvaged, cyan) *(replaces Baseline_V0_50)*
     - Line 4: `Global_Single_54` (red)
     - Line 5: `Trained_Gating_k2` (green)
     (Plus 5 individual per-station plots: `spatial_ece_timeseries_{station}_arch.png`).
  2. **Regime Benchmark Timeseries (`spatial_ece_station_timeseries_regimes.png`):**
     Multi-panel timeseries (5 stations) comparing salvaged Clustering V0 against other regimes (at most 5 lines):
     - Line 1: Observed Ground Truth (black, solid with marker dots)
     - Line 2: `Clustering_V0_Full_k2_c0_0_c1_0` (salvaged, cyan)
     - Line 3: `Univariate_G_API_k2_c0_0_c1_0` (purple)
     - Line 4: `Clustering_Dynamic_k2_c0_0_c1_0` (orange)
     - Line 5: `Seasonal_Binary_k2_c0_0_c1_0` (brown)
     (Plus 5 individual per-station plots: `spatial_ece_timeseries_{station}_regimes.png`).
  3. RMSE boxplots (`spatial_seed_boxplot_rmse.png`), pairwise RMSE differences (`paired_diff_spatial_rmse_*.png`), and delta robustness bars (`delta_robustness_rmse.png`).

---

### Component 5: Report Notebook & README Automation (`build_notebook.py` & `update_readme_from_notebook.py`)

#### [NEW] `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/build_notebook.py`
Build `derived_8.4-formal-eval-2.1-ece-v3.ipynb` containing:
1. Setup & configuration load.
2. 20 pinned configurations table.
3. Temporal evaluation summary (replicated from WA test set).
4. In-Situ ECE Spatial Summary **ranked by RMSE** (ascending, lower is better):
   - Table columns: `config_label`, `delta_source`, `spatial_mean_rmse`, `spatial_median_rmse`, `spatial_mean_pearson`, `spatial_mean_r2`, `spatial_mean_mae`, `spatial_mean_bias`.
5. Station Difficulty Ranking (sorted by mean RMSE):
   - Table columns: `station_id`, `n_configs`, `mean_rmse`, `median_rmse`, `mean_pearson`, `median_r2`, `mean_r2`, `mean_bias`.
6. Per-Configuration x Per-Station Matrices:
   - RMSE matrix: `spatial_per_config_station_rmse.csv`
   - Pearson correlation matrix: `spatial_per_config_station_pearson.csv`
   - $R^2$ matrix: `spatial_per_config_station_r2.csv`
7. Focused Spatial Pairwise Hypothesis Tests (evaluated on RMSE):
   - Win counts $k/5$ (lower RMSE wins), binomial sign test $p$, paired $t$-test $p$, Wilcoxon $p$, Benjamini-Hochberg FDR $q$.
8. Table 4: Cluster Centroid Distance & OOD Domain Shift Diagnostics.
9. Delta-source robustness table (ranked by RMSE).
10. Figure generation displaying both Chart Suite 1 (Architecture Showdown) and Chart Suite 2 (Regime Showdown), plus replication checks.

#### [NEW] `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/update_readme_from_notebook.py`
- Parse stdout of executed notebook cells and update `README.md` verbatim, adhering strictly to AGENTS.md reproducibility verification rules.

---

### Component 6: SLURM Batch Script (`run_slurm.sh`)

#### [NEW] `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/run_slurm.sh`
```bash
#!/bin/bash
#SBATCH --job-name=formal_21_ece_v3
#SBATCH --partition=pvc
#SBATCH --gres=gpu:pvc:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/slurm/%j.out
#SBATCH --error=artifacts/slurm/%j.err

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p artifacts/slurm

echo "=== [1/5] Running temporal evaluation (30 seeds) ==="
uv run python run_temporal.py

echo "=== [2/5] Running spatial evaluation (30 seeds on ECE v3 with routing fix) ==="
uv run python run_spatial.py

echo "=== [3/5] Running cluster distance & OOD diagnostics ==="
uv run python analyze_cluster_distances.py

echo "=== [4/5] Building report notebook ==="
uv run python build_notebook.py

echo "=== [5/5] Executing report notebook ==="
cd ../..
nb execute experiment/derived_8.4-formal-eval-2.1-ece-v3/derived_8.4-formal-eval-2.1-ece-v3.ipynb --uv
cd experiment/derived_8.4-formal-eval-2.1-ece-v3

echo "=== Updating README.md ==="
uv run python update_readme_from_notebook.py

echo "=== All Done ==="
```

---

## Verification Plan

### Automated Tests & Sanity Checks
1. **Directory & Symlink Verification:**
   - Verify `models` and `predictions` resolve to valid directories.
   - Verify `val_selected_deltas.json` is present.
2. **Unit / Smoke Test:**
   - Run `python run_spatial.py --smoke --max-configs 2 --seeds 42 7` to verify CPU execution, data loading, availability gate, and Pearson calculation.
   - Run `python -m eval_formal.stats` to verify statistical test self-checks.
3. **Notebook Build & Execute:**
   - Execute `build_notebook.py` to generate the `.ipynb` file.
   - Execute notebook with `nb execute ... --uv` and verify zero errors.
4. **Output Constraint Verification:**
   - Verify that all generated line charts strictly have $\le 5$ lines.
   - Verify that Chart Suite 1 replaces `Baseline_V0_50` with `Clustering_V0_Full_k2`.
   - Verify that Chart Suite 2 compares salvaged `Clustering_V0_Full_k2` against `Univariate_G_API_k2`, `Clustering_Dynamic_k2`, and `Seasonal_Binary_k2`.
   - Verify that RMSE is the primary sorting key in summary tables.
   - Verify Pearson correlation column is present in generated CSVs and README tables.
5. **SLURM Job Submission:**
   - Submit `run_slurm.sh` with `sbatch` to the `pvc` (XPU) partition.
   - Monitor job progress and ensure completion without error.
