# derived_8.2-eval-3.2 — Feature-set ablation (2-regime MoE)

Isolate **feature-set quality** for K=2 Mixture-of-Experts models on Washington-only `derived_8.2`, under **frozen 1.3-lite** XGBoost hyperparameters. Parallel final evaluation follows `derived_8.2-hyperparameters-1.5` (`ThreadPoolExecutor`, CRC cache, per-worker exception isolation).

## Protocol

| Item | Value |
|------|--------|
| Split | `data/splits/derived_8.2/` (train+val → trainval; test held out) |
| Target | `soil_moisture_5cm` |
| Weighting | Unweighted (no temporal drift) |
| Hyperparameters | **1.3-lite** (max_depth=8, MCW=10, λ=1.5, α=0.03, sub=0.9, col=0.8, Est=1500, LR=0.01) |
| Seed | 42 |
| Device | CUDA if available |
| Parallelism | `XGB_PARALLEL_WORKERS` (default **6**); XGB `n_jobs=1` |
| Feature selection | V6 **c1_baseline_bypass_off** (recommended for 8.2 in feature-selection-2.1) |

## Feature arms

| Arm | Meaning |
|-----|---------|
| **Spec-old** | Specialist features from eval-3.1 / metadata (`previous_features.json`) |
| **Spec-new** | Fresh c1 pipeline selection per K=2 subset (`selected_features.json`) |
| **Global-V3** | All specialists use `OVERALL_SELECTED_FEATURES_V3` (47) |
| **Global-c1** | All specialists use V6 c1 global list (50) from feature-selection-2.1 |

## Model matrix (18)

| ID | Model |
|----|-------|
| 1 | Baseline V3 (global single model) |
| 2 | Baseline c1 (global single model) |
| 3–6 | Trained Gating K=2 × {Spec-old, Spec-new, Global-V3, Global-c1} |
| 7–10 | Univariate G_API K=2 × 4 arms |
| 11–14 | Clustering Dynamic K=2 × 4 arms |
| 15–18 | Seasonal Binary K=2 × 4 arms |

Routing definitions match eval-3.1 (binary dry/wet threshold 0.16; G_API quantiles; KMeans on SMAP lag1 + G_API + LST; seasonal months).

Gating classifiers (trained gating only) always use **V3 features** so only specialist feature lists vary across arms.

## Sanity anchors

| Baseline | Expected test R² (approx.) | Source |
|----------|----------------------------|--------|
| V3 global | **0.6551** | eval-3.1 / feature-selection-2.1 no-drift |
| c1 global | **0.6648** | feature-selection-2.1 no-drift |

Residual ΔR² ≲ 0.003 is normal env noise.

## How to run

```bash
# 1) Specialist feature selection (c1 pipeline on each K=2 subset)
cd /path/to/MDR-Project
PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-eval-3.2/run_feature_selection.py

# Smoke test (fewer stability boots):
PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-eval-3.2/run_feature_selection.py --n-boot 10

# 2) Eval notebook (from notebooks/ so uv env resolves)
cd notebooks
nb execute experiment/derived_8.2-eval-3.2/derived_8.2-eval-3.2.ipynb --uv

# Optional: fewer GPU workers
XGB_PARALLEL_WORKERS=2 nb execute experiment/derived_8.2-eval-3.2/derived_8.2-eval-3.2.ipynb --uv
```

Resume: model predictions are cached under `models/model_{id}_{crc}_*` keyed by config + feature CRC32. Failed jobs land in `failed_configs.csv`; re-run continues.

## Layout

| Path | Purpose |
|------|---------|
| `configs/config_c1_baseline_bypass_off.yaml` | Pipeline params (from feature-selection-2.1) |
| `previous_features.json` | Spec-old + Global-V3 copied from eval-3.1 / metadata |
| `run_feature_selection.py` | Spec-new selection runner |
| `selected_features.json` | Spec-new + embedded global c1 |
| `derived_8.2-eval-3.2.ipynb` | Parallel MoE evaluation |
| `metrics_summary.csv` | Overall metrics |
| `metrics_by_year.csv` | Yearly metrics |
| `ablation_r2_strategy_x_arm.csv` | Strategy × arm R² pivot |
| `gating_performance_summary.csv` | Trained-gating router metrics |
| `all_models_loss_curves.csv` | Step-wise test RMSE |
| `models/` | Cached boosters + pred arrays |

## Results

Executed on CUDA with `XGB_PARALLEL_WORKERS=4`. Sanity anchors reproduced exactly:

| Baseline | R² | Expected |
|----------|---:|---------:|
| V3 | **0.6551** | 0.6551 |
| c1 | **0.6648** | 0.6648 |

### Overall leaderboard

| Rank | Model | R² | RMSE | MAE |
|------|-------|---:|-----:|----:|
| 1 | **Clustering Dynamic K=2 (Global-c1)** | **0.6672** | **0.0607** | **0.0446** |
| 2 | Baseline c1 | 0.6648 | 0.0610 | 0.0450 |
| 3 | Baseline V3 | 0.6551 | 0.0618 | 0.0464 |
| 4 | Univariate G_API K=2 (Global-c1) | 0.6518 | 0.0621 | 0.0449 |
| 5 | Seasonal Binary K=2 (Global-c1) | 0.6484 | 0.0624 | 0.0454 |
| 6 | Seasonal Binary K=2 (Global-V3) | 0.6400 | 0.0632 | 0.0473 |
| 7 | Univariate G_API K=2 (Global-V3) | 0.6388 | 0.0633 | 0.0474 |
| 8 | Clustering Dynamic K=2 (Spec-old) | 0.6273 | 0.0643 | 0.0475 |
| 9 | Seasonal Binary K=2 (Spec-new) | 0.6270 | 0.0643 | 0.0471 |
| 10 | Clustering Dynamic K=2 (Spec-new) | 0.6258 | 0.0644 | 0.0475 |
| 11 | Trained Gating K=2 (Global-c1) | 0.6133 | 0.0655 | 0.0469 |
| 12 | Clustering Dynamic K=2 (Global-V3) | 0.6029 | 0.0664 | 0.0497 |
| 13 | Seasonal Binary K=2 (Spec-old) | 0.5970 | 0.0669 | 0.0497 |
| 14 | Trained Gating K=2 (Spec-new) | 0.5791 | 0.0683 | 0.0488 |
| 15 | Trained Gating K=2 (Global-V3) | 0.5778 | 0.0684 | 0.0498 |
| 16 | Trained Gating K=2 (Spec-old) | 0.5712 | 0.0690 | 0.0487 |
| 17 | Univariate G_API K=2 (Spec-old) | 0.5395 | 0.0715 | 0.0511 |
| 18 | Univariate G_API K=2 (Spec-new) | 0.5368 | 0.0717 | 0.0512 |

### Ablation: R² by Strategy × Arm

| Strategy | Spec-old | Spec-new | Global-V3 | Global-c1 |
|----------|---------:|---------:|----------:|----------:|
| Clustering Dynamic K=2 | 0.6273 | 0.6258 | 0.6029 | **0.6672** |
| Seasonal Binary K=2 | 0.5970 | 0.6270 | 0.6400 | **0.6484** |
| Trained Gating K=2 | 0.5712 | 0.5791 | 0.5778 | **0.6133** |
| Univariate G_API K=2 | 0.5395 | 0.5368 | 0.6388 | **0.6518** |

### Spec-new selection sizes (c1 pipeline, n_boot=50)

| Partition | n features | Notes |
|-----------|----------:|-------|
| binary dry / wet | 20 / 50 | |
| Univariate G_API c0 / c1 | 20 / **1** | Cluster 1 collapses to a single static feature (same pathology as Spec-old) |
| Clustering Dynamic c0 / c1 | 50 / 39 | |
| Seasonal Binary c0 / c1 | 50 / 46 | |

### Key findings

1. **Baseline V3 vs Baseline c1:** c1 global selection lifts single-model R² from 0.6551 → **0.6648** (+0.0097), matching feature-selection-2.1.
2. **Spec-old vs Spec-new:** Mixed. Seasonal Binary improves substantially (0.5970 → 0.6270). Trained gating gains slightly (0.5712 → 0.5791). Clustering Dynamic is essentially flat (0.6273 → 0.6258). Univariate remains broken under both (1-feature specialist).
3. **Spec-new vs Global-c1:** For every strategy, **sharing the global c1 set beats per-subset c1 selection**. Specialist FS still underperforms on small/unstable partitions.
4. **MoE vs global baseline:** **Clustering Dynamic K=2 + Global-c1** reaches **0.6672**, edging past Baseline c1 (0.6648). This is the only MoE config that beats the best global model overall.
5. **Global-c1 is the best feature arm** for all four strategies — upgrading the *global* feature set helps MoE more than re-selecting per regime/cluster with c1.
6. **Trained target-gating** remains weak (max 0.6133) even with c1 features; routing error still dominates (gating accuracy 0.873, identical across arms because the classifier uses fixed V3 features).

## What this deliberately does not do

- No K=3 models
- No hyperparameter retuning (feature isolation only)
- Does not promote feature lists into `dataset_metadata.py` (local experiment artifacts only)
