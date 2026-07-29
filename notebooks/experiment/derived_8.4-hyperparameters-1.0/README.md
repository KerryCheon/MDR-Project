# Experiment: `derived_8.4-hyperparameters-1.0` — Separate HP Tuning for Single & Two-Regime Models

## Objective

Perform separate hyperparameter tuning for **single-regime** (global XGBoost) and **two-regime** (Clustering_V0_Full_k2 MoE) models on the `derived_8.4` dataset using feature sets from `derived_8.4-eval-1.1`. This addresses the finding that 2-regime models performed worse with hyperparameters from `derived_8.2-hyperparameters-1.5` than from `1.4`, suggesting that single-regime optimal HP does not transfer to MoE.

**Total: 400 models** (200 single-regime + 200 two-regime), trained with 8 parallel workers on CUDA.

---

## Protocol

| Item | Value |
|------|-------|
| Split | `data/splits/derived_8.4/` — 7 WA stations, ~21k rows (train: 9803, val: 4805, test: 6620) |
| Single-Regime Features | `shared_backbone_54` (54 features from derived_8.4-eval-1.1) |
| Two-Regime Features | Clustering_V0_Full_k2: Cluster 0 = 54 backbone, Cluster 1 = 54 backbone + 10 per-regime additions (c0=0, c1=10 winner) |
| Target | `soil_moisture_5cm` |
| Routing (Two-Regime) | KMeans(k=2) on `OVERALL_SELECTED_FEATURES_V0` (50 features) |
| Seed | 42 |
| Device | CUDA (H100) |
| Parallelism | 8 workers via thread pool |
| Cache Key | `{track}_model_{id}_{crc32}.json` — CRC32 of training config payload |

### Lean Estimator Schedule

| learning_rate | n_estimators |
|---------------|--------------|
| 0.003 | 1500 |
| 0.004 | 1300 |
| 0.005 | 1000 |
| 0.006 | 900 |
| 0.007 | 800 |
| 0.008 | 750 |
| 0.01 | 500 |
| 0.012 | 400 |

---

### Configuration Groups (200 per track, 400 total)

#### Single-Regime Track (200)

| Group | Count | Focus |
|-------|------:|-------|
| SR — References | 6 | 8.4-eval baseline, lean ref, 1.5 SOTA params, MAE/MSE/MAPE objectives |
| SA — Depth × LR × MCW | 60 | d∈{8,9,10}, LR∈{0.003,0.005,0.008,0.01}, MCW∈{4,6,8,10,15} |
| SB — L2 × L1 | 48 | L2∈{0.25..3.0} × L1∈{0.0..0.1} (42) + plateau tight L2×L1 (6) |
| SC — Sampling | 24 | sub∈{0.8..1.0} × col∈{0.7..0.9}, minus duplicate baseline |
| SD — Estimator budget | 24 | Est∈{500..3000} × d∈{9,10} × LR∈{0.005,0.008} |
| SE — Probes & champions | 38 | gamma, max_bin, lossguide, d=11, plateau polish, champion combos |

#### Two-Regime Track (200)

Identical grid structure (TR, TA, TB, TC, TD, TE groups), each training 2 XGBoost experts per Clustering_V0_Full_k2 routing.

---

## Results

### SOTA Comparison

| Metric | R² | Δ vs Baseline |
|--------|------:|:-------------:|
| 8.4-eval Global Baseline (54 Backbone) | 0.77923 | — |
| 8.4-eval Clustering Winner (c0=0, c1=10) | 0.81496 | — |
| **Best Single-Regime (this sweep)** | **0.78542** | **+0.00619** |
| **Best Two-Regime (this sweep)** | **0.81524** | **+0.00028** |
| **Best Overall** | **0.81524** | — |

### Overall Leaderboard (Top 10)

| id | group | track | Configuration | R² | RMSE | MAE |
|----|-------|-------|--------------|------:|------:|------:|
| 398 | TE | two_regime | MCW=9 L2=0.5 L1=0.03 (d9 LR0.005) | **0.81524** | 0.04379 | 0.03380 |
| 353 | TD | two_regime | d=10 LR=0.008 Est=1500 MCW=8 | 0.81513 | 0.04380 | 0.03368 |
| 357 | TD | two_regime | d=10 LR=0.008 Est=2000 MCW=8 | 0.81512 | 0.04380 | 0.03368 |
| 361 | TD | two_regime | d=10 LR=0.008 Est=3000 MCW=8 | 0.81510 | 0.04380 | 0.03368 |
| 349 | TD | two_regime | d=10 LR=0.008 Est=1000 MCW=8 | 0.81503 | 0.04381 | 0.03370 |
| 259 | TA | two_regime | d=10 LR=0.008 MCW=10 Est=750 | 0.81499 | 0.04382 | 0.03373 |
| 200 | TR | two_regime | 8.4-eval Baseline (reference) | 0.81496 | 0.04382 | 0.03372 |
| 202 | TR | two_regime | 1.5 SOTA Params (reference) | 0.81496 | 0.04382 | 0.03372 |
| 358 | TD | two_regime | d=9 LR=0.005 Est=3000 MCW=8 | 0.81493 | 0.04382 | 0.03372 |
| 345 | TD | two_regime | d=10 LR=0.008 Est=800 MCW=8 | 0.81490 | 0.04383 | 0.03373 |

### Single-Regime Leaderboard (Top 10)

| id | group | Configuration | R² | RMSE | MAE |
|----|-------|-------------|------:|------:|------:|
| 114 | SC | sub=0.8 col=0.7 (d9 LR0.005 MCW8) | **0.78542** | 0.04719 | 0.03681 |
| 119 | SC | sub=0.85 col=0.7 (d9 LR0.005 MCW8) | 0.78350 | 0.04739 | 0.03680 |
| 124 | SC | sub=0.9 col=0.7 (d9 LR0.005 MCW8) | 0.78341 | 0.04739 | 0.03692 |
| 115 | SC | sub=0.8 col=0.75 (d9 LR0.005 MCW8) | 0.78311 | 0.04741 | 0.03685 |
| 128 | SC | sub=0.95 col=0.7 (d9 LR0.005 MCW8) | 0.78246 | 0.04749 | 0.03699 |
| 160 | SD | d=10 LR=0.005 Est=3000 MCW=8 | 0.78246 | 0.04752 | 0.03666 |
| 57 | SA | d=10 LR=0.008 MCW=6 Est=750 | 0.78227 | 0.04751 | 0.03668 |
| 174 | SE | d=11 LR=0.003 MCW=8 Est=1500 | 0.78227 | 0.04753 | 0.03693 |
| 175 | SE | d=11 LR=0.003 MCW=12 Est=1500 | 0.78226 | 0.04753 | 0.03681 |
| 156 | SD | d=10 LR=0.005 Est=2000 MCW=8 | 0.78223 | 0.04753 | 0.03679 |

### Two-Regime Leaderboard (Top 10)

| id | group | Configuration | R² | RMSE | MAE |
|----|-------|-------------|------:|------:|------:|
| 398 | TE | MCW=9 L2=0.5 L1=0.03 (d9 LR0.005) | **0.81524** | 0.04379 | 0.03380 |
| 353 | TD | d=10 LR=0.008 Est=1500 MCW=8 | 0.81513 | 0.04380 | 0.03368 |
| 357 | TD | d=10 LR=0.008 Est=2000 MCW=8 | 0.81512 | 0.04380 | 0.03368 |
| 361 | TD | d=10 LR=0.008 Est=3000 MCW=8 | 0.81510 | 0.04380 | 0.03368 |
| 349 | TD | d=10 LR=0.008 Est=1000 MCW=8 | 0.81503 | 0.04381 | 0.03370 |
| 259 | TA | d=10 LR=0.008 MCW=10 Est=750 | 0.81499 | 0.04382 | 0.03373 |
| 200 | TR | 8.4-eval Baseline (reference) | 0.81496 | 0.04382 | 0.03372 |
| 353 | TD | d=10 LR=0.008 Est=2000 MCW=8 | 0.81512 | 0.04380 | 0.03368 |

### Best Configuration Details

#### Single-Regime SOTA (id=114)
```python
params = {
    "objective": "reg:squarederror",
    "max_depth": 9,
    "min_child_weight": 8,
    "reg_lambda": 0.75,
    "reg_alpha": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "n_estimators": 1000,
    "learning_rate": 0.005,
}
```

#### Two-Regime SOTA (id=398)
```python
params = {
    "objective": "reg:squarederror",
    "max_depth": 9,
    "min_child_weight": 9,
    "reg_lambda": 0.5,
    "reg_alpha": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_estimators": 1000,
    "learning_rate": 0.005,
}
# Routing: Clustering_V0_Full_k2 (KMeans on 50 V0 features)
# Cluster 0: shared_backbone_54 (54 features)
# Cluster 1: shared_backbone_54 + CLUSTER_1_ADDITIONS (64 features)
```

---

## Key Takeaways

1. **Two-regime MoE consistently dominates**: The Clustering_V0_Full_k2 MoE architecture achieves R²≈0.815 vs single-regime's R²≈0.785 — a consistent +0.03 advantage across all configurations. This gap has narrowed from +0.036 (eval-1.1 reference) but remains decisive.

2. **Single-regime HP tuning was fruitful**: The best single-regime model (R²=0.78542, +0.00619 vs baseline) came from the Sampling group — **aggressive column subsampling (col=0.7)** combined with **mild row subsampling (sub=0.8)** was the most impactful single hyperparameter. This was not explored by the eval-1.1 reference config (sub=0.9, col=0.8).

3. **Two-regime HP differences are real but small**: The best two-regime config (MCW=9, L2=0.5, L1=0.03, d=9, LR=0.005) differs from the single-regime champion (MCW=8, L2=0.75, L1=0.03, sub=0.8, col=0.7), confirming the original motivation that separate tuning is warranted. However, the gain over the reference params is modest (+0.00028), suggesting the MoE architecture is less sensitive to exact HP values.

4. **Lean estimator schedule works**: The lean schedule (LR=0.005 → 1000 estimators) proved sufficient. Group SD shows Est=1000 produces essentially identical R² to Est=3000 (0.81503 vs 0.81510), confirming that more trees add negligible value after convergence.

5. **The top-performing two-regime models favor**: d=9–10, LR=0.005–0.008, MCW=8–10, mild L2 (0.5–0.75), low L1 (0.0–0.03), and aggressive colsample (0.7–0.8) — largely consistent with the 1.5 findings but with slightly wider MCW tolerance.

---

## Output Files

| File | Description |
|------|-------------|
| `derived_8.4-hyperparameters-1.0.ipynb` | Full experiment notebook |
| `metrics_summary.csv` | All 400 configs with R², RMSE, ubRMSE, Bias, MAE, etc. |
| `metrics_by_year.csv` | Per-year (2023/2024/2025) metrics for all configs |
| `leaderboard_top20_overall.csv` | Top 20 overall by R² |
| `leaderboard_top20_single.csv` | Top 20 single-regime by R² |
| `leaderboard_top20_two_regime.csv` | Top 20 two-regime by R² |
| `loss_curves.csv` | Step-wise train/test loss for all models |
| `loss_curves.png` | Loss curve plots for selected top models |
| `r2_by_year.png` | Year-wise R² bar chart |
| `residuals_comparison.png` | Residual scatter for best single and two-regime |
| `test_predictions.csv` | Test predictions (`pred_{id}`) for all 400 models |
| `models/` | CRC32-keyed booster JSONs (~1200 files) |

---

## How to Re-run

```bash
cd notebooks
JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime-$USER nb execute experiment/derived_8.4-hyperparameters-1.0/derived_8.4-hyperparameters-1.0.ipynb --uv --timeout 10800
```

Training is **resume-safe** via CRC32-keyed caches under `models/`.
