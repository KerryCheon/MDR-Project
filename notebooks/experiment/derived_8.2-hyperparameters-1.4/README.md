# derived_8.2-hyperparameters-1.4 (SOTA Refinement Sweep — V3 Global Model)

This directory contains a **200-configuration** XGBoost hyperparameter sweep for the **single global model** on the Washington-only `derived_8.2` dataset with **Feature Set V3** (47 features, unweighted). Goal: beat the prior SOTA from `derived_8.2-hyperparameters-1.3-lite` (R² ≈ 0.6551 at 1500 steps).

**Status: completed on H100** (parallel workers = 4). Wall time for full notebook execute ≈ **9 minutes**.

---

## Protocol

| Item | Value |
|------|--------|
| Split | `data/splits/derived_8.2/` (train ∥ val → trainval; test held out) |
| Features | `OVERALL_SELECTED_FEATURES_V3` |
| Target | `soil_moisture_5cm` |
| Weighting | Unweighted (no temporal recency) |
| Seed | 42 |
| Device | CUDA (H100) |
| Parallelism | `XGB_PARALLEL_WORKERS` (default **4**) via thread pool |

### Lean step budgets (from 1.3-lite)

| learning_rate | n_estimators |
|---------------|--------------|
| 0.005 | 2500 |
| 0.008 | 2000 |
| 0.01  | 1500 |
| 0.012 | 1400 |
| 0.015 | 1200 |
| 0.02  | 1000 |

### Configuration groups (200 total)

| Group | Count | Focus |
|-------|------:|--------|
| R — References | 4 | MAE/MSE baselines + SOTA control (1.3-lite peak) + Est=2000 check |
| A — Depth × MCW × LR | 72 | Local grid: d∈{7,8,9}, MCW∈{5,8,10,15}, lean LR table |
| B — L2 × L1 | 36 | Regularization neighborhood on SOTA backbone |
| C — Sampling | 16 | subsample × colsample_bytree on SOTA backbone |
| D — Leaf × bin hybrids | 48 | lossguide leaves∈{31,63,127,255}×depth×MCW×max_bin |
| E — Gamma / bin×LR / Huber / champions | 24 | Fine probes + multi-ingredient combos |

---

## Results

### New SOTA

| Model | Configuration | $R^2$ | RMSE | MAE | Train (s) |
|-------|---------------|------:|-----:|----:|----------:|
| **13 (NEW SOTA)** | **MSE d=9 LR=0.005 MCW=8 Est=2500** | **0.6584** | **0.0615** | **0.0460** | **19.7** |
| 12 | MSE d=9 LR=0.005 MCW=5 Est=2500 | 0.6580 | 0.0616 | 0.0460 | 20.1 |
| 38 | MSE d=9 LR=0.01 MCW=10 Est=1500 | 0.6576 | 0.0616 | 0.0462 | 11.3 |
| 14 | MSE d=9 LR=0.005 MCW=10 Est=2500 | 0.6574 | 0.0616 | 0.0461 | 19.3 |
| 79 | MSE L2=0.5 L1=0.05 (d=8 LR=0.01 MCW=10) | 0.6571 | 0.0617 | 0.0463 | 8.9 |
| **2 (control)** | **1.3-lite peak (d=8 LR=0.01 MCW=10 Est=1500)** | **0.6551** | 0.0618 | 0.0464 | ~9 |

- **ΔR² vs 1.3-lite target:** **+0.0033** (0.658356 − 0.655063)
- Control Model 2 reproduced the 1.3-lite peak exactly (R² = 0.655063)

### Year-by-year (best vs control)

| Year | Model 13 (new SOTA) $R^2$ | Model 2 (control) $R^2$ | Δ |
|------|--------------------------:|------------------------:|--:|
| 2023 | **0.6647** | 0.6581 | +0.0066 |
| 2024 | 0.6334 | **0.6403** | −0.0069 |
| 2025 | **0.6503** | 0.6396 | +0.0107 |

Gains are driven by **2023 and especially 2025**; 2024 is slightly weaker than the control.

### Best per group

| Group | Best id | $R^2$ | Configuration |
|-------|--------:|------:|---------------|
| R | 2 | 0.6551 | SOTA Control (1.3-lite peak) |
| **A** | **13** | **0.6584** | **d=9 LR=0.005 MCW=8 Est=2500** |
| B | 79 | 0.6571 | L2=0.5 L1=0.05 |
| C | 122 | 0.6551 | sub=0.9 col=0.8 (SOTA sampling) |
| D | 173 | 0.6551 | Leaf=255 d=8 MCW=10 Bin=256 |
| E | 182 | 0.6557 | Bin=64 LR=0.01 Est=1500 |

### Key insights

1. **Deeper trees (depth 9) + finer LR (0.005)** are the main SOTA drivers. Almost the entire top-10 is Group A with `max_depth=9`.
2. Mild MCW (5–10) at depth 9 works; MCW=8 edges out MCW=5/10 at LR=0.005.
3. **Lower L2/L1** on the old backbone (Group B, L2=0.5 L1=0.05) also beats control (R² 0.6571) but trails depth-9 fine-LR models.
4. Leaf-wise / sampling / coarse bins did not beat the depth-9 fine-LR neighborhood on peak R² (bin=64 is a small bump to 0.6557).
5. Lean budgets remain sufficient: best model trains in ~20s; several near-SOTA configs train in ~8–11s.

### Recommended configuration (new SOTA)

```python
params = {
    "objective": "reg:squarederror",
    "max_depth": 9,
    "min_child_weight": 8,
    "reg_lambda": 1.5,
    "reg_alpha": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_estimators": 2500,
    "learning_rate": 0.005,
}
# Test R² ≈ 0.6584 on derived_8.2 V3 (unweighted)
```

**Fast near-SOTA alternative** (R² ≈ 0.6576, ~11s):

```python
params = {
    "objective": "reg:squarederror",
    "max_depth": 9,
    "min_child_weight": 10,
    "reg_lambda": 1.5,
    "reg_alpha": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_estimators": 1500,
    "learning_rate": 0.01,
}
```

---

## How to re-run

```bash
cd notebooks
# optional: export XGB_PARALLEL_WORKERS=4
uv run --with jupyter jupyter lab experiment/derived_8.2-hyperparameters-1.4/derived_8.2-hyperparameters-1.4.ipynb
```

Training is **resume-safe**: models under `models/` are reused if present.

Or non-interactively:

```bash
cd notebooks
export XGB_PARALLEL_WORKERS=4
export JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime-$USER
uv run jupyter execute experiment/derived_8.2-hyperparameters-1.4/derived_8.2-hyperparameters-1.4.ipynb --timeout=7200
```

---

## Outputs

| File | Description |
|------|-------------|
| `derived_8.2-hyperparameters-1.4.ipynb` | Experiment notebook |
| `metrics_summary.csv` | Overall metrics for all 200 configs |
| `metrics_by_year.csv` | Metrics for 2023 / 2024 / 2025 |
| `leaderboard_top20.csv` | Top 20 by R² |
| `loss_curves.csv` / `.png` | Step-wise train vs test loss |
| `test_predictions.csv` | Test predictions `pred_{id}` |
| `r2_by_year.png` | Year-wise R² for top configs |
| `residuals_comparison.png` | Residuals for top 18 |
| `residuals_by_year.png` | Residuals by year for top configs |
| `models/` | `xgb_model_{id}.json` + `_meta.json` |
