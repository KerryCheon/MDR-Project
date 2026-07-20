# derived_8.2-hyperparameters-1.5 (SOTA Neighborhood Expansion — V3 Global Model)

This directory contains a **500-configuration** XGBoost hyperparameter refinement sweep for the **single global model** on the Washington-only `derived_8.2` dataset with **Feature Set V3** (47 features, unweighted). Goal: beat the **1.4 SOTA** (R² ≈ 0.6584, d=9 LR=0.005 MCW=8 Est=2500).

**Status: completed on H100** (parallel workers = 6). Resume-safe CRC32 cache; run recovered from a mid-sweep session timeout (478/500 models valid; remaining 22 trained on resume).

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
| Parallelism | `XGB_PARALLEL_WORKERS` (default **6**) via thread pool |
| Cache key | `models/xgb_model_{id}_{crc32}.json` — CRC32 of training config payload |

### Lean / extended step budgets

| learning_rate | n_estimators |
|---------------|--------------|
| 0.003 | 3000 |
| 0.004 | 2800 |
| 0.005 | 2500 |
| 0.006 | 2300 |
| 0.007 | 2100 |
| 0.008 | 2000 |
| 0.01  | 1500 |
| 0.012 | 1400 |

### Configuration groups (500 total)

| Group | Count | Focus |
|-------|------:|--------|
| R — References | 6 | MAE/MSE baselines + 1.3-lite control + 1.4 SOTA control + Est checks |
| A — Depth × LR × MCW | 216 | d∈{8,9,10}, fine LR, expanded MCW |
| B — L2 × L1 | 64 | Regularization on **new** (1.4) backbone |
| C — Sampling | 36 | sub × col on new backbone |
| D — Estimator budget | 48 | Est∈{1500…5000} × d×LR |
| E — Probes & champions | 130 | d=11, gamma, bin, leaf hybrids, multi-ingredient |

### Implementation fixes vs 1.4

1. **CRC32 fingerprint in cache filenames** — resume only if `id` **and** config CRC match.
2. **Exception-safe parallel loop** — one worker failure is recorded; other futures finish; successful models stay on disk.
3. **Robust model I/O** — load path detects UBJSON content mislabeled as `.json` (XGBoost 3 default when intermediate paths end in `.tmp`); save uses a temp name ending in `.json` so format is correct.

---

## Results

### New SOTA

| Model | Configuration | $R^2$ | RMSE | MAE | Train (s) |
|-------|---------------|------:|-----:|----:|----------:|
| **241 (NEW SOTA)** | **B MSE L2=0.75 L1=0.03 (d=9 LR=0.005 MCW=8)** | **0.6589** | **0.0615** | **0.0459** | **28.5** |
| 99 | A MSE d=9 LR=0.005 MCW=6 Est=2500 | 0.6589 | 0.0615 | 0.0460 | 29.5 |
| 425 / 475 | E Leaf/L2 variants with MCW=6 | 0.6589 | 0.0615 | 0.0460 | ~30–72 |
| 333 | D d=9 LR=0.005 Est=5000 MCW=8 | 0.6585 | 0.0615 | 0.0460 | 57.4 |
| **3 (control)** | **1.4 SOTA Control** | **0.6584** | 0.0615 | 0.0460 | ~30 |
| **2 (control)** | **1.3-lite SOTA Control** | **0.6551** | 0.0618 | 0.0464 | ~10 |

- **ΔR² vs 1.4 target:** **+0.00055** (0.658904 − 0.658356)
- 1.4 control (id=3) reproduced to double precision (R² = 0.6583556…)

### Year-by-year (best vs controls)

| Year | Model 241 (new) | Model 3 (1.4) | Model 2 (1.3-lite) |
|------|----------------:|--------------:|-------------------:|
| 2023 | **0.6658** | 0.6647 | 0.6581 |
| 2024 | 0.6334 | 0.6334 | **0.6403** |
| 2025 | **0.6507** | 0.6503 | 0.6396 |

Gains over 1.4 are small and uniform on 2023/2025; **2024 remains weaker** than the old 1.3-lite control (same pattern as 1.4).

### Best per group

| Group | Best id | $R^2$ | Configuration |
|-------|--------:|------:|---------------|
| R | 4 | 0.6584 | 1.4 SOTA Est=3000 |
| A | 99 | 0.6589 | d=9 LR=0.005 MCW=6 Est=2500 |
| **B** | **241** | **0.6589** | **L2=0.75 L1=0.03 on 1.4 backbone** |
| C | 319 | 0.6584 | sub=1.0 col=0.8 |
| D | 333 | 0.6585 | Est=5000 d=9 LR=0.005 |
| E | 475 | 0.6589 | L2=1.5 MCW=6 (d9 LR0.005) |

### Key insights

1. **Peak moves only slightly** (+0.00055 R²). The 1.4 neighborhood is already near a local plateau: among 500 runs, only **14** beat published 1.4 SOTA; the 99th percentile is ~0.6586 vs max 0.6589.
2. **Mild L2≈0.75 with L1=0.03** on the d=9 LR=0.005 MCW=8 backbone is the new peak (Group B). **L1=0.2 is toxic** (~0.651–0.652 across L2 values).
3. **MCW=6** is competitive with MCW=8 at the same depth/LR (ids 99 / 475 / 425 tied at 0.6589); MCW≤4 and MCW≥10 drop.
4. **Longer Est (3000–5000)** helps a little (id 333) but not as much as L2/MCW tweaks; cost scales ~linearly. Est=2500 already captures almost all of the Est curve at d=9 LR=0.005 MCW=8.
5. Controls reproduced cleanly — parallel CUDA + CRC32 resume did not scramble reference configs.

### Slice evidence (Group A / C / D / E)

| Axis (others fixed near backbone) | Best | Notes |
|-----------------------------------|------|--------|
| LR at d=9 MCW=8 | **0.005** | Clear peak vs {0.003…0.012} |
| Depth at LR=0.005 MCW=8 | **d=9** | d=8 ≈0.653, d=10 ≈0.656 |
| MCW at d=9 LR=0.005 | **6 ≳ 8 ≳ 5** | Soft U-shape; avoid extremes |
| Est at d=9 LR=0.005 MCW=8 | 1500→0.6577 … 5000→0.6585 | Diminishing after ~2500–3000 |
| Sampling (Group C) | sub=1.0 col=0.8 | Aggressive colsample (≤0.7) hurts |
| d=11 / gamma | max ~0.656 / ~0.657 | Low priority for peak overall |
| Leaf-wise | can *match* MCW=6 peak | Does not clearly beat L2-tuned depthwise |

### Year robustness (important caveat)

Overall-SOTA configs (241, 99, …) still show **2024 R² ≈ 0.633**, while the older 1.3-lite control keeps **2024 R² = 0.6403**. Gains are mostly 2023/2025. Among overall top-20, the best 2024 is still only ~0.636 (e.g. leaf id=400) — better than pure peak overall, still below 1.3-lite on that year.

With hundreds of test-set looks, **+0.00055 is within optimistic selection noise**; treat champions as candidates, not final frozen science claims, until val-based or external-station re-eval.

### Recommended configuration (new SOTA)

```python
params = {
    "objective": "reg:squarederror",
    "max_depth": 9,
    "min_child_weight": 8,
    "reg_lambda": 0.75,
    "reg_alpha": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_estimators": 2500,
    "learning_rate": 0.005,
}
# Test R² ≈ 0.6589 on derived_8.2 V3 (unweighted)
```

**Near-tied alternative** (MCW=6, R² ≈ 0.6589):

```python
params = {
    "objective": "reg:squarederror",
    "max_depth": 9,
    "min_child_weight": 6,
    "reg_lambda": 1.5,
    "reg_alpha": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_estimators": 2500,
    "learning_rate": 0.005,
}
```

---

## Takeaways for the next iteration

Path 1.3-lite → 1.4 → 1.5 moved roughly **0.6551 → 0.6584 → 0.6589**. Another large random HP grid is unlikely to unlock +0.01 on V3 global. Use 1.5 as a **settled booster recipe** and spend budget on denser plateau polish, year robustness, and non-HP work.

### 1. Stop broad grids; switch to a tight plateau probe

Do **not** re-sweep d∈{8,10,11} or LR∈{0.01,0.02} at scale. Fix **d=9, LR=0.005, Est=2500** (or Est∈{2000,2500,3000} only) and dense-grid:

- MCW ∈ {5, 6, 7, 8, 9}
- L2 ∈ {0.5, 0.65, 0.75, 0.9, 1.0, 1.25}
- L1 ∈ {0.01, 0.02, 0.03, 0.04, 0.05}

(~100–150 models) is denser where it matters than another 500-wide search.

### 2. Optimize a secondary objective (2024 / multi-year)

Rank with more than overall test R², e.g.:

- `mean(R²_year)` or `min(R²_year)` across 2023–2025
- or constrained: max overall s.t. 2024 R² ≥ a floor

Report **dual leaderboards** (overall SOTA vs year-robust). Consider leaf / mild-subsample candidates (e.g. id=400 style) as a parallel “2024-aware” track even if overall is slightly lower.

### 3. Cap tree budget by default

Prefer **Est=2500** (maybe 3000). Avoid large Est∈{4000,5000} grids unless probing early stopping; reallocate that compute to structure (features, weights, eval protocol).

### 4. Keep sampling mild

Prefer **sub∈{0.9, 1.0}**, **col∈{0.75, 0.8, 0.85}**. Drop aggressive colsample (≤0.7) and very low subsample (0.75) from future grids.

### 5. Deprioritize depth-11, gamma, large leaf grids

At most a **small confirmation set** (e.g. lossguide leaves=127/255 with d=9 MCW=6 and best L2/L1). Not another full leaf×bin factorial.

### 6. Selection protocol / scientific hygiene

With 500 test-set looks, freeze a small **candidate set** (top overall + top multi-year) and:

- prefer **val-based ranking** when a non-test holdout is available, **or**
- re-eval champions once under a **frozen** protocol (LOSO / ECE stations)

Always publish year metrics alongside overall R² when claiming SOTA.

### 7. Engineering practices to keep

- CRC32-keyed caches (partial resume worked after session timeout)
- Per-future exception isolation at ≥ hundreds of models
- Save paths that **end in `.json`** (not `.json.tmp`) so XGBoost 3 writes real JSON; keep content-aware load for legacy UBJSON-as-`.json` files
- Next HP iteration can stay **≤150–200** models at 6 workers

### 8. Higher-upside bets than more XGB HP spam

Once the plateau fine-tune is done (or instead of it):

- **2024 residual / regime analysis** (why the weak year resists peak overall models)
- **Feature-set** comparison under **frozen** champion params (V3 vs FS-2.0 winners)
- Light **temporal / station sample weights** for weak years
- Confirm champions on **held-out stations** (ECE sensors)

### Suggested shape of a possible 1.6

| Track | Size | Goal |
|-------|-----:|------|
| A. Plateau fine-tune | ~100–150 | MCW×L2×L1 at d=9 LR=0.005 Est=2500 |
| B. 2024-robust ranking | same models | dual leaderboards: overall vs mean/min year |
| C. Tiny Est/sub check | ~20 | Est∈{2000,2500,3000}, sub∈{0.9,1.0} at best L2/L1 |
| D. Freeze + re-eval | 3–5 champs | val-based pick or external station eval |

### Bottom line

Booster **structure is largely settled**: d=9, LR=0.005, Est~2500, mild MCW (6–8), mild L1/L2 (L2~0.75, L1~0.03).  
Next work should **(a)** polish L2/L1/MCW on that plateau, **(b)** explicitly target **year robustness (esp. 2024)**, and **(c)** move remaining budget to **features / evaluation protocol**, not larger random grids.

---

## How to re-run

```bash
cd notebooks
export XGB_PARALLEL_WORKERS=6
export JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime-$USER
uv run jupyter execute experiment/derived_8.2-hyperparameters-1.5/derived_8.2-hyperparameters-1.5.ipynb --timeout=10800
```

Training is **resume-safe** via CRC32-keyed caches under `models/`. Existing UBJSON-as-`.json` files from the first partial run load correctly via content detection.

---

## Outputs

| File | Description |
|------|-------------|
| `derived_8.2-hyperparameters-1.5.ipynb` | Experiment notebook |
| `metrics_summary.csv` | Overall metrics for all 500 configs |
| `metrics_by_year.csv` | Metrics for 2023 / 2024 / 2025 |
| `leaderboard_top20.csv` | Top 20 by R² |
| `loss_curves.csv` / `.png` | Step-wise train vs test loss |
| `test_predictions.csv` | Test predictions `pred_{id}` |
| `r2_by_year.png` | Year-wise R² for top configs |
| `residuals_comparison.png` | Residuals for top 18 |
| `residuals_by_year.png` | Residuals by year for top configs |
| `models/` | `xgb_model_{id}_{crc32}.json` + `_meta.json` |
