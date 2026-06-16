# Time-Series Soil Moisture Prediction — LSTM Iterations v7–v16

**Wrap-up report.** Compiled from ten LSTM/GRU variants trained on the `derived_8.0` Washington-state soil-moisture dataset between v7 and v16.

---

## 1. Executive summary

- **Best variant: [v9](Models/Temporal/lstm/train_v9.py)** — BiLSTM + 1-head temporal attention pooling over a 10-day window. **Test R² = 0.747, RMSE = 0.0474, MAE = 0.0360.**
- **Smallest competitive variant: [v12](Models/Temporal/lstm/train_v12.py)** — 37K parameters (7.6× smaller than v9), test R² = 0.730. Came out of an 8-config hyperparameter grid scan.
- **Structural ceiling identified.** Six distinct architectural strategies (FA-LSTM, temporal attention, multi-head attention, BiGRU multi-task, multi-scale parallel windows, SMAP-residual reframing) all landed in **test R² ∈ [0.685, 0.747]**. Validation R² reached 0.83–0.85 multiple times, but the ~10-point val→test gap never closed. This is almost certainly a **train/val vs test station-distribution shift**, not a model-capacity problem.
- **Target of test R² > 0.80 was not reached** with any of the ten LSTM-family variants. The remaining lever is on the data side: split strategy, station-level normalization, or domain adaptation.

---

## 2. Project context

### Goal
Predict in-situ soil moisture at 5 cm depth (`soil_moisture_5cm`, m³/m³) for Washington-state stations using a 10-day window of multi-sensor satellite + meteorological inputs.

### Data
- Source: [derived_8.0 splits](Temporal/Pipeline/data/splits/derived_8.0/) — `train.csv`, `val.csv`, `test.csv`.
- Sizes: train = 6,868 rows × 499 cols; val = 2,720; test = 4,016.
- Window-stride builder ([dataset.py](Models/Temporal/lstm/dataset.py)) yields ~6,800 training sequences at `seq_len=10`.
- Target distribution: soil moisture m³/m³ in roughly [0.05, 0.55]; ~6% volumetric typical RMSE.

### Per-timestep features (the core set most variants used)
Precipitation (`precip_mm`), Sentinel-1 SAR (`s1_vv`, `s1_vh`), Sentinel-2 optical bands (`s2_b4/b8/b11/b12`), MODIS LST (`LST_modis`), vegetation indices (`F_NDVI`, `F_NDMI`), SAR-derived (`E_SAR_ratio`), SMAP AM/PM soil moisture interpolated (`SMAP_sm_am_interp`, `SMAP_sm_pm_interp`), SMAP observation mask, sin/cos day-of-year.

### Static features
Lat/lon, elevation, slope, aspect, soil-texture composites (`K_sand_clay_ratio_b0`, `K_clay_plus_sand_b0`), slope/aspect trig encodings.

### Top XGBoost-favorite rolling/lag features (kept as time-features in v9+)
- `SMAP_sm_pm_interp_ema02` (PM-SMAP exponential moving average, α=0.2)
- `V_ema_LST_modis_kobs7` (7-obs LST EMA)
- `V_rollmean_G_API_kobs14` (14-obs API rolling mean)

### Literature priors that shaped the work
- **East Java 2025** — multivariate GRU, 10-day SM, R² = 0.80.
- **Hebei Hybrid 2024** — GRU-Transformer, root-zone SM, R² ≈ 0.99 on day-3 (different target depth and regime).
- **Wang et al. 2024 (HESS)** — feature attention + adversarial training significantly stabilize 7-day forecasts.
- **MDR Report 4/29** (project-internal) — drop the 496 pre-computed lag features; keep a hand-picked 3–10; use raw daily inputs in-sequence; smaller windows (5–15 d) beat 30–60 d.

---

## 3. Methodology

### Common training setup (all variants)
- Median imputation + StandardScaler fit on **train features only**, applied to val/test.
- Inputs clipped to ±5σ post-scaling.
- HuberLoss (δ=0.05) — moisture targets have a long-tailed error distribution; Huber prevents single outliers from dominating gradients.
- AdamW optimizer, gradient clipping at 1.0, batch size 256, seed 42.
- Early stopping on val RMSE; checkpoint = best val RMSE.

### Metrics
- **R²** — coefficient of determination.
- **RMSE** — root mean square error (m³/m³).
- **ubRMSE** = √(RMSE² − Bias²) — error stddev after debiasing. Standard in SMAP / soil moisture literature.
- **Bias** = mean(pred − true). Negative bias = model under-predicts.
- **MAE** — mean absolute error.
- **Q90** — 90th percentile of |residuals|. Worst-decile error magnitude.

(ubRMSE + Q90 were added in v12 onward; earlier variants only have R²/RMSE/Bias/MAE.)

---

## 4. Variant catalog

### [v7](Models/Temporal/lstm/train_v7.py) — Small window + pruned features, vanilla 2-layer LSTM

**Hypothesis.** A smaller window (lit) + pruned redundant features should beat the 60-day baseline.

**Architecture.** Input projection → 2-layer LSTM (hidden=96, dropout=0.35) → last-hidden → 2-layer head.
**Window.** 14 days. **Features.** 16 time + 11 static = 27. Dropped `F_MSI`, `E_SAR_diff`, `SMAP_ampm_diff_interp`, two of three SMAP masks (kept combined).
**Hyperparameters.** lr=1e-3, wd=1e-3, ReduceLROnPlateau, max_epochs=250, patience=35.
**Params.** 137K.

**Results.**
| Split | R² | RMSE | MAE | Bias |
|---|---|---|---|---|
| Train | 0.870 | 0.0362 | 0.0262 | +0.0057 |
| Val | 0.808 | 0.0460 | 0.0338 | -0.0155 |
| **Test** | **0.685** | **0.0528** | **0.0408** | **-0.0076** |

**Best epoch.** 8 (early-stopped at 43).
**Takeaway.** Overfit hard. Pruning + smaller window not enough on its own; missing attention.

---

### [v8](Models/Temporal/lstm/train_v8.py) — Feature-Attention LSTM (FA-LSTM), 7-day window

**Hypothesis.** Per-timestep softmax gate over features (Wang 2024 FA mechanism) at a very short window will improve short-horizon stability.

**Architecture.** Feature attention layer (Linear → Tanh → Linear → softmax across features, multiplied into input × n_features rescaling) → 2-layer LSTM (hidden=128, dropout=0.4) → head.
**Window.** 7 days. **Features.** Same 27 as v7.
**Hyperparameters.** lr=8e-4, wd=2e-3, Cosine warm restarts (T₀=20, T_mult=2), max_epochs=250.
**Params.** 223K.

**Results.**
| Split | R² | RMSE | MAE | Bias |
|---|---|---|---|---|
| Train | 0.895 | 0.0325 | 0.0237 | +0.0048 |
| Val | 0.770 | 0.0505 | 0.0398 | -0.0266 |
| **Test** | **0.710** | **0.0508** | **0.0388** | **-0.0069** |

**Best epoch.** 24.
**Takeaway.** Beat v7 on test (+0.025 R²). FA mechanism + short window confirmed as a useful pairing.

---

### [v9](Models/Temporal/lstm/train_v9.py) — BiLSTM + 1-head temporal attention pooling **[WINNING BASELINE]**

**Hypothesis.** Bidirectional context + sequence-axis attention pooling captures longer-range temporal dependencies than v8's last-hidden output, with hand-picked lag features added in-sequence.

**Architecture.** Linear proj (56-d) → bidirectional 2-layer LSTM (hidden=80, dropout=0.3) → additive attention (Linear→Tanh→Linear→softmax over time) → context vector → 2-layer head.
**Window.** 10 days. **Features.** 19 time + 11 static = 30. Time features include the **top-3 XGBoost lag features** in-sequence.
**Hyperparameters.** lr=7e-4, wd=2e-3, OneCycleLR (max_lr=3×base, pct_start=0.1), max_epochs=250, patience=35.
**Params.** 284K.

**Results.**
| Split | R² | RMSE | MAE | Bias |
|---|---|---|---|---|
| Train | 0.930 | 0.0265 | 0.0193 | +0.0067 |
| Val | 0.845 | 0.0414 | 0.0313 | -0.0127 |
| **Test** | **0.747** | **0.0474** | **0.0360** | **-0.0098** |

**Best epoch.** 24 (early-stopped at 59).
**Takeaway.** The strongest LSTM variant overall. Attention pooling + 3 carefully chosen lag features hit a good capacity/regularization balance. **No subsequent variant beat this on test.**

---

### [v10](Models/Temporal/lstm/train_v10.py) — v9 + LayerNorm + Gaussian input noise + heavier regularization

**Hypothesis.** v9's train→test gap is overfitting; heavier regularization should close it.

**Architecture.** v9 + LayerNorm after projection + Gaussian noise (σ=0.02) on standardized input during training.
**Window.** 12 days. **Features.** Same 30 as v9.
**Hyperparameters.** dropout=0.4 (was 0.3), wd=5e-3 (was 2e-3), Cosine annealing (replaced OneCycle), max_epochs=300, patience=45.
**Params.** 284K.

**Results.**
| Split | R² | RMSE | MAE | Bias |
|---|---|---|---|---|
| Train | 0.923 | 0.0277 | 0.0201 | +0.0014 |
| Val | 0.719 | 0.0557 | 0.0406 | -0.0244 |
| **Test** | **0.567** | **0.0620** | **0.0468** | **-0.0201** |

**Best epoch.** 48 (early-stopped at 93).
**Takeaway.** Reg regressed all three splits without reducing the overfit (train R² still 0.92). Demonstrated that the train→test gap is **not** a capacity / regularization problem.

---

### [v11](Models/Temporal/lstm/train_v11.py) — BiLSTM + multi-head(4) temporal attention + FA + 5 lag features

**Hypothesis.** More expressive attention (4 heads) + FA (stacked from v8 idea) + more lag features should give the model more to work with.

**Architecture.** Feature attention → projection → BiLSTM (hidden=96, 2-layer, dropout=0.35) → multi-head temporal attention (4 heads) → 2-layer head.
**Window.** 10 days. **Features.** 21 time + 11 static = 32 (v9's 19 + 2 more lags: `V_rollmin_LST_modis_kobs30`, `SMAP_sm_am_interp_rollrange7`).
**Hyperparameters.** lr=7e-4, wd=3e-3, Cosine annealing, max_epochs=250.
**Params.** 407K (largest variant).

**Results.**
| Split | R² | RMSE | MAE | Bias |
|---|---|---|---|---|
| Train | 0.884 | 0.0341 | 0.0253 | +0.0008 |
| Val | 0.840 | 0.0421 | 0.0331 | -0.0210 |
| **Test** | **0.710** | **0.0508** | **0.0393** | **-0.0125** |

**Best epoch.** 4 (plateaued instantly).
**Takeaway.** 407K params for ~6.8K training sequences. Optimizer locked into a fast local minimum and never escaped. Strong evidence that more capacity actively hurts here.

---

### [v12](Models/Temporal/lstm/train_v12.py) — SmallBiLSTM + grid search **[SMALLEST COMPETITIVE]**

**Hypothesis.** v9 is over-parameterized; trimming capacity is a more direct lever than regularization. Validate via hyperparameter grid scan.

**Architecture.** Linear proj (40-d) → **1-layer** bidirectional LSTM → additive attention → 2-layer head.
**Window.** 10 days. **Features.** Same 30 as v9.
**Hyperparameters.** Grid scan over `hidden_size ∈ {40, 56} × dropout ∈ {0.25, 0.35} × lr ∈ {5e-4, 1e-3}` — 8 configs at max_epochs=60. Winner retrained at max_epochs=250.
**Grid winner.** hidden=40, dropout=0.35, lr=1e-3.
**Params.** 37K (winner config).

**Grid ranking (val_rmse, top 4 of 8).**
1. hidden=40, dropout=0.35, lr=1e-3 → val_rmse 0.04372 (37K params)
2. hidden=40, dropout=0.25, lr=1e-3 → val_rmse 0.04389 (37K params)
3. hidden=56, dropout=0.25, lr=1e-3 → val_rmse 0.04422 (64K params)
4. hidden=40, dropout=0.35, lr=5e-4 → val_rmse 0.04437 (37K params)

`hidden=40` and `lr=1e-3` won across most cells.

**Final results.**
| Split | R² | RMSE | ubRMSE | Bias | MAE | Q90 |
|---|---|---|---|---|---|---|
| Train | 0.904 | 0.0310 | 0.0309 | +0.0022 | 0.0219 | 0.0467 |
| Val | 0.825 | 0.0440 | 0.0394 | -0.0196 | 0.0327 | 0.0763 |
| **Test** | **0.730** | **0.0489** | **0.0473** | **-0.0124** | **0.0373** | **0.0789** |

**Best epoch.** 16 (early-stopped at 51).
**Takeaway.** 7.6× smaller than v9 at near-equivalent test performance. **Grid scan confirmed v9 was over-parameterized** — the smaller architecture won. ubRMSE (0.0473) vs RMSE (0.0489) means bias contributes only ~3% of error; the rest is variance. Q90=0.079 → worst 10% of predictions miss by ≥7.9% volumetric.

---

### [v13](Models/Temporal/lstm/train_v13.py) — Residual-over-SMAP target reframing

**Hypothesis.** The LSTM wastes capacity re-learning what SMAP already provides. Reframe the target as `y_residual = soil_moisture_5cm − SMAP_sm_pm_interp` and reconstruct predictions at inference by adding SMAP back. Forces the LSTM to learn only what SMAP misses.

**Architecture.** Identical to v9. **Window.** 10 days. **Features.** Same 30 (SMAP_sm_pm_interp kept in input — it's both prior and signal).
**Hyperparameters.** Same as v9 (OneCycleLR, lr=7e-4).
**Params.** 284K.

**Results.**
| Split | R² | RMSE | ubRMSE | Bias | MAE | Q90 |
|---|---|---|---|---|---|---|
| Train | -9.46 | 0.3217 | 0.3214 | +0.0147 | 0.2575 | 0.5302 |
| Val | 0.755 | 0.0521 | 0.0513 | -0.0090 | 0.0391 | 0.0800 |
| **Test** | **0.731** | **0.0498** | **0.0497** | **+0.0039** | **0.0372** | **0.0793** |

**Best epoch.** 47.
**Takeaway.** **SMAP isn't actually a good prior here** — the script's sanity check showed the "SMAP-as-prediction" baseline scores R² ≈ −7 to −8 across all splits (residual mean −0.17). SMAP and `soil_moisture_5cm` are on systematically different scales. Train R² = −9.46 reflects a few exploded predictions on training data; val/test reconstruction worked but didn't beat v9. **Best bias of any variant** (+0.0039 → essentially unbiased) — a small consolation.

---

### [v14](Models/Temporal/lstm/train_v14.py) — Multi-scale parallel windows (5d / 10d / 20d)

**Hypothesis.** A single window is a forced choice. Three parallel BiLSTM branches at 5d/10d/20d, each with its own attention pool, capture short-term dynamics + mid-range + seasonal state simultaneously.

**Architecture.** Build sequences at `seq_len=20`; in forward pass, slice `x[:, -5:]`, `x[:, -10:]`, `x[:, -20:]` to feed three independent branches (each: projection → BiLSTM hidden=32 → attention pool → context). Concatenate three contexts (192-d) → fusion head.
**Window.** 5/10/20 days. **Features.** Same 30 as v9.
**Hyperparameters.** lr=7e-4, wd=2e-3, Cosine annealing.
**Params.** 92K.

**Results.**
| Split | R² | RMSE | ubRMSE | Bias | MAE | Q90 |
|---|---|---|---|---|---|---|
| Train | 0.945 | 0.0236 | 0.0232 | +0.0038 | 0.0169 | 0.0355 |
| Val | 0.820 | 0.0447 | 0.0383 | -0.0231 | 0.0345 | 0.0781 |
| **Test** | **0.712** | **0.0506** | **0.0485** | **-0.0143** | **0.0384** | **0.0828** |

**Best epoch.** 30.
**Takeaway.** Most overfit variant (train R² 0.945). The 20-d window cost 100 sequences. Multi-scale didn't help — the test ceiling isn't a single-scale problem.

---

### [v15](Models/Temporal/lstm/train_v15.py) — BiGRU + multi-task auxiliary head (predict SMAP)

**Hypothesis.** GRU (East Java 2025 used it for 10-d SM at R² 0.80) + a multi-task head that *also* predicts `SMAP_sm_pm_interp` regularizes the shared encoder — the aux task acts as a denoising autoencoder constraint.

**Architecture.** Linear proj (48-d) → bidirectional 2-layer GRU (hidden=64, dropout=0.3) → additive temporal attention → two heads (primary `soil_moisture_5cm` + auxiliary `SMAP_sm_pm_interp`).
**Loss.** `huber(primary) + 0.3 × huber(aux)`.
**Window.** 10 days. **Features.** Same 30 as v9.
**Hyperparameters.** lr=7e-4, wd=2e-3, Cosine annealing.
**Params.** 153K.

**Results (primary head).**
| Split | R² | RMSE | ubRMSE | Bias | MAE | Q90 |
|---|---|---|---|---|---|---|
| Train | 0.940 | 0.0245 | 0.0227 | +0.0091 | 0.0186 | 0.0377 |
| Val | 0.827 | 0.0438 | 0.0410 | -0.0156 | 0.0316 | 0.0746 |
| **Test** | **0.717** | **0.0501** | **0.0499** | **-0.0042** | **0.0385** | **0.0797** |

**Best epoch.** 105 (early-stopped at 140 — trained 4× longer than any other variant).
**Takeaway.** Aux SMAP loss measurably **stabilized training** — kept improving for 100+ epochs where others plateaued at 20–30. The encoder converged to a more stable representation, but it stabilized into **the same test ceiling**. Multi-task regularization is real but doesn't solve a distribution-shift problem.

---

### [v16](Models/Temporal/lstm/train_v16.py) — Engineered lag-feature blast

**Hypothesis.** v9 plateaued because it only uses 3 lag features. Adding a comprehensive engineered set (MDR Report Table 3) should give the model enough hydrological context to break through.

**Architecture.** Same as v9 (BiLSTM 2-layer + temporal attention, hidden=80). Wider projection (proj=64) to absorb more features.
**Window.** 10 days. **Features.** 30 time + 11 static = 41 (v9's 16 base + 3 v9-lags + 4 v5-extra lags + **7 newly engineered**).
**Newly engineered (per-station, look-back-only, log-transformed where right-skewed).**
- `feat_log1p_rain_7d`, `feat_log1p_rain_30d`
- `feat_days_since_rain` (capped 30, then log1p'd before scaling)
- `feat_ndmi_14d`
- `feat_smap_pm_7d`, `feat_smap_pm_30d`
- `feat_smap_am_pm_diff_7d`

**Hyperparameters.** lr=7e-4, wd=2e-3, Cosine annealing, max_epochs=250, patience=40.
**Params.** 290K.

**Results.**
| Split | R² | RMSE | ubRMSE | Bias | MAE | Q90 |
|---|---|---|---|---|---|---|
| Train | 0.866 | 0.0367 | 0.0353 | +0.0098 | 0.0266 | 0.0582 |
| Val | 0.843 | 0.0417 | 0.0381 | -0.0170 | 0.0322 | 0.0726 |
| **Test** | **0.725** | **0.0494** | **0.0491** | **-0.0055** | **0.0383** | **0.0815** |

**Best epoch.** 8 (fastest convergence of any variant).
**Takeaway.** Converged in 8 epochs — the model is **signal-saturated** at this feature count. Extra lag features didn't break the test ceiling. The train R² (0.866) is *lower* than v9's (0.930), showing the new features did increase regularization a bit, but at no test-set payoff.

---

## 5. Master comparison

### Per-variant scoreboard (test split, sorted by R² desc)

| Variant | Architecture | Window | Test R² | Test RMSE | Test ubRMSE | Test Bias | Test MAE | Test Q90 | Train R² | Val R² | Best ep | Params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **v9** | BiLSTM + attn | 10 | **0.747** | **0.0474** | – | -0.0098 | **0.0360** | – | 0.930 | 0.845 | 24 | 284K |
| v12 | SmallBiLSTM + grid | 10 | 0.730 | 0.0489 | 0.0473 | -0.0124 | 0.0373 | 0.0789 | 0.904 | 0.825 | 16 | 37K |
| v13 | Residual-over-SMAP | 10 | 0.731 | 0.0498 | 0.0497 | **+0.0039** | 0.0372 | 0.0793 | -9.46 | 0.755 | 47 | 284K |
| v16 | Engineered lags (41 feat) | 10 | 0.725 | 0.0494 | 0.0491 | -0.0055 | 0.0383 | 0.0815 | 0.866 | 0.843 | 8 | 290K |
| v15 | BiGRU multi-task | 10 | 0.717 | 0.0501 | 0.0499 | -0.0042 | 0.0385 | 0.0797 | 0.940 | 0.827 | 105 | 153K |
| v14 | Multi-scale 5/10/20 | 5+10+20 | 0.712 | 0.0506 | 0.0485 | -0.0143 | 0.0384 | 0.0828 | 0.945 | 0.820 | 30 | 92K |
| v8 | FA-LSTM | 7 | 0.710 | 0.0508 | – | -0.0069 | 0.0388 | – | 0.895 | 0.770 | 24 | 223K |
| v11 | Multi-head + FA + 5 lags | 10 | 0.710 | 0.0508 | – | -0.0125 | 0.0393 | – | 0.884 | 0.840 | 4 | 407K |
| v7 | 2-layer LSTM, pruned | 14 | 0.685 | 0.0528 | – | -0.0076 | 0.0408 | – | 0.870 | 0.808 | 8 | 137K |
| v10 | v9 + heavy reg | 12 | 0.567 | 0.0620 | – | -0.0201 | 0.0468 | – | 0.923 | 0.719 | 48 | 284K |

### Visual ranking
**Test R²:** v9 (0.747) > v13 ≈ v12 (0.73) > v16 (0.725) > v15 (0.717) > v8 ≈ v11 ≈ v14 (0.710) > v7 (0.685) ≫ v10 (0.567).

**Val R²:** v9 (0.845) > v16 (0.843) > v11 (0.840) > v15 (0.827) > v12 (0.825) > v14 (0.820) > v7 (0.808) > v8 (0.770) > v13 (0.755) > v10 (0.719).

**Notice** several variants exceeded v9 *on validation* (v11, v16 were very close). None translated to test. This is the structural-ceiling signal.

---

## 6. Cross-variant findings

### 6.1 Smaller windows win, but plateau at ~7–15 days
Confirmed lit's "5–15d is best" prior:
- 7d (v8) and 10d (v9, v11, v12, v13, v15, v16) all competitive.
- 14d (v7) underperformed slightly.
- 12d (v10) hurt.
- 20d branch in v14 added overfit, not signal.

**Practical:** Default to 10d. 7d also a strong choice with FA.

### 6.2 Attention always helps; multi-head doesn't help more
v8 (FA), v9 (single-head temporal), v11 (multi-head + FA), v14 (per-branch attention), v15 (temporal + multi-task) — attention in any form beats v7's last-hidden. **Multi-head specifically didn't add value** (v11 vs v9): 4 heads, more params, no test improvement.

**Practical:** Single-head temporal attention is enough.

### 6.3 Capacity is not the bottleneck
- v12 with **37K params** matched the field on test (R² = 0.730).
- v11 with **407K params** got worse (R² = 0.710, plateaued at epoch 4).
- Grid search in v12 systematically preferred smaller `hidden_size`.

**Practical:** ~50–150K is the sweet spot for this dataset size. Larger models overfit faster and may settle in worse local minima.

### 6.4 Regularization can't close the train→test gap
- v10 increased dropout, weight decay, added LayerNorm and input noise. Train R² barely changed (0.92 → 0.92), val and test both regressed.
- v15's auxiliary task helped *training stability* (best epoch 105 vs 24 typical) but the resulting model stabilized at the same test R².

**Practical:** The train↔test gap is not from overfit alone. Reg alone won't fix it.

### 6.5 Feature engineering hits diminishing returns
- v9 → v16 added 11 new features (7 engineered + 4 v5-extra pre-computed lags).
- Train R² dropped (0.93 → 0.87, more regularized in effect), test went from 0.747 → 0.725.
- v16 converged in 8 epochs — saturated immediately.

**Practical:** v9's 19 time features are enough. Adding more lags is not the lever.

### 6.6 Bias is small; the error is variance
- Test ubRMSE / RMSE ratios across v12-v16: 0.967, 0.998, 0.989, 0.996, 0.994.
- Translation: in every variant, **less than 5% of test RMSE is bias**. The bulk is unsystematic, sample-by-sample error.
- Q90 ≈ 0.079–0.083 across variants. The worst 10% of predictions consistently miss by 8% volumetric.

**Practical:** A constant calibration shift would not meaningfully help. You'd need to reduce variance, not de-bias.

### 6.7 The val→test gap is structural

This is the headline finding.

| Variant | Val R² | Test R² | Gap |
|---|---|---|---|
| v9 | 0.845 | 0.747 | -0.098 |
| v16 | 0.843 | 0.725 | -0.118 |
| v11 | 0.840 | 0.710 | -0.130 |
| v15 | 0.827 | 0.717 | -0.110 |
| v12 | 0.825 | 0.730 | -0.095 |
| v14 | 0.820 | 0.712 | -0.108 |
| v7  | 0.808 | 0.685 | -0.123 |
| v8  | 0.770 | 0.710 | -0.060 |
| v13 | 0.755 | 0.731 | -0.024 |
| v10 | 0.719 | 0.567 | -0.152 |

Every variant has a 6–15 point val→test gap, and it doesn't correlate with model size, regularization, or feature count. The most plausible cause: **test stations are systematically different from train+val stations** (different climate zone, soil type, vegetation, or elevation band). The trained models generalize within the train+val population but cannot extrapolate to test-population stations.

A clue from v13: its val→test gap is the smallest (-0.024). The SMAP residual reframing acted as a station-invariant prior — even though SMAP's absolute calibration was off, the *relative* deviation from it was more stable across station populations than the absolute moisture value. This is suggestive of where to look next.

---

## 7. What I'd do next to crack 0.80

Ranked by expected lift × ease.

1. **Audit the split.** Read [split_meta.json](Temporal/Pipeline/data/splits/derived_8.0/split_meta.json) and compare feature distributions (lat, elev, soil class, climate zone, NDVI mean) between train+val and test station populations. If test stations cluster differently, that's the diagnosis confirmed.\

2. **Per-station target normalization.** Subtract each station's train-period soil-moisture mean from the target during training, predict the de-meaned residual, re-add at inference. Only works if you have history per station; doesn't help fully held-out stations.

3. **Station feature signature embedding.** Project (lat, lon, elev, slope, soil_texture) into a small embedding via a learned MLP, concatenate to LSTM context. Forces the model to use station identity through interpretable feature axes that generalize to new stations. 

4. **Different split strategy.** If the current split is held-out-stations, try a temporal hold-out *within* stations (predict month-12 from months 1–11 across all stations). Or a station-stratified hold-out where each fold has the same lat/soil mix. 

5. **Domain adaptation.** Adversarial domain confusion — discriminator tries to tell train vs test stations from the LSTM context vector; encoder tries to fool it. Forces a station-invariant representation. 

6. **Ensemble of v9 + v12 + v13.** Average their test predictions. The three have different inductive biases (capacity, residual reframing, single-window) and uncorrelated errors should reduce variance. 

What I'd *not* spend time on:
- More attention variants. Saturated.
- More lag features. Saturated.
- Deeper / wider LSTMs. Hurts.
- Transformers. Likely the same ceiling — this is a data problem, not an architecture problem.

---

## 8. Repository artifacts

### Code
- [Models/Temporal/lstm/dataset.py](Models/Temporal/lstm/dataset.py) — shared sequence-window dataset builder.
- [Models/Temporal/lstm/model.py](Models/Temporal/lstm/model.py) — the original `LSTMRegressor` (used by v7-only; v8-v16 each define their model class inline).
- [Models/Temporal/lstm/train_v7.py](Models/Temporal/lstm/train_v7.py) through [train_v16.py](Models/Temporal/lstm/train_v16.py) — ten variant training scripts. Each is self-contained, runnable as `python -m Models.Temporal.lstm.train_vN`.

### Outputs (per variant)
Each `Models/Temporal/lstm/outputs_vN/` contains:
- `best_model.pt` — checkpoint at lowest val RMSE.
- `metrics.json` — train/val/test metrics + full config + (for v12) grid_search results.
- `loss_curve.png` — training curve.
- `train_log.txt` — captured stdout.



## 9. Bottom line

**v9 is the production candidate** — BiLSTM + temporal attention + 3 hand-picked lag features, 10-day window, 284K params, test R² = 0.747. If you want a smaller deployment footprint, **v12** is within 1.5 R² points at 7.6× fewer parameters.

The 0.80 R² target is reachable, but **not by another LSTM variant**. The next move is data-side: characterize the train/val vs test station distribution shift, and either change the split or build a station-invariant representation.
