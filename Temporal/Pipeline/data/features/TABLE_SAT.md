# TABLE_SAT: Final Feature Set (Feb 19)

**Last updated:** Fri Feb 20, 2026
**File:** `Models/Temporal/final_set_feb_19.json`

This doc is now final-set-first and tied to the Feb 19 selection artifacts. It replaces the old conceptual keep/bench tables with the actual selected feature list used in modeling.

## Final Set Summary

- Final selected features: **108**
- Ordering source: `Models/Temporal/final_set_feb_19.json`
- Cross-check source: `Temporal/Pipeline/data/features/selection/prune_runs/run01/stage_fine/fine_best_features.csv`

### Family counts in final set

| Family      | Count | Notes                                  |
| ----------- | ----: | -------------------------------------- |
| A           |    39 | Change dynamics (diff, grad, pct)      |
| B (`V_`)    |    33 | Volatility and smoothing windows       |
| C           |     7 | Memory and lag structure               |
| D           |     6 | Seasonality and frequency features     |
| G           |     5 | Meteorological forcing and rain memory |
| S (`SMAP_`) |    18 | SMAP-derived temporal signals          |

## How This Set Was Extracted

1. Start from the train split feature pool and keep only configured families from `config_used.yaml` (`C, B, A, D, G, S`). That gives **408** candidate features. `H_` family was dropped due to instability and missing values/`inf` (div by zero for edge cases)
2. Run permutation importance using XGBoost (`seeds=[1,2,3,4,5]`, `repeats=5`, `scoring=r2`).
3. Coarse pruning drops low-importance features in chunks and keeps the best validation $R^2$ checkpoint. Best coarse set lands at **180** features.
4. Fine pruning repeats the same idea with tighter guardrails (`allow_r2_drop`, patience logic, smaller drop fraction) and keeps the best validation checkpoint rather than forcing a hard target when score degrades.

## Final Selected Features (108)

|   # | Feature Code                      | Family |
| --: | --------------------------------- | :----: |
|   1 | `SMAP_sm_pm_interp_ema02`         |   S    |
|   2 | `V_rollmin_LST_modis_kobs30`      |   B    |
|   3 | `D_sin_DOY`                       |   D    |
|   4 | `G_rain_sum_3d`                   |   G    |
|   5 | `V_ema_G_API_kobs7`               |   B    |
|   6 | `V_rollmin_G_API_kobs30`          |   B    |
|   7 | `G_rain_sum_7d`                   |   G    |
|   8 | `C_lag_LST_modis_kobs30`          |   C    |
|   9 | `C_lag_G_API_kobs1`               |   C    |
|  10 | `V_ema_G_API_kobs14`              |   B    |
|  11 | `V_rollmean_G_API_kobs14`         |   B    |
|  12 | `G_API`                           |   G    |
|  13 | `A_pct_G_API`                     |   A    |
|  14 | `V_rollcv_G_API_kobs30`           |   B    |
|  15 | `G_DSLR`                          |   G    |
|  16 | `SMAP_ampm_diff_interp`           |   S    |
|  17 | `V_rollmax_G_API_kobs30`          |   B    |
|  18 | `V_rollmin_G_API_kobs7`           |   B    |
|  19 | `V_ema_G_API_kobs30`              |   B    |
|  20 | `V_rollmean_s2_b11_kobs7`         |   B    |
|  21 | `V_ema_LST_modis_kobs7`           |   B    |
|  22 | `C_smm_G_API_alpha0.85_n5`        |   C    |
|  23 | `C_lag_G_API_kobs5`               |   C    |
|  24 | `V_rollmean_G_API_kobs7`          |   B    |
|  25 | `C_lag_s2_b11_kobs30`             |   C    |
|  26 | `D_z_LST_modis`                   |   D    |
|  27 | `A_d_G_API_kobs1`                 |   A    |
|  28 | `V_rollcv_LST_modis_kobs30`       |   B    |
|  29 | `V_rollcv_G_API_kobs7`            |   B    |
|  30 | `V_rollstd_LST_modis_kobs30`      |   B    |
|  31 | `A_d_E_SAR_diff_kobs14`           |   A    |
|  32 | `C_lag_G_API_kobs6`               |   C    |
|  33 | `V_rollrng_F_NDMI_kobs7`          |   B    |
|  34 | `V_rollcv_G_API_kobs14`           |   B    |
|  35 | `C_lag_LST_modis_kobs6`           |   C    |
|  36 | `A_d_E_SAR_diff_kobs30`           |   A    |
|  37 | `A_d_LST_modis_kobs14`            |   A    |
|  38 | `SMAP_sm_am_interp_rollrange7`    |   S    |
|  39 | `V_rollstd_LST_modis_kobs14`      |   B    |
|  40 | `D_fft_ent_E_SAR_ratio_kobs30`    |   D    |
|  41 | `A_d_E_SAR_diff_kobs5`            |   A    |
|  42 | `SMAP_sm_pm_interp_rollrange7`    |   S    |
|  43 | `V_rollstd_F_NDMI_kobs7`          |   B    |
|  44 | `V_rollstd_E_SAR_ratio_kobs7`     |   B    |
|  45 | `V_rollrng_E_SAR_diff_kobs7`      |   B    |
|  46 | `V_rollstd_s2_b12_kobs7`          |   B    |
|  47 | `A_grad_E_SAR_diff_kobs14`        |   A    |
|  48 | `D_fft_dom_LST_modis_kobs30`      |   D    |
|  49 | `V_rollcv_s2_b12_kobs7`           |   B    |
|  50 | `A_d_E_SAR_ratio_kobs5`           |   A    |
|  51 | `D_fft_ent_LST_modis_kobs30`      |   D    |
|  52 | `V_rollstd_F_NDVI_kobs7`          |   B    |
|  53 | `A_grad_s2_b12_kobs7`             |   A    |
|  54 | `A_pct_F_NDVI`                    |   A    |
|  55 | `A_d_s2_b12_kobs2`                |   A    |
|  56 | `A_grad_E_SAR_diff_kobs30`        |   A    |
|  57 | `A_d_F_NDVI_kobs2`                |   A    |
|  58 | `A_grad_E_SAR_diff_kobs7`         |   A    |
|  59 | `SMAP_sm_interp_rollrange7`       |   S    |
|  60 | `A_d_s2_b12_kobs7`                |   A    |
|  61 | `A_d_F_NDVI_kobs1`                |   A    |
|  62 | `V_rollcv_LST_modis_kobs14`       |   B    |
|  63 | `SMAP_sm_am_interp_rollstd7`      |   S    |
|  64 | `V_rollstd_SMAP_sm_interp_kobs7`  |   B    |
|  65 | `A_d_s2_b12_kobs5`                |   A    |
|  66 | `A_pct_SMAP_sm_interp`            |   A    |
|  67 | `SMAP_sm_am_interp_pctchg`        |   S    |
|  68 | `V_rollrng_SMAP_sm_interp_kobs7`  |   B    |
|  69 | `SMAP_sm_interp_pctchg`           |   S    |
|  70 | `A_d_E_SAR_diff_kobs2`            |   A    |
|  71 | `G_DSLR_isnan`                    |   G    |
|  72 | `SMAP_sm_pm_interp_mask`          |   S    |
|  73 | `SMAP_sm_interp_mask`             |   S    |
|  74 | `SMAP_sm_am_interp_mask`          |   S    |
|  75 | `SMAP_sm_am_interp_diff1`         |   S    |
|  76 | `SMAP_sm_interp_rollstd7`         |   S    |
|  77 | `SMAP_sm_interp_diff1`            |   S    |
|  78 | `V_rollstd_E_SAR_diff_kobs7`      |   B    |
|  79 | `A_grad_s2_b12_kobs14`            |   A    |
|  80 | `A_d_E_SAR_ratio_kobs7`           |   A    |
|  81 | `A_grad_LST_modis_kobs14`         |   A    |
|  82 | `A_d_SMAP_sm_interp_kobs1`        |   A    |
|  83 | `SMAP_sm_pm_interp_pctchg`        |   S    |
|  84 | `A_grad_E_SAR_ratio_kobs7`        |   A    |
|  85 | `A_d_E_SAR_diff_kobs1`            |   A    |
|  86 | `A_d_E_SAR_diff_kobs7`            |   A    |
|  87 | `SMAP_sm_pm_interp_diff1`         |   S    |
|  88 | `A_d_F_NDVI_kobs5`                |   A    |
|  89 | `A_d_E_SAR_ratio_kobs2`           |   A    |
|  90 | `A_d_G_API_kobs5`                 |   A    |
|  91 | `A_d_SMAP_sm_interp_kobs2`        |   A    |
|  92 | `D_fft_dom_E_SAR_ratio_kobs30`    |   D    |
|  93 | `SMAP_sm_pm_interp_rollstd7`      |   S    |
|  94 | `V_rollrng_s2_b12_kobs7`          |   B    |
|  95 | `V_rollrng_F_NDVI_kobs7`          |   B    |
|  96 | `A_d_SMAP_sm_interp_kobs14`       |   A    |
|  97 | `A_pct_E_SAR_ratio`               |   A    |
|  98 | `V_rollstd_SMAP_sm_interp_kobs30` |   B    |
|  99 | `A_d_E_SAR_ratio_kobs1`           |   A    |
| 100 | `A_pct_LST_modis`                 |   A    |
| 101 | `A_grad_SMAP_sm_interp_kobs14`    |   A    |
| 102 | `A_pct_E_SAR_diff`                |   A    |
| 103 | `SMAP_sm_interp_grad7`            |   S    |
| 104 | `A_grad_SMAP_sm_interp_kobs7`     |   A    |
| 105 | `A_d_LST_modis_kobs1`             |   A    |
| 106 | `V_rollcv_E_SAR_diff_kobs7`       |   B    |
| 107 | `A_d_s2_b11_kobs5`                |   A    |
| 108 | `V_rollstd_LST_modis_kobs7`       |   B    |

## Why N = 108

Short version: this was the best retained point under the pruning guardrails, not an arbitrary cutoff.

| Stage                 | Feature count | What happened                                            |
| --------------------- | ------------: | -------------------------------------------------------- |
| Family-filtered start |           408 | Included only families C, B, A, D, G, S                  |
| Coarse best           |           180 | Big reductions while improving validation score          |
| Fine best             |           108 | Smaller drops with rollback/patience when score softened |

The fine config had `target_n=60`, but the run kept bouncing between smaller sets and rollback states, so the best validated checkpoint stayed at **108**. That is why I'm keeping 108 instead of forcing the model down to 60...

## Appendix: Family Explanations

### Family A (39 features): Change dynamics

**What it captures:** Short-horizon movement in signals.

**Why it probably survived:** Seems like these features consistently carry event response info, especially fast wetting and dry-down behavior

**Representative pattern:**

$$A\_d\_x\_k = x_t - x_{t-k}, \qquad A\_grad\_x\_k = \frac{x_t - x_{t-k}}{k}, \qquad A\_pct\_x = \frac{x_t-x_{t-1}}{x_{t-1}+\epsilon}$$

### Family B (`V_`, 33 features): Volatility and smoothing

**What it captures:** Rolling spread, local extrema, and smoothed trend behavior at multiple windows.

**Why it probably survived:** They all show very stable signal + quality signal across validation rounds, especially on API, LST, SAR, and SMAP-derived series

**Representative pattern:**

$$V\_rollstd\_x\_k,\; V\_rollcv\_x\_k,\; V\_rollrng\_x\_k,\; V\_ema\_x\_k$$

### Family C (7 features): Memory and lag

**What it captures:** Persistence and delayed response.

**Why it probably survived:** In essence, rain and temperature effects are not one-step processes, so lag and memory terms stayed useful after pruning

**Representative pattern:**

$$C\_lag\_x\_k = x_{t-k}, \qquad C\_smm\_x = \sum_{j=1}^{n} \alpha^j x_{t-j}$$

### Family D (6 features): Seasonality and frequency context

**What it captures:** Calendar phase and spectral behavior in temporal signals.

**Why it probably survived:** They add context that helps separate seasonal baseline behavior from true anomalies

**Representative pattern:**

$$D\_sin\_DOY = \sin\!\left(2\pi\,\frac{DOY}{365}\right),\; D\_z\_x = \frac{x-\mu_m}{\sigma_m},\; D\_fft\_dom/ent\_x$$

### Family G (5 features): Meteorological forcing

**What it captures:** Direct rainfall forcing and dry-down timing.

**Why it probably survived:** From our experiments, API and rain accumulations are core drivers

**Representative pattern:**

$$G\_API_t = P_t + k\,G\_API_{t-1}, \qquad G\_rain\_sum\_{kd}, \qquad G\_DSLR_t$$

### Family S (`SMAP_`, 18 features): SMAP-derived temporal signals

**What it captures:** Coarse microwave moisture context plus derived temporal transforms (mask, pct change, diff, rolling, ema)

**Why it probably survived:** More of a guess here since we haven't experimented a whole lot, but from Dr. Zhou's feedback, it seems that these add complementary large-scale moisture signal and improves fit when paired with A/B/C/G style transforms

**Representative pattern:**

$$SMAP\_sm\_interp\_{t},\; SMAP\_ampm\_diff\_interp,\; SMAP\_*\_pctchg,\; SMAP\_*\_rollstd7$$

---

_Jakob Balkovec_
