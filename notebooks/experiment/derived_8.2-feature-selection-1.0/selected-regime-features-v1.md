# Walkthrough: Regime-Specific Feature Selection for `derived_8.2`

I have successfully executed the regime-specific feature selection pipeline for the 5 target regimes of `derived_8.2`. The results have been saved to [dataset_metadata.py](file:///c:/Users/pan/Documents/GitHub/MDR-Project/data/splits/derived_8.2/dataset_metadata.py).

## Execution Details

All selections were executed using the **V3 settings** (`top_k = 50`, `stability_n_boot = 100`, `min_freq = 0.6`, `mi k = 300`, `elasticnet k = 60`), running through a script [run_regime_selections_v1.py](file:///c:/Users/pan/Documents/GitHub/MDR-Project/notebooks/experiment/derived_8.2-feature-selection/run_regime_selections_v1.py) designed to handle noisy regimes gracefully via bootstrap frequency fallback.

### Selections & Validation Performance

Below is a summary of the results across the 5 target regimes:

| Target Variable | Regime / Mask | Features Selected | Fallback Triggered | Top Features | Best Model Val R2 (Test R2) |
|---|---|---|---|---|---|
| `TERNARY_REGIME_DRY_SELECTED_FEATURES_V1` | Ternary Dry (`y < 0.16`) | 27 | No | `D_sin_DOY`, `J_bio_bio02`, `SMAP_sm_pm_interp_rollrange30`, `SMAP_x_year` | N/A (Only selected features run) |
| `TERNARY_REGIME_TRANSITION_SELECTED_FEATURES_V1` | Ternary Transition (`0.16 <= y < 0.25`) | 50 | Yes (`min_freq=0.0` fallback) | `API_x_year`, `J_bio_bio06`, `V_rollmin_G_API_kobs30`, `G_rain_sum_3d` | **RF**: 0.246 (0.045) / **XGB**: 0.229 (-0.240) |
| `TERNARY_REGIME_WET_SELECTED_FEATURES_V1` | Ternary Wet (`y >= 0.25`) | 6 | No | `D_cos_DOY`, `V_rollrng_E_SAR_diff_kobs30`, `latitude`, `slope` | **Linear**: 0.241 (-0.343) / **XGB**: 0.219 (-0.418) |
| `BINARY_REGIME_DRY_SELECTED_FEATURES_V1` | Binary Dry (`y < 0.16`) | 27 | No | `D_sin_DOY`, `J_bio_bio02`, `SMAP_sm_pm_interp_rollrange30`, `SMAP_x_year` | N/A (Only selected features run) |
| `BINARY_REGIME_WET_SELECTED_FEATURES_V1` | Binary Wet (`y >= 0.16`) | 45 | No | `DOY`, `D_cos_DOY`, `D_sa_F_NDMI`, `D_z_E_SAR_ratio` | **RF**: 0.316 (-0.069) / **XGB**: 0.287 (-0.117) |

---

## Technical Highlights

### 1. Robust Fallback in Noisy Regimes
In the **Ternary Transition** regime, the target distribution is very noisy, resulting in `0` stable features when enforcing the high stability threshold of `min_freq: 0.6`. To avoid crashes downstream in model training, the runner script dynamically falls back to `min_freq: 0.0`, extracting the top 50 features ranked by their bootstrap selection frequency.

### 2. High Stability in Wet Regimes
The **Ternary Wet** regime demonstrated highly distinct patterns, selecting only 6 highly stable features (like `D_cos_DOY`, `slope`, `latitude`, and `V_rollrng_E_SAR_diff_kobs30`) that met or exceeded the `0.6` stability frequency threshold.

---

## Verifications Completed

1. **Syntax Check**: Ran `python -m py_compile` on the modified `dataset_metadata.py` to ensure it compiles without any syntax or parsing errors.
2. **Git Status / Output Integrity**: Verified that all variables were successfully saved to [dataset_metadata.py](file:///c:/Users/pan/Documents/GitHub/MDR-Project/data/splits/derived_8.2/dataset_metadata.py) with the exact naming conventions requested by the user.
