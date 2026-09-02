# ECE SMAP Missingness Correction and Ablation
## Kerry Cheon 08/31/2026

## Context:

The `derived_8.4-ece` evaluation set contains 150 daily observations from five newly deployed ECE stations in Bellevue and Renton. The original error-analysis report described the ECE SMAP problem as a global 2026 Google Earth Engine latency gap and reported 85 SMAP-related features as zero across the evaluation set.

The underlying issue is more specific. The active pipeline queries `NASA/SMAP/SPL3SMP_E/005` and `/006`, not the `NASA_USDA/HSL/SMAP10KM_soil_moisture` collection named in the earlier report. The ECE satellite caches contain null SMAP AM and PM retrieval bands for every evaluation interval, while another station cache contains valid SMAP during overlapping dates. Later August observations may still be affected by catalog latency, but latency alone does not explain the location-specific null retrievals.

Two missing source streams, SMAP AM and PM, expand into 85 columns after masks, lags, rolling statistics, gradients, and interactions are generated. The dataset builder converted the missing source values to `0.0` before generating those descendants. It then marked the artificial zeros as observed. This made the model interpret missing satellite retrievals as physically dry soil.

One additional correction came from the repository fixtures. A SMAP quality value of `1` cannot be treated as universally unavailable because historical caches contain finite retrievals paired with that bit-field value. The retrieval band's actual null mask is therefore the availability authority in this work; quality flags remain diagnostic metadata.

## What I ran this week

| # | Experiment | Code / Output | Main Result | Bottom line |
|---|---|---|---|---|
| 1 | Trace ECE SMAP nulls through the pipeline and correct the split builder | `data/splits/derived_8.4-ece/make_derived_8.4_ece.py`, ECE satellite caches | Confirmed that missing AM/PM values were zero-filled before masks and 82 value/derived columns were generated | The original split has a real missing-data semantics bug; zero is not an observed SMAP value |
| 2 | Build a versioned native-missing ECE split | `data/splits/derived_8.4-ece-v2-native-missing/` | Preserved 150 rows and the exact 499-column schema; 82 SMAP value columns are missing and three masks are zero | The correction is available without modifying the historical split |
| 3 | Five-seed SMAP policy ablation | `notebooks/experiment/derived_8.4-ece-smap-ablation-1.0/` | Monthly climatology reaches RMSE=0.06934 and R²=-1.192 versus RMSE=0.07450 and R²=-1.527 for zero-fill | Zero-fill is harmful, but SMAP policy explains only part of the ECE transfer error |
| 4 | Seed- and station-level robustness audit | `seed_metrics.csv`, `station_metrics.csv`, executed notebook | Climatology improves all five seeds but is not best at every station | Keep native missing and no-SMAP baselines; do not claim one imputation policy is universally optimal |

## Experiment scope and protocol

The ablation uses the weighted 38-feature XGBoost configuration from `derived_8.4-ece-additional-eval-1.0`. The training pool is `derived_8.4` train plus validation: 14,608 rows from seven Washington reference stations. The evaluation pool is the same 150 ECE rows under two representations: the historical zero-filled split and the corrected native-missing split.

The selected model inputs contain six SMAP-derived features. The zero-filled evaluation input has 650 finite cells across those six features; the corrected input has none. ECE targets are never used for model fitting or imputation. All results use seeds 42, 7, 13, 101, and 123.

I report pooled RMSE, MAE, bias, ubRMSE, R², and Pearson correlation. RMSE and bias remain the primary physical metrics. R² is retained because it is useful for policy comparison on identical rows, but its absolute magnitude is unstable for these short, low-variance station records.

The five policies are:

1. Original zero-filled ECE inputs with standard training.
2. Native `NaN` inputs with standard training and XGBoost missing branches.
3. Training-only monthly median climatology for the six selected SMAP features.
4. Retraining after removing the six selected SMAP features.
5. Retraining after masking deterministic 30-day SMAP blocks in 20% of each station's training blocks.

The 20% masking fraction is a documented simulation setting rather than a tuned constant. It still needs a sensitivity sweep.

## Experiment 1: Correct the ECE SMAP data semantics

Finding: the missingness audit was directionally correct—ECE SMAP is unavailable—but the cause and downstream representation were wrong.

The canonical builder previously performed `.fillna(0.0)` on `SMAP_sm_am_interp` and `SMAP_sm_pm_interp` before calling the feature generator. The feature generator then calculated observation masks from those zeros. All later SMAP lags, rolling values, and interactions inherited the artificial signal.

I changed the builder to preserve retrieval-band `NaN` values and added `prepare_smap_columns` for explicit numeric conversion without zero substitution. The builder now defaults to a versioned output directory instead of overwriting `derived_8.4-ece`.

The raw and processed ECE station files are absent from this checkout, so the canonical raw-to-split build cannot currently be rerun. I added a tracked recovery builder that first verifies every station/date against the committed satellite-cache intervals. It aborts if any finite SMAP AM/PM value is found or if a date is not covered. Only after that verification does it restore all SMAP descendants to native missing values.

| Dataset property | Historical split | Corrected split |
|---|---:|---:|
| Rows | 150 | 150 |
| Columns | 499 | 499 |
| Column order | Reference | Identical to reference |
| SMAP-related columns | 85 | 85 |
| Finite SMAP value/derived cells | Artificial zeros present | 0 |
| SMAP observation masks | Incorrectly observed | 0 for every row |

The original `derived_8.4-ece` split remains unchanged for historical comparison. The corrected split is `derived_8.4-ece-v2-native-missing`.

## Experiment 2: Compare SMAP missing-data policies

Finding: every alternative improves mean RMSE over zero-fill. Training-only monthly climatology is best on average and is the only policy that improves all five paired seeds.

| Policy | Mean RMSE | RMSE std | Mean MAE | Mean bias | Mean ubRMSE | Mean R² | Mean Pearson r | RMSE change vs. zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-filled existing policy | 0.074504 | 0.007252 | 0.067093 | -0.055493 | 0.049306 | -1.526627 | 0.089987 | 0.000000 |
| Native missing, existing training | 0.072539 | **0.003958** | 0.065026 | -0.054195 | **0.048075** | -1.382752 | **0.117342** | -0.001965 |
| **Training-month climatology** | **0.069335** | 0.007528 | **0.062513** | -0.048486 | 0.048974 | **-1.192240** | 0.087492 | **-0.005169** |
| No-SMAP retraining | 0.070625 | 0.004369 | 0.063337 | -0.050787 | 0.048976 | -1.260177 | 0.104094 | -0.003879 |
| Block-masked retraining | 0.070121 | 0.011260 | 0.062790 | **-0.048459** | 0.049594 | -1.267034 | 0.079917 | -0.004384 |

Monthly climatology lowers RMSE by 0.005169 m³/m³, or 6.94%, relative to zero-fill. Its mean R² improves by +0.3344, from -1.5266 to -1.1922. Native missing alone improves RMSE by 0.001965 and R² by +0.1439, confirming that the artificial zeros were harmful even without retraining.

All R² values remain negative. Under squared error, every policy is worse than predicting the pooled ECE target mean. The correction therefore improves the model without solving the broader transfer problem.

The best-policy bias is still -0.048486 m³/m³, which is much larger than the 0.005169 RMSE gain attributable to the SMAP policy. The remaining error is dominated by calibration and domain mismatch rather than satellite missingness alone.

## Experiment 3: Check seed and station consistency

Finding: climatology is the most seed-consistent policy, but its effect varies by station.

| Policy | Mean paired RMSE delta | Seeds improved |
|---|---:|---:|
| Zero fill | 0.000000 | 0/5 |
| Native missing | -0.001965 | 3/5 |
| **Training-month climatology** | **-0.005169** | **5/5** |
| No-SMAP retraining | -0.003879 | 3/5 |
| Block-masked retraining | -0.004384 | 4/5 |

| Station | Zero fill | Native missing | Monthly climatology | No SMAP | Block masked | Best policy |
|---|---:|---:|---:|---:|---:|---|
| ECE_BBG_Lost_Meadow | 0.072933 | **0.067974** | 0.069033 | 0.073044 | 0.071875 | Native missing |
| ECE_BBG_Main_St | 0.069081 | 0.066829 | 0.061026 | 0.061976 | **0.060089** | Block masked |
| ECE_Renton_Garden_North | 0.039446 | **0.036958** | 0.043987 | 0.042026 | 0.044900 | Native missing |
| ECE_Renton_Garden_Shed | 0.057334 | 0.055909 | **0.049735** | 0.052408 | 0.049766 | Monthly climatology |
| ECE_Renton_Home | 0.112782 | 0.112584 | 0.104630 | 0.105653 | **0.104454** | Block masked |

Climatology performs best at Garden Shed and is close to best at Main St and Renton Home, but it makes Renton Garden North worse than zero-fill. Native missing performs best at Lost Meadow and Garden North. The mean result is useful for choosing a default policy, but the station table argues against presenting climatology as universally reliable.

## What I think is going on

1. **Zero-fill creates a real but secondary error.** The model treats a missing coarse satellite retrieval as physically dry soil. Removing that false signal improves mean RMSE and R².
2. **Climatology mainly reduces systematic bias.** It moves mean bias from -0.0555 to -0.0485 m³/m³, but does not materially improve ubRMSE relative to native missing.
3. **The remaining failure is not primarily SMAP.** Even the best policy has negative pooled R² and large dry bias. Sensor calibration, urban micro-siting, and reference-to-ECE domain mismatch remain larger problems.
4. **Different stations need different missing-data behavior.** Garden North benefits from native missing and is harmed by climatology, while Shed and Home benefit from a supplied climatological or outage-robust signal.
5. **No-SMAP is a credible operational fallback.** It is better than zero-fill on average and avoids inventing satellite values for urban footprints with persistent retrieval gaps.
6. **Block masking helps, but its variance is high.** It improves four of five seeds and has strong mean performance, but RMSE standard deviation rises to 0.0113. The masking rate and block construction need a sweep.


