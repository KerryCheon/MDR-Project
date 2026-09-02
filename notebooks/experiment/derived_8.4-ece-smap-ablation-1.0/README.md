# Experiment: `derived_8.4-ece-smap-ablation-1.0`

This experiment measures the effect of the confirmed SMAP zero-fill defect in
the 2026 ECE evaluation and compares operational missing-data policies without
using ECE targets for training or imputation. The model is the weighted
`derived_8.4` 38-feature configuration from
`derived_8.4-ece-additional-eval-1.0`, evaluated over five fixed seeds.

## Input audit

The executed notebook reported 14,608 training rows, 150 ECE evaluation rows,
38 selected features, and six selected SMAP features. The parent zero-filled
input contained 650 finite SMAP cells across those selected features; the
corrected native-missing input contained zero.

## Five-seed pooled results

| strategy | RMSE mean | RMSE std | MAE mean | bias mean | ubRMSE mean | R-squared mean | Pearson r mean | RMSE change vs zero |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| zero_filled_existing_policy | 0.074504 | 0.007252 | 0.067093 | -0.055493 | 0.049306 | -1.526627 | 0.089987 | 0.000000 |
| native_missing_existing_training | 0.072539 | 0.003958 | 0.065026 | -0.054195 | 0.048075 | -1.382752 | 0.117342 | -0.001965 |
| training_month_climatology | 0.069335 | 0.007528 | 0.062513 | -0.048486 | 0.048974 | -1.192240 | 0.087492 | -0.005169 |
| no_smap_retrained | 0.070625 | 0.004369 | 0.063337 | -0.050787 | 0.048976 | -1.260177 | 0.104094 | -0.003879 |
| block_masked_retrained | 0.070121 | 0.011260 | 0.062790 | -0.048459 | 0.049594 | -1.267034 | 0.079917 | -0.004384 |

## Paired seed consistency

| strategy | mean RMSE delta vs zero | improved seeds | total seeds |
|:--|--:|--:|--:|
| zero_filled_existing_policy | 0.000000 | 0 | 5 |
| native_missing_existing_training | -0.001965 | 3 | 5 |
| training_month_climatology | -0.005169 | 5 | 5 |
| no_smap_retrained | -0.003879 | 3 | 5 |
| block_masked_retrained | -0.004384 | 4 | 5 |

## Station-level mean RMSE

| station | zero fill | native missing | training-month climatology | no SMAP | block masked |
|:--|--:|--:|--:|--:|--:|
| ECE_BBG_Lost_Meadow | 0.072933 | 0.067974 | 0.069033 | 0.073044 | 0.071875 |
| ECE_BBG_Main_St | 0.069081 | 0.066829 | 0.061026 | 0.061976 | 0.060089 |
| ECE_Renton_Garden_North | 0.039446 | 0.036958 | 0.043987 | 0.042026 | 0.044900 |
| ECE_Renton_Garden_Shed | 0.057334 | 0.055909 | 0.049735 | 0.052408 | 0.049766 |
| ECE_Renton_Home | 0.112782 | 0.112584 | 0.104630 | 0.105653 | 0.104454 |

## Interpretation

Training-month climatology has the best mean RMSE, improving over zero-fill by
0.005169 m3/m3, or 6.94%, and improves all five paired seeds. This establishes
that zero-fill was harmful, but SMAP handling is not the dominant ECE failure:
the remaining mean bias is approximately -0.0485 m3/m3 even under the best
policy.

The effect is heterogeneous. Native missing performs best at Lost Meadow and
Renton Garden North, whereas climatology or block masking performs best at the
other three stations. The current operational recommendation is therefore to
use training-only monthly climatology with explicit missingness provenance for
this model family, retain a no-SMAP fallback for persistent urban retrieval
gaps, and avoid describing climatology as universally station-optimal.

## Reproduction

From `notebooks/`, run:

```powershell
nb execute experiment/derived_8.4-ece-smap-ablation-1.0/derived_8.4-ece-smap-ablation-1.0.ipynb --uv --timeout 900
```

The tables above are transcribed strictly from stdout emitted by the fully
executed notebook. Versioned outputs are `summary.csv`, `seed_metrics.csv`,
`station_metrics.csv`, and `masking_audit.json`.
