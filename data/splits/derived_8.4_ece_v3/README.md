# `derived_8.4_ece_v3`

Quality-verified 499-column in-situ evaluation split for the 5 ECE soil moisture sensor stations across Bellevue and Renton, Washington (July 20 – August 19, 2026).

## Key Upgrades from `derived_8.4-ece` (v1) and `derived_8.4-ece-v2-native-missing` (v2)

1. **30-Day Rolling Warmup Continuity (Eliminating August 19 Boundary Drop)**:
   - In v1/v2, the input time series started on July 20 without historical warmup. For Days 1–29, 30-day rolling statistics (`V_rollmin_G_API_kobs30`, etc.) evaluated to `NaN`. On Day 30 (August 19), exactly 30 observations accumulated, abruptly switching `V_rollmin_G_API_kobs30` to a numerical value (`0.0`). Because this feature had 23.9% importance in `d84_weighted`, this sudden switch routed predictions to an extreme dry leaf node.
   - In v3, a 30-day warmup scaffold (June 20 – July 19, 2026) was processed through ERA5 weather and GEE satellite pipes. All 30-day rolling statistics (`V_rollmin_G_API_kobs30`, `V_rollmean_G_API_kobs30`, etc.) are fully warm, continuous, and non-NaN from the very first evaluation day. The evaluation split is then strictly filtered to July 20 – August 19.

2. **Empirically-Grounded Native-Missing SMAP Policy**:
   - Direct GEE probing of `NASA/SMAP/SPL3SMP_E/006` across all 5 ECE stations confirmed that SMAP Level-3 soil moisture retrievals are masked (`None`) across all years (including 2025 and 2026) because NASA's radiometer algorithm flags the urban/suburban Puget Sound corridor as an urban retrieval failure (`retrieval_qual_flag = 1`) due to RFI and complex heterogeneous land cover.
   - All 82 SMAP value columns are strictly preserved as native `NaN` with all 3 observation masks set to `0`. There are 0 spurious `0.0` values. Tree-based models (XGBoost / LightGBM) follow their trained default missing-value direction rather than traversing false bone-dry splits.

3. **MODIS NDVI 16-Day Latency Fallback**:
   - `OptimizedSatellitePipe` includes conditional fallback to the latest available 16-day MODIS observation (`MODIS/061/MOD13Q1` / `MODIS/061/MOD13A3`) when weekly reduction windows encounter upstream publication latency, accompanied by explicit warning logs.

## Split Statistics

- **Total rows**: 150 (30 days × 5 stations)
  - `ECE_BBG_Lost_Meadow`: 30
  - `ECE_BBG_Main_St`: 30
  - `ECE_Renton_Garden_North`: 30
  - `ECE_Renton_Garden_Shed`: 30
  - `ECE_Renton_Home`: 30
- **Total columns**: 499 (exact 1:1 schema and ordering parity with `data/splits/derived_8.4/train.csv`)
- **Evaluation date range**: `2026-07-20` to `2026-08-19` (excluding partial start Jul 19, partial end Aug 20, missing Aug 1)
- **Target `soil_moisture_5cm`**: Min 0.0145, Max 0.2151, Mean 0.0988, 0 NaNs

## Rebuilding & Auditing

From the repository root:

```bash
# 1. Run pipeline for ECE v3 stations (warmup scaffold + evaluation)
PYTHONPATH=. uv run python -m src.pipeline.main --config src/pipeline/config_8.4_ece_v3.yaml

# 2. Build the split
PYTHONPATH=. uv run python data/splits/derived_8.4_ece_v3/make_derived_8.4_ece_v3.py

# 3. Run automated quality audit
PYTHONPATH=. uv run python data/splits/derived_8.4_ece_v3/quality_audit.py
```
