# SMAP Probe Findings — ECE Stations

**Date:** 2026-09-02
**Probe script:** `smap_availability_probe.py`

## Conclusion: SMAP Permanently Unavailable for ECE Stations

All 5 ECE stations return `soil_moisture_am = None` for **all years** (tested 2025 July and 2026 August), not just for the NSIDC geolocation error period (May 14 – Jul 28, 2026).

| Station | 2025 Jul sm_am | 2026 Aug sm_am | Verdict |
|---|---|---|---|
| ECE_BBG_Main_St | None | None | ALWAYS NULL |
| ECE_BBG_Lost_Meadow | None | None | ALWAYS NULL |
| ECE_Renton_Home | None | None | ALWAYS NULL |
| ECE_Renton_Garden_North | None | None | ALWAYS NULL |
| ECE_Renton_Garden_Shed | None | None | ALWAYS NULL |

## Root Cause

SMAP's L-Band soil moisture retrieval is masked in urban/suburban environments due to:
1. **Radio Frequency Interference (RFI)** from urban electronics corrupting the 1.41 GHz signal
2. **Mixed land-cover** within the ~9km resolution pixel (roads, buildings, impervious surfaces)

The retrieval quality algorithm flags these pixels as failed (`retrieval_qual_flag_am = 1`),
causing `soil_moisture_am` to be masked to `None` permanently.

The brightness temperature bands (`tb_h_corrected_am`, `tb_v_corrected_am`) are present,
but the downstream soil moisture retrieval fails.

## Implication for derived_8.4_ece_v3

SMAP-derived features (85 columns in the 499-column schema) will be NaN for all ECE rows.
This is **physically correct** and must be represented as NaN (not 0.0).
The v3 builder must ensure no zero-fill occurs in `add_smap_features()`.
