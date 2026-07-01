# data/

Processed dataset splits used for model training, validation, and evaluation.

## splits/

| Dataset | Parent | Stations | Rows (train / val / test) | Split Years (train / val / test) | Time Range | Key Changes | Notes |
|---|---|---|---|---|---|---|---|
| `derived_7.0/` | derived_6.0 | <details><summary>5 stations</summary>Spokane, Darrington, Quinault, Touchet_WA_824, SourdoughGulch_WA_985</details> | 18,889 (10,424 / 4,449 / 4,016) | 2014–2019 / 2020–2022 / 2023–2025 | 2014–2025 | Added drift features (`year_frac`, `sin_year`, `cos_year`, `API_x_year`, `SMAP_x_year`) | Wider pre-SMAP coverage (2014–2016) |
| `derived_8.0/` | derived_7.0 | <details><summary>5 stations</summary>Spokane, Darrington, Quinault, Touchet_WA_824, SourdoughGulch_WA_985</details> | 13,604 (6,868 / 2,720 / 4,016) | 2017–2020 / 2021–2022 / 2023–2025 | 2017–2025 | Filtered to SMAP post-2016; added station-level pass-specific LIA features | |
| `derived_8.1/` | derived_9.0? | <details><summary>13 stations</summary><ul><li><b>Original:</b> Spokane, Darrington, Quinault, Touchet_WA_824, SourdoughGulch_WA_985</li><li><b>WA SNOTEL:</b> BeaverPass_WA_990, BurntMountain_WA, CayusePass_WA, HartsPass_WA_515, MFNooksack_WA_1011, MartenRidge_WA_999, Paradise_WA, RainyPass_WA_711</li></ul></details> | 34,775 (16,462 / 7,714 / 10,599) | 2017–2020 / 2021–2022 / 2023–2025 | 2017–2025 | Added 8 WA SNOTEL stations (total 13 stations) with full feature parity; bimodal valley-based regime calibration thresholds (T1=0.16, T2=0.25) | Washington-only dataset used for localized modeling and regime calibration validation |
| `derived_8.1_pos/` | derived_8.1 | <details><summary>13 stations</summary><ul><li><b>Original:</b> Spokane, Darrington, Quinault, Touchet_WA_824, SourdoughGulch_WA_985</li><li><b>WA SNOTEL:</b> BeaverPass_WA_990, BurntMountain_WA, CayusePass_WA, HartsPass_WA_515, MFNooksack_WA_1011, MartenRidge_WA_999, Paradise_WA, RainyPass_WA_711</li></ul></details> | 32,015 (15,964 / 7,149 / 8,902) | 2017–2020 / 2021–2022 / 2023–2025 | 2017–2025 | Filtered `soil_moisture_5cm > 0.0` from derived_8.1 (~2,760 zero/NaN rows dropped); full 499-feature suite preserved | Positive-only subset for scenarios where non-positive moisture is undesirable |
| `derived_9.0/` | derived_8.0 | <details><summary>31 stations</summary><ul><li><b>Original:</b> Spokane, Darrington, Quinault, Touchet_WA_824, SourdoughGulch_WA_985</li><li><b>WA SNOTEL:</b> BeaverPass_WA_990, BurntMountain_WA, CayusePass_WA, HartsPass_WA_515, MFNooksack_WA_1011, MartenRidge_WA_999, Paradise_WA, RainyPass_WA_711</li><li><b>USCRN:</b> USCRN_Arco_17_SW, USCRN_Coos_Bay_8_SW, USCRN_Corvallis_10_SSW, USCRN_Darrington_21_NNE, USCRN_Dillon_18_WSW, USCRN_John_Day_35_WNW, USCRN_Murphy_10_W, USCRN_Quinault_4_NE, USCRN_Riley_10_WSW, USCRN_Spokane_17_SSW, USCRN_St_Mary_1_SSW</li><li><b>SCAN:</b> SCAN_ConradAgRc, SCAN_CookFarmFieldD, SCAN_JordanValleyCwma, SCAN_Lind_1, SCAN_OrchardRangeSite, SCAN_TableMountain, SCAN_Violett</li></ul></details> | ~63,165 (~29,362 / ~13,637 / ~20,166) | 2017–2020 / 2021–2022 / 2023–2025 | 2017–2025 | Appended 49,561 new rows via ISMN SNOTEL download; added HWSD soil fields; some remote-sensing features still pending upstream | This is the canonical final split used for all results reported in the paper |

- `archive/` — earlier split versions (base_1.0, base_2.0, derived_1.0–7.0) and unseen ECE sensor datasets retained for reproducibility

## Raw data (not committed)

Raw station data lives in `Temporal/Pipeline/data/raw/` and is excluded from version control (gitignored). USCRN and SNOTEL source files are available from NOAA and NRCS respectively. API response caches are in `Temporal/Pipeline/data/cache/`.
