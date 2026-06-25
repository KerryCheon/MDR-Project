# data/

Processed dataset splits used for model training, validation, and evaluation.

## splits/

| Dataset | Parent | Stations | Rows (train / val / test) | Time Range | Key Changes | Notes |
|---|---|---|---|---|---|---|
| `derived_7.0/` | derived_6.0 | 5 (Spokane, Darrington, Quinault, Touchet_WA_824, SourdoughGulch_WA_985) | 18,889 (10,424 / 4,449 / 4,016) | 2014–2025 | Added drift features (year_frac, sin_year, cos_year, API_x_year, SMAP_x_year) | Wider pre-SMAP coverage (2014–2016) |
| `derived_8.0/` | derived_7.0 | 5 (Spokane, Darrington, Quinault, Touchet_WA_824, SourdoughGulch_WA_985) | 13,604 (6,868 / 2,720 / 4,016) | 2017–2025 | Filtered to SMAP post-2016; added station-level pass-specific LIA features ||
| `derived_9.0/` | derived_8.0 | 31 (5 original + 8 WA SNOTEL, 11 USCRN, 7 SCAN) | ~63,165 (~29,362 / ~13,637 / ~20,166) | 2017–2025 | Appended 49,561 new rows via ISMN SNOTEL download; added HWSD soil fields; some remote-sensing features still pending upstream | This is the canonical final split used for all results reported in the paper |

**Split years:** derived_7.0 uses train = 2014–2019, val = 2020–2022, test = 2023–2025. Derived_8.0 and derived_9.0 use train = 2017–2020, val = 2021–2022, test = 2023–2025.

- `archive/` — earlier split versions (base_1.0, base_2.0, derived_1.0–7.0) and unseen ECE sensor datasets retained for reproducibility

## Raw data (not committed)

Raw station data lives in `Temporal/Pipeline/data/raw/` and is excluded from version control (gitignored). USCRN and SNOTEL source files are available from NOAA and NRCS respectively. API response caches are in `Temporal/Pipeline/data/cache/`.
