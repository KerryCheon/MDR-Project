# data/

Processed dataset splits used for model training, validation, and evaluation.

## splits/

- `derived_9.0/` — canonical final split used for all results reported in the paper (temporal train/val/test: 2017–2020 / 2021–2022 / 2023–2025)
- `archive/` — earlier split versions (base_1.0, base_2.0, derived_1.0–8.0) and unseen ECE sensor datasets retained for reproducibility

## Raw data (not committed)

Raw station data lives in `Temporal/Pipeline/data/raw/` and is excluded from version control (gitignored). USCRN and SNOTEL source files are available from NOAA and NRCS respectively. API response caches are in `Temporal/Pipeline/data/cache/`.
