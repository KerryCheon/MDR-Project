# Audit: Touchet_WA_824 data completeness across derived splits

**Date:** 2026-08-05
**Reproducible source:** [`audit_touchet_data.py`](audit_touchet_data.py) — rerun with
`uv run --project notebooks python docs/audits/audit_touchet_data.py`. All tables below are copied verbatim from its stdout.

## Question

While building new splits based on `derived_8.1`, it was noticed that station `Touchet_WA_824` is missing large amounts of data — for several years it has no entries at all — and it was removed from the dataset on grounds of unreliable data. The question:

> **Was the Touchet multi-year gap newly introduced since `derived_8.1`, or has it always been there?**

## Answer (executive summary)

**The gap has always been there — it is not a `derived_8.1` regression.**

- Touchet has **zero entries in 2021, 2022, and 2023 in every split version that contains it** — from the earliest (`base_2.0`) through `derived_8.0`, `derived_8.1`, `derived_8.2`, and `derived_9.0`.
- `derived_8.0` already had the full problem: **1,397 train / 0 val / 170 test** rows. The val split (2021–2022) was empty and the test split had zero 2023 rows — exactly the "missing 2021–2023" pattern — back when Touchet was one of the original 5 stations.
- `derived_8.1` reproduces the identical pattern (1,397 / 0 / 171). Because two independent processing chains (the old pipeline behind `6.0`–`8.0`, and the fresh pipeline run behind `8.1`) yield the same gap, the missing years live in the **source SNOTEL record** (Touchet WA 824 has no soil-moisture data for 2021–2023; 2024–2025 are sparse, ~1 reading per 4 days), not in any pipeline step.
- What *is* new in the `8.1 → 8.2` chain is a secondary, smaller effect: the `soil_moisture_5cm > 0.0` filter applied in `derived_8.1_pos` further cut Touchet's train rows from 1,397 to 1,208. That shrank the train set but did **not** create the gap.
- Touchet was removed in **`derived_8.3`** (2026-07-21) on data-quality grounds; `derived_8.4` (2026-07-26) simply inherits that removal (it only prunes `MartenRidge_WA_999` and `RainyPass_WA_711`). The removal rationale explicitly noted the 0-val-row problem already existed in `derived_8.0`.

## Evidence

### Table 1. Touchet_WA_824 rows per split part, by dataset version

| Version | train | val | test | total |
|---|---:|---:|---:|---:|
| derived_8.0 | 1397 | 0 | 170 | 1567 |
| derived_8.1 | 1397 | 0 | 171 | 1568 |
| derived_8.1_pos | 1208 | 0 | 171 | 1379 |
| derived_8.2 | 1208 | 0 | 171 | 1379 |
| derived_8.3 | 0 | 0 | 0 | 0 |
| derived_8.4 | 0 | 0 | 0 | 0 |
| derived_9.0 | 1397 | 0 | 170 | 1567 |
| archive/base_1.0 | 0 | 0 | 0 | 0 |
| archive/base_2.0 | 2437 | 522 | 522 | 3481 |
| archive/derived_1.0 | 2437 | 522 | 522 | 3481 |
| archive/derived_2.0 | 2437 | 522 | 522 | 3481 |
| archive/derived_3.0 | 3311 | 0 | 170 | 3481 |
| archive/derived_4.0 | 3311 | 0 | 170 | 3481 |
| archive/derived_5.0 | 3311 | 0 | 170 | 3481 |
| archive/derived_6.0 | 3311 | 0 | 170 | 3481 |
| archive/derived_7.0 | 2191 | 302 | 170 | 2663 |

### Table 2. Touchet_WA_824 rows per calendar year (all parts combined)

| Version | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| derived_8.0 | 0 | 0 | 0 | 0 | 0 | 0 | 365 | 365 | 365 | 302 | 0 | 0 | 0 | 89 | 81 | 1567 |
| derived_8.1 | 0 | 0 | 0 | 0 | 0 | 0 | 365 | 365 | 365 | 302 | 0 | 0 | 0 | 90 | 81 | 1568 |
| derived_8.1_pos | 0 | 0 | 0 | 0 | 0 | 0 | 305 | 309 | 337 | 257 | 0 | 0 | 0 | 90 | 81 | 1379 |
| derived_8.2 | 0 | 0 | 0 | 0 | 0 | 0 | 305 | 309 | 337 | 257 | 0 | 0 | 0 | 90 | 81 | 1379 |
| derived_8.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| derived_8.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| derived_9.0 | 0 | 0 | 0 | 0 | 0 | 0 | 365 | 365 | 365 | 302 | 0 | 0 | 0 | 89 | 81 | 1567 |
| archive/base_1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| archive/base_2.0 | 87 | 366 | 365 | 365 | 365 | 366 | 365 | 207 | 50 | 302 | 0 | 0 | 0 | 89 | 81 | 3008 |
| archive/derived_1.0 | 87 | 366 | 365 | 365 | 365 | 366 | 365 | 207 | 50 | 302 | 0 | 0 | 0 | 89 | 81 | 3008 |
| archive/derived_2.0 | 87 | 366 | 365 | 365 | 365 | 366 | 365 | 207 | 50 | 302 | 0 | 0 | 0 | 89 | 81 | 3008 |
| archive/derived_3.0 | 87 | 366 | 365 | 365 | 365 | 366 | 365 | 365 | 365 | 302 | 0 | 0 | 0 | 89 | 81 | 3481 |
| archive/derived_4.0 | 87 | 366 | 365 | 365 | 365 | 366 | 365 | 365 | 365 | 302 | 0 | 0 | 0 | 89 | 81 | 3481 |
| archive/derived_5.0 | 87 | 366 | 365 | 365 | 365 | 366 | 365 | 365 | 365 | 302 | 0 | 0 | 0 | 89 | 81 | 3481 |
| archive/derived_6.0 | 87 | 366 | 365 | 365 | 365 | 366 | 365 | 365 | 365 | 302 | 0 | 0 | 0 | 89 | 81 | 3481 |
| archive/derived_7.0 | 0 | 0 | 0 | 365 | 365 | 366 | 365 | 365 | 365 | 302 | 0 | 0 | 0 | 89 | 81 | 2663 |

*Note:* `base_2.0`/`derived_1.0`/`derived_2.0` also carry pre-2011 Touchet rows (outside the 2011–2025 window shown here), which is why their Table 1 totals (3,481) exceed the Table 2 year-window sums (3,008). The 2018/2019 counts (207/50 in `base_2.0` → full 365/365 from `derived_3.0` onward) show the SNOTEL record was later backfilled, but the 2021–2023 gap is unchanged across every version.

### Table 3. derived_8.0: rows per station per year (the original 5-station split)

| station_id | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | total |
|---|---|---|---|---|---|---|---|---|---|---|
| Darrington | 340 | 350 | 324 | 363 | 335 | 336 | 317 | 350 | 332 | 3047 |
| Quinault | 353 | 364 | 364 | 366 | 351 | 362 | 323 | 361 | 360 | 3204 |
| SourdoughGulch_WA_985 | 365 | 365 | 365 | 366 | 365 | 364 | 365 | 366 | 175 | 3096 |
| Spokane | 273 | 307 | 274 | 332 | 330 | 277 | 261 | 327 | 309 | 2690 |
| Touchet_WA_824 | 365 | 365 | 365 | 302 | 0 | 0 | 0 | 89 | 81 | 1567 |

Touchet is the **only** one of the original 5 stations with empty years. The other four have essentially continuous daily coverage across 2017–2025.

### Table 4. derived_8.1 train: effect of the `soil_moisture_5cm > 0.0` filter
(applied in derived_8.1_pos and inherited by derived_8.2)

| station_id | rows before | rows after | retention % |
|---|---:|---:|---:|
| BeaverPass_WA_990 | 1459 | 1459 | 100.0% |
| BurntMountain_WA | 847 | 747 | 88.2% |
| CayusePass_WA | 1461 | 1414 | 96.8% |
| Darrington | 1377 | 1377 | 100.0% |
| HartsPass_WA_515 | 1187 | 1071 | 90.2% |
| MFNooksack_WA_1011 | 260 | 260 | 100.0% |
| MartenRidge_WA_999 | 1460 | 1454 | 99.6% |
| Paradise_WA | 1461 | 1459 | 99.9% |
| Quinault | 1447 | 1447 | 100.0% |
| RainyPass_WA_711 | 1459 | 1421 | 97.4% |
| SourdoughGulch_WA_985 | 1461 | 1461 | 100.0% |
| Spokane | 1186 | 1186 | 100.0% |
| Touchet_WA_824 | 1397 | 1208 | 86.5% |

Touchet loses the second-largest share of its train rows to this filter (86.5% retention, only `BurntMountain_WA` worse), consistent with a station whose record contains many frozen/zero soil-moisture readings. This is a real but *secondary* degradation introduced in the `8.1_pos`/`8.2` chain; it is unrelated to the 2021–2023 gap, which predates it.

## Removal timeline (why was it removed?)

| Date | Event | Source |
|---|---|---|
| 2026-07-20 | Touchet flagged as poor-quality station: per-station R² outlier, ~half the observations of peers, huge gaps | `docs/changelogs/2026-07-20-Pan.md` |
| 2026-07-21 | Data-quality-based removal analysis: 1,379 total rows, zero entries 2021–2023, 0 val rows, tiny test set. Doc explicitly notes *"Touchet was in the original 5-station set and had 0 val rows even [in derived_8.0]. It's always been problematic."* | `docs/plans/20260721-remove-incomplete-stations.md` |
| 2026-07-21 | **`derived_8.3` removes Touchet** (+ `BurntMountain_WA`, `HartsPass_WA_515`) | `data/splits/derived_8.3/split_meta.json` |
| 2026-07-26 | `derived_8.4` = `derived_8.3` minus `MartenRidge_WA_999`, `RainyPass_WA_711` (alpine-snowpack, out-of-scope rationale). Touchet is absent only because it was already removed in 8.3. | `data/splits/derived_8.4/split_meta.json`, `docs/plans/20260726-stations-removal.md` |

## Conclusion

1. The 2021–2023 data gap for `Touchet_WA_824` is **not** a `derived_8.1` regression — it is present in every split version that contains the station, going back to the earliest versions, and in the underlying SNOTEL record.
2. `derived_8.0` already contained the exact problem (0 val rows, zero 2023 test rows, sparse 2024–2025) while Touchet was one of the original 5 stations; the issue was simply less visible when diluted among 5 stations than among the 13-station `8.1` set.
3. Removal in `derived_8.3` was therefore based on a real, pre-existing data-quality problem, not on something introduced by `derived_8.1`. The `8.1 → 8.2` chain added only the secondary `target ≤ 0.0` filter, which trimmed Touchet's train set further but did not create the gap.
4. Side note: `derived_9.0` is built on top of `derived_8.0`, so it inherits the same Touchet gap (1,397 / 0 / 170).
