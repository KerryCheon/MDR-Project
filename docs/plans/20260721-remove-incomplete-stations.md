# Removing incomplete data stations

**From the data exploration notebook (Section 2):**

The station data entry counts show severe incompleteness for several stations:

| Station | Total Rows | Notable Gaps |
|---------|-----------|--------------|
| **BurntMountain_WA** | 1,483 | 0 entries in 2019, only 7 in 2024, 49 in 2025 |
| **Touchet_WA_824** | 1,379 | Zero entries 2021-2023, only 90 in 2024, 81 in 2025 |
| **HartsPass_WA_515** | 1,600 | Only 91 in 2017, 19 in 2023, 70 in 2024, 0 in 2025 |
| **BeaverPass_WA_990** | 2,811 | Only 225 in 2024, 36 in 2025 |
| **MartenRidge_WA_999** | 2,957 | Only 81 in 2025 |
| **SourdoughGulch_WA_985** | 3,097 | Only 175 in 2025 |

And from the diagnostics table in the data exploration notebook, the worst-performing stations (model_r2) are:
- **Touchet_WA_824**: R² = -1.621 (only 1,379 rows, zero data 2021-2023)
- **BurntMountain_WA**: R² = -0.783 (only 1,483 rows, zero in 2019)
- **MartenRidge_WA_999**: R² = -0.186 (only 2,957 rows, only 81 in 2025)

Here's my analysis:

**Yes, it is scientifically sound** to remove stations with significantly incomplete data, for these reasons:

1. **BurntMountain_WA** (1,483 rows, R² = -0.783): Missing entire year 2019, only 7 entries in 2024. The model literally cannot learn meaningful patterns from this — the negative R² means it's worse than predicting the mean.

2. **Touchet_WA_824** (1,379 rows, R² = -1.621): Zero data 2021-2023, only 90 in 2024, 81 in 2025. This is the worst performer by far.

3. **HartsPass_WA_515** (1,600 rows, R² = 0.540): Only 91 entries in 2017, 19 in 2023, 70 in 2024, 0 in 2025. The model R² of 0.540 is the 3rd worst.

4. **MartenRidge_WA_999** (2,957 rows, R² = -0.186): Only 81 entries in 2025, and the model R² is negative.

The key scientific justification: these stations have **insufficient temporal coverage** — large multi-year gaps and sparse end-of-record data. The model cannot learn meaningful soil moisture dynamics from such fragmented time series, and including them actually **hurts the model's ability to generalize** to well-sampled stations by forcing it to accommodate pathological patterns from data-poor stations.

**However**, there's a nuance: if the goal is to build a model that generalizes across *all* stations in Washington (including poorly-sampled ones), removing them changes the problem statement. But if the goal is to build the best possible model for well-sampled stations (which is the stated focus — "stations within Washington state as part of the collaboration with the ECE team that develop and deploy new in-situ soil moisture sensors"), then removing data-poor stations is scientifically sound.

**My recommendation:** Remove **BurntMountain_WA** (R² = -0.783, only 1,483 rows, missing entire 2019) and **Touchet_WA_824** (R² = -1.621, only 1,379 rows, zero data 2021-2023). These are clear outliers with insufficient data. **HartsPass_WA_515** and **MartenRidge_WA_999** are borderline — they have more data but still poor performance and significant gaps. You could also consider removing them, but the case is weaker.

Would you like me to look at the derived_8.0 split to compare which stations were included there vs. derived_8.2?

---

## Data-Quality-Based Station Removal (Without Using Performance)

The key insight: we should identify stations to remove based on **data completeness alone**, not on model performance. Here's the data-quality assessment for each of the 7 new stations added in derived_8.2:

### Stations with Severe Data Quality Issues

**1. BurntMountain_WA** — **Strong candidate for removal**
- Only 747 train rows (vs. ~1,400+ for most stations)
- Only 246 test rows
- **Missing entire year 2019** (0 entries)
- Only 7 entries in 2024, 49 in 2025
- Total: 1,483 rows (least of any station)
- This is a fundamentally incomplete time series — you can't model soil moisture dynamics with a multi-year gap

**2. Touchet_WA_824** — **Strong candidate for removal**
- Only 1,208 train rows, **0 val rows**, 171 test rows
- **Zero data 2021-2023** (3 full years missing)
- Total: 1,379 rows
- Already had 0 val rows in derived_8.0 too — this station has always been problematic
- The 3-year gap means the model can't learn temporal dependencies for this station

**3. HartsPass_WA_515** — **Moderate candidate for removal**
- Only 1,071 train rows, 440 val rows, **89 test rows** (fewest test rows of any station)
- Only 91 entries in 2017, 19 in 2023, 70 in 2024, 0 in 2025
- Total: 1,600 rows
- The test set only has 89 samples — too few for reliable evaluation

**4. BurntMountain_WA** — **Moderate candidate for removal**
- Only 747 train rows (fewest), 246 test rows
- Missing entire 2019
- Total: 1,483 rows

---

Here's the plan I'd propose:

## Plan: Data-Quality-Based Station Filtering

### Criteria (data quality only, no performance feedback)

1. **Minimum total observations** — e.g., < 1,500 rows total (less than half of a full 9-year record)
2. **Multi-year gaps** — any station with ≥1 complete year of zero data
3. **Test set too small** — < 200 test rows (unreliable evaluation)

### Stations to Remove

| Station | Total Rows | Data Quality Issue | Criterion |
|---------|-----------|-------------------|-----------|
| **Touchet_WA_824** | 1,379 | Zero data 2021-2023 (3 full years), 0 val rows | Multi-year gap |
| **BurntMountain_WA** | 1,483 | Zero data 2019, only 7 in 2024, 49 in 2025 | Multi-year gap + sparse |
| **HartsPass_WA_515** | 1,600 | Only 19 in 2023, 70 in 2024, 0 in 2025, only 89 test rows | Sparse tail + tiny test set |

### Borderline
- **MartenRidge_WA_999** (2,957 rows) — only 81 in 2025, but otherwise well-sampled. Borderline.
- **BeaverPass_WA_990** (2,811 rows) — only 225 in 2024, 36 in 2025. Borderline.

### Keep (well-sampled)
- CayusePass_WA, Darrington, Paradise_WA, Quinault, RainyPass_WA_711, SourdoughGulch_WA_985, Spokane

### Key observation: Touchet_WA_824 was already in derived_8.0
Touchet was in the original 5-station set and had 0 val rows even there. It's always been problematic. The difference is that in derived_8.0 with only 5 stations, its impact was diluted less — but it still had only 170 test rows.

### Recommendation

Remove these 3 stations based on **data quality alone** (no performance feedback):

| Station | Total Rows | Data Quality Issue | Scientific Justification |
|---------|-----------|-------------------|-------------------------|
| **Touchet_WA_824** | 1,379 | 3 full years missing (2021-2023), 0 val rows | Cannot learn temporal dynamics with a 3-year gap; no validation data |
| **BurntMountain_WA** | 1,483 | Missing entire 2019, only 7 entries in 2024 | Cannot learn meaningful seasonal patterns with a full year gap |
| **HartsPass_WA_515** | 1,600 | Only 89 test rows, 0 in 2025, 19 in 2023 | Test set too small for reliable evaluation; end-of-record collapse |

### Borderline (could go either way)
- **MartenRidge_WA_999** (2,957 rows) — only 81 in 2025, but otherwise well-sampled. The tail gap is concerning but less severe.
- **BeaverPass_WA_990** (2,811 rows) — only 225 in 2024, 36 in 2025. Similar tail sparsity issue.

### What this means for the split
- Removing the 3 worst stations (Touchet, BurntMountain, HartsPass) would drop from 12 → 9 stations
- Removing all 5 problematic stations would drop from 12 → 7 stations (back closer to the original 5)
- The 5 original stations (Darrington, Quinault, SourdoughGulch, Spokane, Touchet) were all well-sampled except Touchet — which was already problematic in 8.0

### Recommended Plan

Create a **derived_8.3** split that removes stations based on objective data-quality thresholds:

1. **Minimum total observations < 1,500** → remove BurntMountain_WA (1,483), Touchet_WA_824 (1,379)
2. **Multi-year gap (≥1 complete year with 0 entries)** → remove BurntMountain_WA (0 in 2019), Touchet_WA_824 (0 in 2021-2023), HartsPass_WA_515 (0 in 2025)
3. **Test set < 100 rows** → remove HartsPass_WA_515 (89 test rows)

This gives a clean 9-station split (removing Touchet, BurntMountain, HartsPass) — or a stricter 7-station split if you also remove MartenRidge and BeaverPass for their sparse tails.