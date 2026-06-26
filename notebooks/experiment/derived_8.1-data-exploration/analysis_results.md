# derived_8.1 Soil Moisture Dataset Exploratory Data Analysis Report

This report documents the exploratory data analysis (EDA) conducted on the `derived_8.1` soil moisture dataset, comparing it against the `derived_8.0` baseline. 

The primary objective is to evaluate:
1. **Dataset Quality**: Whether the new dataset provides a larger, more representative sample size across the target space.
2. **Regime Distribution**: How samples are distributed across **Dry**, **Transition**, and **Wet** moisture regimes.
3. **Threshold Validity**: Whether the original thresholds hold or require recalibration.

The code and figures generated for this study are located in:
- Interactive Notebook: [derived_8.1_exploration.ipynb](derived_8.1_exploration.ipynb)

---

## 1. Dataset Scale and Station Coverage

`derived_8.1` expands the spatial coverage of the Washington-only subset by adding **8 new SNOTEL stations** to the **5 original stations**, raising the total station count from 5 to 13.

* **derived_8.0 Stations (5)**: Spokane, Darrington, Quinault, Touchet_WA_824, SourdoughGulch_WA_985
* **derived_8.1 Stations (13)**: Original 5 + BeaverPass_WA_990, BurntMountain_WA, CayusePass_WA, HartsPass_WA_515, MFNooksack_WA_1011, MartenRidge_WA_999, Paradise_WA, RainyPass_WA_711

This yields a **2.5x increase** in total observations:

| Metric | derived_8.0 | derived_8.1 | Change |
|---|---|---|---|
| **Stations** | 5 | 13 | +8 stations |
| **Total Rows** | 13,604 | 34,775 | +21,171 rows (+155.6%) |
| **Train Split** | 6,868 | 16,462 | +9,594 rows |
| **Val Split** | 2,720 | 7,714 | +4,994 rows |
| **Test Split** | 4,016 | 10,599 | +6,583 rows |

---

## 2. Soil Moisture Target Distribution & Quantiles

Analyzing the percentiles of `soil_moisture_5cm` in the training splits reveals a shift in the overall distribution:

| Percentile | derived_8.0 (Train) | derived_8.1 (Train) | Difference |
|---|---|---|---|
| **0% (Min)** | 0.0000 | 0.0000 | 0.0000 |
| **10%** | 0.0390 | 0.0320 | -0.0070 |
| **25%** | 0.1210 | 0.1160 | -0.0050 |
| **33% (t1-cal)** | **0.1570** | **0.1430** | **-0.0140** |
| **50% (Median)** | 0.2060 | 0.2075 | +0.0015 |
| **66% (t2-cal)** | **0.2530** | **0.2690** | **+0.0160** |
| **75%** | 0.2800 | 0.3020 | +0.0220 |
| **90%** | 0.3203 | 0.3530 | +0.0327 |
| **100% (Max)** | 0.4390 | 0.4390 | 0.0000 |

### Density Distribution comparison
The plot below displays the density distribution of `soil_moisture_5cm` for the training sets of both versions, showing the locations of the original and recalibrated boundaries:

![Soil Moisture Density Comparison](soil_moisture_density_comparison.png)

> [!NOTE]
> The new dataset has a wider, flatter profile in the middle-to-high moisture range (0.20 to 0.38), indicating a richer sample set for intermediate and wet regimes.

---

---

## 3. Aggregated Regime Proportions Comparison

Below is the aggregated distribution of Dry, Transition, and Wet regimes across both datasets, comparing:
1. `derived_8.0` Train set (with original thresholds: $t_1 = 0.20, t_2 = 0.313$)
2. `derived_8.1` Train set (with original thresholds: $t_1 = 0.20, t_2 = 0.313$)
3. `derived_8.1` Train set (with recalibrated thresholds: $t_1 = 0.143, t_2 = 0.269$)

![Aggregated Regime Comparison](aggregated_regime_comparison.png)

This highlights how the inclusion of new Washington SNOTEL stations increases the proportion of wet observations under the original thresholds (from 12.8% to 21.1%), and how recalibrating the thresholds balances the dataset perfectly at 33% per regime.

---

## 4. Regime Threshold Validation

In the three-regime modeling framework, samples are routed into **Dry**, **Transition**, and **Wet** classes. We evaluated two threshold options on the training sets:

### Option A: Original 8.0 Thresholds ($t_1 = 0.20, t_2 = 0.313$)
These boundaries were calibrated on `derived_8.0` and result in severe class imbalance:
* **derived_8.0 Train**: Dry: 47.7%, Transition: 39.5%, **Wet: 12.8%**
* **derived_8.1 Train**: Dry: 47.8%, Transition: 31.1%, **Wet: 21.1%**

While the new SNOTEL stations double the proportion of Wet samples under the original thresholds (from 12.8% to 21.1%), the overall distribution remains heavily skewed toward Dry and Transition.

### Option B: Recalibrated 8.1 Thresholds ($t_1 = 0.143, t_2 = 0.269$)
To ensure a balanced sample base for each regime-specific model, we recalibrated the thresholds using the **33rd and 66th percentiles** of the `derived_8.1` training split:
* **Dry**: $y < 0.143$
* **Transition**: $0.143 \leq y < 0.269$
* **Wet**: $y \geq 0.269$

This recalibration successfully partitions the training split into balanced classes and maintains a reasonable balance in the validation and test splits:

| Split | Dry | Transition | Wet | Total Rows |
|---|---|---|---|---|
| **Train** | 5,419 (32.9%) | 5,431 (33.0%) | 5,612 (34.1%) | 16,462 |
| **Val** | 3,253 (42.2%) | 1,850 (24.0%) | 2,611 (33.8%) | 7,714 |
| **Test** | 4,756 (44.9%) | 3,448 (32.5%) | 2,395 (22.6%) | 10,599 |

> [!WARNING]
> The original thresholds ($t_1 = 0.20, t_2 = 0.313$) are misaligned with the new dataset structure and should **not** be reused. Recalibrating to $t_1 = 0.143$ and $t_2 = 0.269$ provides a much more balanced base for MoE model training.

---

## 5. Station-by-Station Regime Distributions

The regime counts and percentages vary significantly across the 13 Washington stations due to localized environmental conditions:

![Regime Distribution by Station](regime_distribution_by_station.png)

### Station-Level Metrics (under Recalibrated Thresholds)
Below is the tabular summary of observations, average soil moisture, and regime counts per station:

| Station | Total Obs | Mean SM | Min SM | Max SM | Dry % (Count) | Trans % (Count) | Wet % (Count) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BeaverPass_WA_990** | 2,811 | 0.2773 | 0.0070 | 0.4040 | 13.0% (366) | 18.2% (511) | **68.8% (1934)** |
| **BurntMountain_WA** | 2,673 | 0.0406 | 0.0000 | 0.3280 | **95.5% (2552)** | 4.5% (120) | 0.0% (1) |
| **CayusePass_WA** | 3,179 | 0.1926 | 0.0000 | 0.4010 | 31.5% (1001) | 39.0% (1240) | 29.5% (938) |
| **Darrington** | 3,047 | 0.2192 | 0.0220 | 0.4440 | 29.2% (891) | 28.8% (878) | 41.9% (1278) |
| **HartsPass_WA_515** | 2,737 | 0.1123 | 0.0000 | 0.4480 | 63.5% (1738) | 18.3% (502) | 18.2% (497) |
| **MFNooksack_WA_1011** | 265 | 0.3341 | 0.0000 | 0.4060 | 13.2% (35) | 5.7% (15) | **81.1% (215)** |
| **MartenRidge_WA_999** | 2,981 | 0.2544 | 0.0000 | 0.3960 | 26.2% (780) | 14.4% (428) | **59.5% (1773)** |
| **Paradise_WA** | 3,258 | 0.1827 | 0.0000 | 0.4030 | 30.3% (987) | **52.4% (1707)** | 17.3% (564) |
| **Quinault** | 3,204 | 0.2146 | 0.0160 | 0.4310 | 19.2% (616) | **53.5% (1713)** | 27.3% (875) |
| **RainyPass_WA_711** | 3,265 | 0.1206 | 0.0000 | 0.3410 | 62.3% (2035) | 36.4% (1187) | 1.3% (43) |
| **SourdoughGulch_WA_985** | 3,097 | 0.2390 | 0.0320 | 0.3780 | 22.7% (704) | 25.2% (779) | **52.1% (1614)** |
| **Spokane** | 2,690 | 0.1683 | 0.0110 | 0.3590 | 43.5% (1171) | 27.7% (746) | 28.7% (773) |
| **Touchet_WA_824** | 1,568 | 0.1590 | 0.0000 | 0.3540 | 35.2% (552) | **57.6% (903)** | 7.2% (113) |

### Individual Target Histograms per Station
The small multiples grid below shows the soil moisture density distributions for each station, overlaid with the new recalibrated regime boundaries ($t_1$ and $t_2$):

![Soil Moisture Histogram Grid](soil_moisture_by_station_grid.png)

### Key Insights from Station-Level Analysis:
1. **High Heterogeneity**: Stations have very different soil moisture ranges. **BurntMountain** is extremely dry (mean = 0.0406, 95.5% of days in Dry), whereas **MFNooksack** (mean = 0.3341, 81.1% Wet) and **BeaverPass** (mean = 0.2773, 68.8% Wet) are highly wet. 
2. **Data Sparsity**: **MFNooksack** has only **265 observations** total due to missing target sensor data. It should be treated as a sparse target station during model training.
3. **Balanced Stations**: **Spokane**, **Darrington**, and **CayusePass** span a wide range of values and display bimodal or spread distributions across all three regimes.

---

## 6. Conclusions & Next Steps

### 1. Dataset Quality Verdict: **EXCELLENT**
* The `derived_8.1` dataset provides a **2.5x larger sample set** (34,775 rows).
* Under recalibrated thresholds, the **Wet regime has 5,612 training samples** (compared to 879 in `derived_8.0`). This solves the minority class undersupply issue highlighted in the handoff notes, providing a robust base to train a high-quality **Wet Specialist** expert model.

### 2. Threshold Verdict: **RECALIBRATE**
* Do **not** use the original thresholds ($t_1=0.20, t_2=0.313$), as they result in a heavily dry-skewed model.
* Adopt the **recalibrated thresholds ($t_1=0.143, t_2=0.269$)** to ensure balanced class sizes during MoE training.

### 3. Recommendations for MoE Modeling
* **Gating Design**: Because individual stations are highly skewed (e.g. BurntMountain is 95% Dry), station-level static features (latitude, longitude, elevation, HWSD clay/sand fractions) will be critical for the router to identify spatial regime shifts.
* **Evaluation**: The test split in `derived_8.1` is also larger (10,599 rows), allowing for a much cleaner out-of-sample evaluation of MoE routing strategies (e.g. comparing 3-class vs. 2-class gating as discussed in the handoff notes).
