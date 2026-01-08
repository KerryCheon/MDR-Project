<!-- All set to **Keep** as of Nov 13 — will revisit after professor feedback -->

# Table 1: Complete Candidate Feature Inventory

<!-- All set to KEEP as of Nov 13 — will revisit after professor feedback -->

| **#**&nbsp;&nbsp;&nbsp; | **Feature Name**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | **Abbreviation**&nbsp;&nbsp;            | **Source**  |
| :---------------------: | :--------------------------------------------------- | :-------------------------------------- | ----------- |
|            1            | First Difference                                     | $$\Delta x_t$$                          | Derived     |
|            2            | Second Difference                                    | $$\Delta^2 x_t$$                        | Derived     |
|            3            | n-Step Difference                                    | $$D_n(t)$$                              | Derived     |
|            4            | Temporal Gradient                                    | $$\text{grad}_t(k)$$                    | Derived     |
|            5            | Percent Change                                       | $$\text{pct}_t$$                        | Derived     |
|            6            | Moving Average                                       | $$\text{MA}_t(k)$$                      | Derived     |
|            7            | Exponential Moving Average                           | $$\text{EMA}_t$$                        | Derived     |
|            8            | Rolling Standard Deviation                           | $$\sigma_t(k)$$                         | Derived     |
|            9            | Rolling Coefficient of Variation                     | $$\text{CV}_t$$                         | Derived     |
|           10            | Rolling Minimum                                      | $$\min(x)$$                             | Derived     |
|           11            | Rolling Maximum                                      | $$\max(x)$$                             | Derived     |
|           12            | Rolling Range                                        | $$\text{range}_t$$                      | Derived     |
|           13            | Lag-6 Feature                                        | $$x_{t-1}$$                             | Derived     |
|           14            | Lag-12 Feature                                       | $$x_{t-2}$$                             | Derived     |
|           15            | Lag-30 Feature                                       | $$x_{t-5}$$                             | Derived     |
|           16            | Soil Moisture Memory Index                           | $$\text{SMM}_t$$                        | Derived     |
|           17            | Z-Score Anomaly                                      | $$z_t$$                                 | Derived     |
|           18            | Seasonal Anomaly                                     | $$\text{SA}_t$$                         | Derived     |
|           19            | Dominant Fourier Frequency                           | $$k^*$$                                 | Derived     |
|           20            | Spectral Entropy                                     | $$H$$                                   | Derived     |
|           21            | VV/VH Ratio (Sentinel-1)                             | $$\frac{\text{VV}}{\text{VH}}$$         | Sentinel-1  |
|           22            | Backscatter Difference (VV – VH)                     | $$\text{VV} - \text{VH}$$               | Sentinel-1  |
|           23            | Radar Coherence                                      | $$\gamma$$                              | Sentinel-1  |
|           24            | Radar Temporal Roughness Index                       | $$\text{RTI}$$                          | Sentinel-1  |
|           25            | Time Since Last Wetness Spike                        | $$\text{TSWS}_t$$                       | Sentinel-1  |
|           26            | NDVI Time Series                                     | $$\text{NDVI}$$                         | Sentinel-2  |
|           27            | NDMI (Moisture Index)                                | $$\text{NDMI}$$                         | Sentinel-2  |
|           28            | MSI (Moisture Stress Index)                          | $$\text{MSI}$$                          | Sentinel-2  |
|           29            | SWIR Reflectance Temporal Curve                      | $$\text{SWIR}(t)$$                      | Sentinel-2  |
|           30            | Rainfall Accumulation                                | $$R_k(t)$$                              | Precip Data |
|           31            | Days Since Last Rain                                 | $$\text{DSLR}_t$$                       | Precip Data |
|           32            | Antecedent Precipitation Index (API)                 | $$\text{API}_t$$                        | Precip Data |
|           33            | Temperature Anomaly (LST)                            | $$\text{TA}_t$$                         | MODIS / S2  |
|           34            | Radar–Optical Lag Correlation                        | $$\text{Corr}(\text{VV}, \text{NDVI})$$ | S1 + S2     |
|           35            | Temperature–Moisture Temporal Coupling               | $$C_{TM}(k)$$                           | S1 + LST    |

---

# Table 2. Feature Selection Decisions (Keep vs Bench)

| **#** | **Feature Name**                       |  **Decision**   | **Primary Reason**                     |
| :---: | -------------------------------------- | :-------------: | -------------------------------------- |
|   1   | First Difference                       |      Bench      | Redundant with temporal gradient       |
|   2   | Second Difference                      |      Bench      | High variance, limited added signal    |
|   3   | n-Step Difference                      |      Bench      | Superseded by gradient and memory      |
|   4   | Temporal Gradient                      |    **Keep**     | Best short-term change descriptor      |
|   5   | Percent Change                         |      Bench      | Scale issues, redundant with gradient  |
|   6   | Moving Average                         |      Bench      | Trend captured via memory + EMA        |
|   7   | Exponential Moving Average             | **Keep** (Opt.) | Noise suppression with fast response   |
|   8   | Rolling Standard Deviation             |    **Keep**     | Captures local volatility              |
|   9   | Rolling Coefficient of Variation       |      Bench      | Redundant with rolling std             |
|  10   | Rolling Minimum                        |      Bench      | Encoded indirectly by memory + forcing |
|  11   | Rolling Maximum                        |      Bench      | Encoded indirectly by memory + forcing |
|  12   | Rolling Range                          |      Bench      | Redundant volatility measure           |
|  13   | Lag-6 Feature                          |      Bench      | Replaced by memory index               |
|  14   | Lag-12 Feature                         |      Bench      | Replaced by memory index               |
|  15   | Lag-30 Feature                         |      Bench      | Replaced by memory index               |
|  16   | Soil Moisture Memory Index             |    **Keep**     | Core soil persistence signal           |
|  17   | Z-Score Anomaly                        |      Bench      | Redundant once seasonality removed     |
|  18   | Seasonal Anomaly                       |    **Keep**     | Removes climatological bias            |
|  19   | Dominant Fourier Frequency             |      Bench      | Data-hungry, second-phase feature      |
|  20   | Spectral Entropy                       |      Bench      | High variance, weak interpretability   |
|  21   | VV/VH Ratio (Sentinel-1)               |    **Keep**     | Strong direct moisture sensitivity     |
|  22   | Backscatter Difference (VV – VH)       |      Bench      | Redundant with ratio                   |
|  23   | Radar Coherence                        |    **Keep**     | Structural surface change indicator    |
|  24   | Radar Temporal Roughness Index         |      Bench      | Overlaps with rolling std              |
|  25   | Time Since Last Wetness Spike          |      Bench      | Secondary timing feature               |
|  26   | NDVI Time Series                       |      Bench      | Indirect moisture proxy                |
|  27   | NDMI (Moisture Index)                  |    **Keep**     | Direct vegetation water signal         |
|  28   | MSI (Moisture Stress Index)            |      Bench      | Highly correlated with NDMI            |
|  29   | SWIR Reflectance Temporal Curve        |      Bench      | Redundant optical moisture info        |
|  30   | Rainfall Accumulation                  |      Bench      | Superseded by API                      |
|  31   | Days Since Last Rain                   |    **Keep**     | Encodes dry-down phase                 |
|  32   | Antecedent Precipitation Index (API)   |    **Keep**     | Core forcing variable                  |
|  33   | Temperature Anomaly (LST)              |    **Keep**     | Orthogonal dry-down signal             |
|  34   | Radar–Optical Lag Correlation          |      Bench      | Fragile, second-order relationship     |
|  35   | Temperature–Moisture Temporal Coupling |      Bench      | Second-phase interaction feature       |

---

# Table 3: Final Refined Feature Set Used for Modeling

| **#** | **Feature Name**                      | **Abbreviation**                | **Source**  |
| :---: | ------------------------------------- | ------------------------------- | ----------- |
|   1   | Temporal Gradient                     | $$\text{grad}_t(k)$$            | Derived     |
|   2   | Rolling Standard Deviation            | $$\sigma_t(k)$$                 | Derived     |
|   3   | Soil Moisture Memory Index            | $$\text{SMM}_t$$                | Derived     |
|   4   | Seasonal Anomaly                      | $$\text{SA}_t$$                 | Derived     |
|   5   | VV/VH Ratio (Sentinel-1)              | $$\frac{\text{VV}}{\text{VH}}$$ | Sentinel-1  |
|   6   | Radar Coherence (Sentinel-1)          | $$\gamma$$                      | Sentinel-1  |
|   7   | NDMI (Moisture Index)                 | $$\text{NDMI}$$                 | Sentinel-2  |
|   8   | Antecedent Precipitation Index        | $$\text{API}_t$$                | Precip Data |
|   9   | Days Since Last Rain                  | $$\text{DSLR}_t$$               | Precip Data |
|  10   | Temperature Anomaly (LST)             | $$\text{TA}_t$$                 | MODIS / S2  |
|  11   | Exponential Moving Average (Optional) | $$\text{EMA}_t$$                | Derived     |

## Final Set of Features

_Pulled from the original document that defines the features_

### 1. Temporal Gradient (Rate of Change)

> **Note:** `grad` denotes gradient

**Formula**:
$$\text{grad}_t(k) = \frac{x_t - x_{t-k}}{k}$$

> **Description**: Average slope across a time window.

> **Why useful**: Moisture doesn’t just go up and down, it changes at different speeds depending on weather, soil type, and vegetation. The gradient shows whether things are drying out slowly, dropping fast, or steadily recovering after rain. It gives the model a sense of momentum instead of just direction.

---

### 2. Rolling Standard Deviation

**Formula**:
$$\sigma_t(k)=\sqrt{\frac{1}{k}\sum_{i=0}^{k-1}(x_{t-i}-\text{MA}_t)^2}$$

> **Description**: Short-term variability.

> **Why useful**: When the soil moisture signal starts jumping around, it usually means the landscape is going through quick changes, storms, irrigation, runoff, rapid drying, you name it. Rolling standard deviation picks up those chaotic stretches that a simple average would completely smooth out.

---

### 3. Soil Moisture Memory Index

**Formula**:

$$
\text{SMM}_t = \sum_{i=1}^{n} \alpha^i x_{t-i} \qquad (0 < \alpha < 1)
$$

> **Description**: Exponentially weighted sum of past moisture-related observations.

> **Why useful**: Soil doesn’t reset every day, whatever happened last week or even last month still affects how fast it dries or how much new rain it can absorb. This metric gives you a smooth “memory score” of past moisture, which models love because it fills in the gaps between noisy individual measurements.

---

### 4. Seasonal Anomaly

**Formula**:

$$\text{SA}_t = x_t - \mu_{\text{month}(t)}$$

> **Description**: Removes seasonal vegetation/moisture cycles.

> **Why useful**: Seasonal patterns can hide what’s really going on with moisture. By stripping out the “expected” behavior for that time of year, you can spot when the soil is genuinely wetter or drier than normal. It separates real events from just regular seasonal swings.

---

### 5. VV/VH Ratio (Sentinel-1)

**Formula**:
$$R_t = \frac{VV_t}{VH_t}$$

> **Description**: Normalized radar polarization ratio.

> **Why useful**: This ratio is one of the cleanest ways to see how the ground and vegetation are interacting with moisture. When the soil gets wet, VV and VH don’t change at the same pace, the ratio shifts fast and noticeably. It gives you a quick, normalized signal of whether the scene is getting wetter, drying out, or going through a vegetation-driven change.

---

### 6. Radar Coherence (Sentinel-1)

**Formula**:
$$\gamma = \frac{|\sum S_t S^*_{t+\Delta}|}{\sqrt{\sum|S_t|^2 \sum|S_{t+\Delta}|^2}}$$

> **Description**: Measures temporal phase stability.

> **Why useful**: Coherence is like radar’s way of telling you whether the ground stayed “the same” between two passes. When the soil suddenly gets wet, the surface changes enough that the phase relationship falls apart. A big coherence drop is basically radar shouting, “Something just changed down there,” and that something is usually moisture.

---

### 7. NDMI (Moisture Index)

**Formula**:
$$\text{NDMI} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}}$$

> **Description**: Direct proxy for vegetation water content.

> **Why useful**: When vegetation is stressed from lack of water, it reflects way more in SWIR and less in NIR. MSI picks up that shift immediately. If this ratio starts climbing, it’s a pretty clear sign the canopy is drying out faster than normal, which ties directly back to soil moisture conditions.

---

### 8. Antecedent Precipitation Index (API)

**Formula**:

$$
\text{API}_t = P_t + kP_{t-1} + k^2P_{t-2} + \dots \qquad (0 < k < 1)
$$

> **Description**: Exponentially decaying accumulation of historic rainfall.

> **Why useful**: Rain doesn’t just disappear after it hits the ground, the soil hangs onto it for days or even weeks. API gives us a clean way to quantify how “soaked” the system still is, even if it hasn’t rained recently. It captures that lingering influence of past storms that simple rainfall totals completely miss.

---

### 9. Days Since Last Rain

**Formula**:
$$\text{DSLR}_t = t - t_{\text{rain(last)}}$$

> **Description**: Time since wetting event.

> **Why useful**: Soil follows a pretty simple rule after rain: the longer it's been dry, the drier it gets. This feature captures that whole story in one number. It tells the model whether we’re looking at soil that’s still fresh from a storm or well into a long dry spell, which makes a huge difference for predicting moisture.

---

### 10. Temperature Anomaly (LST)

**Formula**:
$$\text{TA}_t = \text{LST}_t - \mu_{\text{LST}}$$

> **Description**: Land surface temperature deviation.

> **Why useful**: Soil temperature swings tell you a lot about moisture without actually measuring it. When the land heats up way faster or slower than it usually does, it’s usually because the soil is unusually dry or unusually wet. This anomaly basically highlights those “something’s off” days that line up with real moisture shifts.

---

### 11. Exponential Moving Average

**Formula**:
$$\text{EMA}_t = \alpha x_t + (1-\alpha) \text{EMA}_{t-1}$$

> **Description**: Recency-weighted trend.

> **Why useful**: Radar data jumps around a lot from pass to pass, even when nothing dramatic is happening on the ground. EMA smooths out that noise without losing the real trend. It reacts fast enough to catch new wetting events but slow enough to ignore random fluctuations.

---

<!-- This section should be removed after the features have been updated

- Kerry: #1-12
- Jacob: #13-24
- Daniel: #25-35

-->
