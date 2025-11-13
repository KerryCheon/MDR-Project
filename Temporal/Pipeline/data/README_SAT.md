# List of Satellite Features (Temporal)

**Jakob Balkovec & Kerry Cheon**
**Thu Nov 13th**

---

## README

This document compiles the temporal features we believe have the strongest potential impact on soil-moisture modeling performance. Each feature is presented with two short explanations to help interpret its role:

**Formula**: The formula to calculate the value of the mentioned feature, given the data obtained from the satellite.

> **Description:** A high-level overview of what the feature represents.

> **Why Useful:** A plain-language explanation of why the feature matters and how it contributes to understanding or predicting soil moisture.

## Table of Contents

- [1. First Difference](#1-first-difference)
- [2. Second Difference](#2-second-difference)
- [3. n-Step Difference](#3-n-step-difference)
- [4. Temporal Gradient (Rate of Change)](#4-temporal-gradient-rate-of-change)
- [5. Percent Change](#5-percent-change)
- [6. Moving Average](#6-moving-average)
- [7. Exponential Moving Average](#7-exponential-moving-average)
- [8. Rolling Standard Deviation](#8-rolling-standard-deviation)
- [9. Rolling Coefficient of Variation](#9-rolling-coefficient-of-variation)
- [10. Rolling Minimum](#10-rolling-minimum)
- [11. Rolling Maximum](#11-rolling-maximum)
- [12. Z-Score Anomaly](#12-z-score-anomaly)
- [13. Seasonal Anomaly](#13-seasonal-anomaly)
- [14. Lag-1 Feature](#14-lag-1-feature)
- [15. Lag-7 Feature](#15-lag-7-feature)
- [16. Lag-30 Feature](#16-lag-30-feature)
- [17. Rolling Range](#17-rolling-range)
- [18. Dominant Fourier Frequency](#18-dominant-fourier-frequency)
- [19. Spectral Entropy](#19-spectral-entropy)
- [20. VV/VH Ratio (Sentinel-1)](#20-vv-vh-ratio-sentinel-1)
- [21. Backscatter Difference (VV – VH)](#21-backscatter-difference-vv--vh)
- [22. Radar Coherence (Sentinel-1)](#22-radar-coherence-sentinel-1)
- [23. NDVI Time Series](#23-ndvi-time-series)
- [24. NDMI (Moisture Index)](#24-ndmi-moisture-index)
- [25. MSI (Moisture Stress Index)](#25-msi-moisture-stress-index)
- [26. SWIR Reflectance Temporal Curve](#26-swir-reflectance-temporal-curve)
- [27. Rainfall Accumulation](#27-rainfall-accumulation-3-7-30-days)
- [28. Days Since Last Rain](#28-days-since-last-rain)
- [29. Temperature Anomaly (LST)](#29-temperature-anomaly-lst)
- [30. Radar–Optical Lag Correlation](#30-radar–optical-lag-correlation)
- [31. Time Since Last Wetness Spike (Radar)](#31-time-since-last-wetness-spike-radar)
- [32. Soil Moisture Memory Index](#32-soil-moisture-memory-index)
- [33. Radar Temporal Roughness Index](#33-radar-temporal-roughness-index)
- [34. Temperature–Moisture Temporal Coupling](#34-temperaturemoisture-temporal-coupling)
- [35. Antecedent Precipitation Index (API)](#35-antecedent-precipitation-index-api)

## Features

### 1. First Difference

**Formula**:
$$\Delta x_t = x_t - x_{t-1}$$

> **Description**: Measures immediate change between consecutive observations.

> **Why useful**: This is the simplest way to see whether things are getting wetter or drier right now. When a storm hits or the soil starts drying fast, the first difference jumps immediately. It’s the quickest, cleanest signal of short-term moisture change.

### 2. Second Difference

**Formula**:
$$\Delta^2 x_t = (x_t - x_{t-1}) - (x_{t-1} - x_{t-2})$$

> **Description**: Measures acceleration of change.

> **Why useful**: This catches those moments when the soil isn’t just changing, it’s changing _faster_ than before. Big second differences usually show up during sudden wetting events or sharp dry-downs, making it a great detector for rapid, meaningful shifts in soil conditions.

### 3. n-Step Difference

**Formula**:
$$D_n(t) = x_t - x_{t-n}$$

> **Description**: Long-term temporal contrast.

> **Why useful**: Looking back several steps gives a sense of how much the system has changed over a longer stretch, not just day to day. It’s great for spotting bigger seasonal swings, like transitions from wet spring conditions into dry summer phases.

### 4. Temporal Gradient (Rate of Change)

> **Note:** `grad` denotes gradient

**Formula**:
$$\text{grad}_t(k) = \frac{x_t - x_{t-k}}{k}$$

> **Description**: Average slope across a time window.

> **Why useful**: Moisture doesn’t just go up and down, it changes at different speeds depending on weather, soil type, and vegetation. The gradient shows whether things are drying out slowly, dropping fast, or steadily recovering after rain. It gives the model a sense of momentum instead of just direction.

### 5. Percent Change

**Formula**:
$$\text{pct}_t = \frac{x_t - x_{t-1}}{x_{t-1}} \times 100$$

> **Description**: Normalized rate of change.

> **Why useful**: Some places naturally have higher or lower moisture values, so raw changes don’t always mean much. Percent change levels the playing field. It shows how big the shift really is relative to what’s normal for that spot, which helps catch meaningful jumps that absolute differences might hide..

### 6. Moving Average

**Formula**:
$$\text{MA}_t(k) = \frac{1}{k} \sum_{i=0}^{k-1} x\_{t-i}$$

> **Description**: Smooths episodic noise.

> **Why useful**: Soil moisture doesn’t change instantly, it shifts slowly over days. A moving average helps you see that slow drift by ironing out random bumps in the data. It gives the model a clearer picture of the underlying trend instead of getting distracted by every little fluctuation.

### 7. Exponential Moving Average

**Formula**:
$$\text{EMA}_t = \alpha x_t + (1-\alpha) \text{EMA}_{t-1}$$

> **Description**: Recency-weighted trend.

> **Why useful**: Radar data jumps around a lot from pass to pass, even when nothing dramatic is happening on the ground. EMA smooths out that noise without losing the real trend. It reacts fast enough to catch new wetting events but slow enough to ignore random fluctuations.

### 8. Rolling Standard Deviation

**Formula**:
$$\sigma_t(k)=\sqrt{\frac{1}{k}\sum_{i=0}^{k-1}(x_{t-i}-\text{MA}_t)^2}$$

> **Description**: Short-term variability.

> **Why useful**: When the soil moisture signal starts jumping around, it usually means the landscape is going through quick changes, storms, irrigation, runoff, rapid drying, you name it. Rolling standard deviation picks up those chaotic stretches that a simple average would completely smooth out.

### 9. Rolling Coefficient of Variation

**Formula**:
$$CV_t = \frac{\sigma_t}{MA_t}$$

> **Description**: Standard deviation normalized by mean.

> **Why useful**: This tells you how “jittery” the signal is compared to its average level. Two places might have the same standard deviation, but if one has a tiny mean and the other a big one, the impact is totally different. CV exposes that difference. It highlights spots where the soil moisture is bouncing around a lot relative to its typical state, which usually means the system is sensitive to rain, heat, or vegetation shifts.

### 10. Rolling Minimum

**Formula**:
$$\min_{i \in [t-k,\, t]} x_i$$

> **Description**: Lowest value in window.

> **Why useful**: The lowest point in the window gives you a sense of the soil’s baseline dryness. If the minimum is really low, it means the area has hit true dry conditions recently, which shapes how quickly it can re-wet and how much new rainfall it can absorb.

### 11. Rolling Maximum

**Formula**:
$$\max_{i \in [t-k,\, t]} x_i$$

> **Description**: Highest value in window.

> **Why useful**: Knowing the wettest point in the recent past helps anchor where the soil sits on its drying curve. If the recent max was very high, the soil likely started out saturated and is still working its way down. If the max was low, the whole period has been dry. It’s a quick way to judge how “topped up” the system has been lately.

### 12. Z-Score Anomaly

**Formula**:
$$z_t = \frac{x_t - \mu}{\sigma}$$

> **Description**: Measures deviation from expected climatological behavior.

> **Why useful**: This tells you how unusual today’s moisture signal really is. A big positive or negative z-score means the soil isn’t behaving like it normally does, it’s either way too wet or way too dry. Models benefit from that context because it highlights the standout moments that actually drive big changes on the ground.

### 13. Seasonal Anomaly

**Formula**:

$$\text{SA}_t = x_t - \mu_{\text{month}(t)}$$

> **Description**: Removes seasonal vegetation/moisture cycles.

> **Why useful**: Seasonal patterns can hide what’s really going on with moisture. By stripping out the “expected” behavior for that time of year, you can spot when the soil is genuinely wetter or drier than normal. It separates real events from just regular seasonal swings.

### 14. Lag-1 Feature

**Formula**:
$$x_{t-1}$$

> **Description**: Previous timestep value.

> **Why useful**: Soil moisture doesn’t swing wildly from one observation to the next, it usually changes gradually. Yesterday’s value is often the single best clue about today’s, and giving the model that immediate context makes its predictions way more grounded.

### 15. Lag-7 Feature

**Formula**:
$$x_{t-7}$$

> **Description**: Weekly memory.

> **Why useful**: Soil moisture doesn’t usually overhaul itself in just a few days. A weekly lag captures that medium-term “echo” of past conditions, helping the model understand whether today’s moisture is building off a still-wet week or recovering from a long dry stretch.

### 16. Lag-30 Feature

**Formula**:
$$x_{t-30}$$

> **Description**: Monthly memory.

> **Why useful**: Conditions a month ago still matter, especially in systems where moisture builds up or drains slowly. This gives the model a snapshot of the broader seasonal backdrop, whether the landscape is heading into a wet phase, a dry phase, or sitting somewhere in between.

### 17. Rolling Range

**Formula**:
$$\text{range}_t = \max(x_{t-k:t}) - \min(x_{t-k:t})$$

> **Description**: Spread over time window.

> **Why useful**: When the highs and lows in a window are far apart, it usually means the soil has been swinging between wet and dry pretty aggressively. That kind of variability tells the model a lot about how reactive the landscape is, whether it soaks up water and holds it, or dries out fast between events.

### 18. Dominant Fourier Frequency

**Formula**:
$$k^* = \underset{k}{\arg\max}\; |X(k)|$$

> **Description**: Strength of seasonal or sub-seasonal cycle.

> **Why useful**: Soil moisture tends to follow repeating rhythms, think daily heating, weekly weather patterns, or full seasonal cycles. The dominant frequency tells you which of those rhythms is actually driving the signal. If one frequency pops out, you know the moisture pattern isn’t random; it’s following a predictable beat that the model can learn from.

### 19. Spectral Entropy

**Formula**:
$$H = -\sum p_k \log p_k$$

> **Description**: Measures unpredictability of temporal signal.

> **Why useful**: Entropy tells you how “messy” or unpredictable the moisture signal is. Stable soils with slow, steady drying have low entropy, while areas hit by frequent storms, irrigation, or rapid drying cycles look much more chaotic. It’s a quick way to spot spots where moisture behavior refuses to follow a neat pattern.

### 20. VV/VH Ratio (Sentinel-1)

**Formula**:
$$R_t = \frac{VV_t}{VH_t}$$

> **Description**: Normalized radar polarization ratio.

> **Why useful**: This ratio is one of the cleanest ways to see how the ground and vegetation are interacting with moisture. When the soil gets wet, VV and VH don’t change at the same pace, the ratio shifts fast and noticeably. It gives you a quick, normalized signal of whether the scene is getting wetter, drying out, or going through a vegetation-driven change.

### 21. Backscatter Difference (VV – VH)

**Formula**:
$$D_t = VV_t - VH_t$$

> **Description**: Polarization separation metric.

> **Why useful**: VV and VH react differently depending on whether the change is coming from the soil or the vegetation above it. Taking the difference helps tease those two influences apart. If the gap suddenly widens or shrinks, it usually points to real moisture changes rather than just plants getting thicker or thinner.

### 22. Radar Coherence (Sentinel-1)

**Formula**:
$$\gamma = \frac{|\sum S_t S^*_{t+\Delta}|}{\sqrt{\sum|S_t|^2 \sum|S_{t+\Delta}|^2}}$$

> **Description**: Measures temporal phase stability.

> **Why useful**: Coherence is like radar’s way of telling you whether the ground stayed “the same” between two passes. When the soil suddenly gets wet, the surface changes enough that the phase relationship falls apart. A big coherence drop is basically radar shouting, “Something just changed down there,” and that something is usually moisture.

### 23. NDVI Time Series

**Formula**:
$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$

> **Description**: Vegetation greenness.

> **Why useful**: Plants act like a slow-motion indicator of soil moisture. They don’t instantly green up when the soil gets wet, but over the next few days to weeks you’ll see the canopy react. Tracking NDVI through time tells you whether the vegetation is bouncing back, holding steady, or starting to stress, all of which are tied to what's happening in the soil underneath.

### 24. NDMI (Moisture Index)

**Formula**:
$$\text{NDMI} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}}$$

> **Description**: Direct proxy for vegetation water content.

> **Why useful**: When vegetation is stressed from lack of water, it reflects way more in SWIR and less in NIR. MSI picks up that shift immediately. If this ratio starts climbing, it’s a pretty clear sign the canopy is drying out faster than normal, which ties directly back to soil moisture conditions.

### 25. MSI (Moisture Stress Index)

**Formula**:
$$\text{MSI} = \frac{\text{SWIR}}{\text{NIR}}$$

> **Description**: Ratio sensitive to drought stress.
> **Why useful**: Increases sharply during moisture deficit.

### 26. SWIR Reflectance Temporal Curve

> **Description**: B11/B12 time-series behavior.

> **Why useful**: SWIR reacts strongly to water content, especially when there isn’t much vegetation in the way. Watching how these bands change over time gives you a clean signal of the soil slowly drying out or getting re-wet. It’s one of the most direct optical clues about bare-soil moisture you can get.

### 27. Rainfall Accumulation (3, 7, 30 days)

**Formula**:
$$R_k(t) = \sum_{i=0}^{k} \text{rain}_{t-i}$$

> **Description**: Recent precipitation totals.

> **Why useful**: Rain is the main fuel for soil moisture, and short-term totals tell you how much water actually made it into the system. Using different windows (3, 7, 30 days) gives you a feel for quick hits, weekly patterns, and longer soaking periods. Models latch onto these because they explain most of the “why is the soil wet right now?” behavior.

### 28. Days Since Last Rain

**Formula**:
$$\text{DSLR}_t = t - t_{\text{rain(last)}}$$

> **Description**: Time since wetting event.

> **Why useful**: Soil follows a pretty simple rule after rain: the longer it's been dry, the drier it gets. This feature captures that whole story in one number. It tells the model whether we’re looking at soil that’s still fresh from a storm or well into a long dry spell, which makes a huge difference for predicting moisture.

### 29. Temperature Anomaly (LST)

**Formula**:
$$\text{TA}_t = \text{LST}_t - \mu_{\text{LST}}$$

> **Description**: Land surface temperature deviation.

> **Why useful**: Soil temperature swings tell you a lot about moisture without actually measuring it. When the land heats up way faster or slower than it usually does, it’s usually because the soil is unusually dry or unusually wet. This anomaly basically highlights those “something’s off” days that line up with real moisture shifts.

### 30. Radar–Optical Lag Correlation

**Formula**:
$$\text{Corr}(\text{VV}_{t-k}, \text{NDVI}_t)$$

> **Description**: Temporal lag between moisture and vegetation response.

> **Why useful**: Plants don’t respond instantly when the soil gets wet, there’s always a delay before the canopy greens up or stress signals fade. This feature captures that “response gap.” When the lagged radar moisture signal lines up well with today’s NDVI, it tells you the vegetation is tightly tuned to soil conditions rather than just following seasonal patterns.

### 31. Time Since Last Wetness Spike (Radar)

**Formula**:

$$
\text{TSWS}_t = t - t_{\text{max}(\Delta \text{VV})}
$$

> **Description**: Measures how long it has been since the last major increase in radar backscatter caused by a wetting event.

> **Why useful**: After a big wetting event, the soil goes through a pretty predictable “cooling-off” period as it dries. Knowing how long it's been since the last spike tells you exactly where you are in that cycle. It’s basically a timer that tracks whether the ground is still freshly wet or well into its drying phase.

### 32. Soil Moisture Memory Index

**Formula**:

$$
\text{SMM}_t = \sum_{i=1}^{n} \alpha^i x_{t-i} \qquad (0 < \alpha < 1)
$$

> **Description**: Exponentially weighted sum of past moisture-related observations.

> **Why useful**: Soil doesn’t reset every day, whatever happened last week or even last month still affects how fast it dries or how much new rain it can absorb. This metric gives you a smooth “memory score” of past moisture, which models love because it fills in the gaps between noisy individual measurements.

### 33. Radar Temporal Roughness Index

**Formula**:

$$
\text{RTI} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (VV_i - VV_{i-1})^2}
$$

> **Description**: Measures the temporal roughness or volatility in radar backscatter.

> **Why useful**: When radar values are jumping around a lot, something on the ground is changing quickly. Sometimes that’s vegetation, but often it’s moisture shifting after rain or during a fast dry-down. RTI helps you spot those “chaotic” periods that simple averages totally smooth over.

### 34. Temperature–Moisture Temporal Coupling

**Formula**:

$$
C_{TM}(k) = \text{Corr}(\text{LST}_{t-k},\, \text{VV}_t)
$$

> **Description**: Cross-correlation between lagged land surface temperature and radar backscatter.

> **Why useful**: Soil basically broadcasts its moisture level through temperature. When it’s dry, it heats up fast; when it’s wet, it stays cool longer. By looking at how temperature and radar moisture signals line up over time, you get a clearer sense of whether the land is actually drying out or just going through normal daily swings.

### 35. Antecedent Precipitation Index (API)

**Formula**:

$$
\text{API}_t = P_t + kP_{t-1} + k^2P_{t-2} + \dots \qquad (0 < k < 1)
$$

> **Description**: Exponentially decaying accumulation of historic rainfall.

> **Why useful**: Rain doesn’t just disappear after it hits the ground, the soil hangs onto it for days or even weeks. API gives you a clean way to quantify how “soaked” the system still is, even if it hasn’t rained recently. It captures that lingering influence of past storms that simple rainfall totals completely miss.

---

**End of Document**
