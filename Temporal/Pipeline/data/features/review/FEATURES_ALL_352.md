# Temporal Features Reference: All 352 Features

**Authors:** Jakob Balkovec, Kerry Cheon
**Last updated:** Fri Jan 23

A compact, professional reference for all **352 non-ID feature columns** in
`Temporal/Pipeline/data/splits/derived_all/train_derived_all.csv`.

## Overview

- Data is **sorted by date** prior to feature generation
- All features are produced by the **Temporal derived feature pipeline**

## Source of truth

Feature definitions and formulas follow the generator code exactly:

- `Temporal/Pipeline/data/splits/derived_all/make_derived_all_split.py`
- `Temporal/Pipeline/data/splits/derived_all/utils/derived_features_all_math.py`

This document is intended as a quick reference and sanity check.
If anything here conflicts with the implementation, **the code is authoritative**.

## Conventions

- $t$ is the current day, $x_t$ is the series value at $t$, and $k$ is a lag in observations (kobs).
- Rolling windows use the last $w$ observations ending at $t$ unless noted as past-only.
- $\epsilon = 10^{-6}$ (to avoid divide-by-zero).

## Core inputs

1. `longitude`: Station longitude in degrees.
2. `latitude`: Station latitude in degrees.
3. `elev`: Elevation (m).
4. `slope`: Terrain slope (degrees).
5. `aspect`: Terrain aspect (degrees).
6. `DOY`: Day of year (1-366).
7. `precip_mm`: Daily precipitation in mm.
8. `s1_vv`: Sentinel-1 VV backscatter (dB).
9. `s1_vh`: Sentinel-1 VH backscatter (dB).
10. `s2_b4`: Sentinel-2 band 4 (red).
11. `s2_b8`: Sentinel-2 band 8 (NIR).
12. `s2_b11`: Sentinel-2 band 11 (SWIR1).
13. `s2_b12`: Sentinel-2 band 12 (SWIR2).
14. `LST_modis`: MODIS land surface temperature.
15. `soil_moisture_5cm`: Target soil moisture at 5 cm (model label).

## Optical indices (`F_family`)

1. **`F_NDVI`**
   Formula:
   $$\frac{\mathrm{NIR} - \mathrm{RED}}{\mathrm{NIR} + \mathrm{RED} + \epsilon}$$
   Vegetation vigor proxy from Sentinel-2 (b8/b4).

2. **`F_NDMI`**
   Formula:
   $$\frac{\mathrm{NIR} - \mathrm{SWIR1}}{\mathrm{NIR} + \mathrm{SWIR1} + \epsilon}$$
   Canopy/soil moisture proxy from Sentinel-2 (b8/b11).

3. **`F_MSI`**
   Formula:
   $$\frac{\mathrm{SWIR1}}{\mathrm{NIR} + \epsilon}$$
   Moisture stress index (higher can mean drier).

## Radar indices (`E_family`)

1. **`E_SAR_ratio`**
   Formula:
   $$\frac{VV}{VH + \epsilon}$$
   Radar polarization ratio, roughness/moisture-sensitive.

2. **`E_SAR_diff`**
   Formula:
   $$VV - VH$$
   Radar polarization difference, roughness/moisture contrast.

3. **`E_rough_s1_vv_kobs7`**
   Formula:
   $$\mathrm{mean}(|\Delta VV|_{t-w:t-1}), \quad w = 7 \; \text{(past-only)}$$
   Radar roughness proxy.

4. **`E_rough_s1_vh_kobs7`**
   Formula:
   $$\mathrm{mean}(|\Delta VH|_{t-w:t-1}), \quad w = 7 \; \text{(past-only)}$$
   Radar roughness proxy.

5. **`E_rough_s1_vv_kobs14`**
   Formula:
   $$\mathrm{mean}(|\Delta VV|_{t-w:t-1}), \quad w = 14 \; \text{(past-only)}$$
   Radar roughness proxy.

6. **`E_rough_s1_vh_kobs14`**
   Formula:
   $$\mathrm{mean}(|\Delta VH|_{t-w:t-1}), \quad w = 14 \; \text{(past-only)}$$
   Radar roughness proxy.

## Meteorology (`G_family`)

1. **`G_API`**
   Formula:
   $$\mathrm{API}_t = P_t + 0.9\,\mathrm{API}_{t-1}$$
   Antecedent precipitation index (rain memory).

2. **`G_DSLR`**
   Definition:
   $$\Delta t \;\text{since last day with}\; P_t \ge 0.5\ \text{mm}$$
   Days since last rain event (NaN until first rain).

3. **`G_rain_sum_3d`**
   Formula:
   $$\sum_{\tau \in [t-3,\,t]} P_\tau$$
   Short-term rainfall accumulation over calendar days.

4. **`G_rain_sum_7d`**
   Formula:
   $$\sum_{\tau \in [t-7,\,t]} P_\tau$$
   Short-term rainfall accumulation over calendar days.

5. **`G_rain_sum_30d`**
   Formula:
   $$\sum_{\tau \in [t-30,\,t]} P_\tau$$
   Short-term rainfall accumulation over calendar days.

6. **`G_DSLR_isnan`**
   Definition:
   $$\mathbb{1}[\mathrm{G\_DSLR}\ \text{is NaN}]$$
   Flag indicating DSLR is undefined (no prior rain).

## Dynamics / deltas (`A_family`)

1. **`A_d_G_API_kobs1`**
   Formula:
   $$x_t - x_{t-1} \quad \text{where } x_t \text{ is } \mathrm{G\_API}$$
   Captures 1-step change.

2. **`A_d_G_API_kobs2`**
   Formula:
   $$x_t - x_{t-2} \quad \text{where } x_t \text{ is } \mathrm{G\_API}$$
   Captures 2-step change.

3. **`A_d_G_API_kobs5`**
   Formula:
   $$x_t - x_{t-5} \quad \text{where } x_t \text{ is } \mathrm{G\_API}$$
   Captures 5-step change.

4. **`A_d_G_API_kobs7`**
   Formula:
   $$x_t - x_{t-7} \quad \text{where } x_t \text{ is } \mathrm{G\_API}$$
   Captures 7-step change.

5. **`A_d_G_API_kobs14`**
   Formula:
   $$x_t - x_{t-14} \quad \text{where } x_t \text{ is } \mathrm{G\_API}$$
   Captures 14-step change.

6. **`A_d_G_API_kobs30`**
   Formula:
   $$x_t - x_{t-30} \quad \text{where } x_t \text{ is } \mathrm{G\_API}$$
   Captures 30-step change.

7. **`A_grad_G_API_kobs7`**
   Formula:
   $$\frac{x_t - x_{t-7}}{7} \quad \text{where } x_t \text{ is } \mathrm{G\_API}$$
   Average slope over 7 observations.

8. **`A_grad_G_API_kobs14`**
   Formula:
   $$\frac{x_t - x_{t-14}}{14} \quad \text{where } x_t \text{ is } \mathrm{G\_API}$$
   Average slope over 14 observations.

9. **`A_grad_G_API_kobs30`**
   Formula:
   $$\frac{x_t - x_{t-30}}{30} \quad \text{where } x_t \text{ is } \mathrm{G\_API}$$
   Average slope over 30 observations.

10. **`A_pct_G_API`**
    Formula:
    $$\frac{x_t - x_{t-1}}{x_{t-1} + \epsilon}, \quad x_t = \mathrm{G\_API}$$
    One-step percent change.

11. **`A_d_F_NDMI_kobs1`**
    Formula:
    $$x_t - x_{t-1} \quad \text{where } x_t \text{ is } \mathrm{F\_NDMI}$$
    Captures 1-step change.

12. **`A_d_F_NDMI_kobs2`**
    Formula:
    $$x_t - x_{t-2} \quad \text{where } x_t \text{ is } \mathrm{F\_NDMI}$$
    Captures 2-step change.

13. **`A_d_F_NDMI_kobs5`**
    Formula:
    $$x_t - x_{t-5} \quad \text{where } x_t \text{ is } \mathrm{F\_NDMI}$$
    Captures 5-step change.

14. **`A_d_F_NDMI_kobs7`**
    Formula:
    $$x_t - x_{t-7} \quad \text{where } x_t \text{ is } \mathrm{F\_NDMI}$$
    Captures 7-step change.

15. **`A_d_F_NDMI_kobs14`**
    Formula:
    $$x_t - x_{t-14} \quad \text{where } x_t \text{ is } \mathrm{F\_NDMI}$$
    Captures 14-step change.

16. **`A_d_F_NDMI_kobs30`**
    Formula:
    $$x_t - x_{t-30} \quad \text{where } x_t \text{ is } \mathrm{F\_NDMI}$$
    Captures 30-step change.

17. **`A_grad_F_NDMI_kobs7`**
    Formula:
    $$\frac{x_t - x_{t-7}}{7} \quad \text{where } x_t \text{ is } \mathrm{F\_NDMI}$$
    Average slope over 7 observations.

18. **`A_grad_F_NDMI_kobs14`**
    Formula:
    $$\frac{x_t - x_{t-14}}{14} \quad \text{where } x_t \text{ is } \mathrm{F\_NDMI}$$
    Average slope over 14 observations.

19. **`A_grad_F_NDMI_kobs30`**
    Formula:
    $$\frac{x_t - x_{t-30}}{30} \quad \text{where } x_t \text{ is } \mathrm{F\_NDMI}$$
    Average slope over 30 observations.

20. **`A_pct_F_NDMI`**
    Formula:
    $$\frac{x_t - x_{t-1}}{x_{t-1} + \epsilon} \quad \text{where } x_t \text{ is } \mathrm{F\_NDMI}$$
    One-step percent change.

21. **`A_d_E_SAR_ratio_kobs1`**
    Formula:
    $$x_t - x_{t-1} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_ratio}$$
    Captures 1-step change.

22. **`A_d_E_SAR_ratio_kobs2`**
    Formula:
    $$x_t - x_{t-2} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_ratio}$$
    Captures 2-step change.

23. **`A_d_E_SAR_ratio_kobs5`**
    Formula:
    $$x_t - x_{t-5} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_ratio}$$
    Captures 5-step change.

24. **`A_d_E_SAR_ratio_kobs7`**
    Formula:
    $$x_t - x_{t-7} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_ratio}$$
    Captures 7-step change.

25. **`A_d_E_SAR_ratio_kobs14`**
    Formula:
    $$x_t - x_{t-14} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_ratio}$$
    Captures 14-step change.

26. **`A_d_E_SAR_ratio_kobs30`**
    Formula:
    $$x_t - x_{t-30} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_ratio}$$
    Captures 30-step change.

27. **`A_grad_E_SAR_ratio_kobs7`**
    Formula:
    $$\frac{x_t - x_{t-7}}{7} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_ratio}$$
    Average slope over 7 observations.

28. **`A_grad_E_SAR_ratio_kobs14`**
    Formula:
    $$\frac{x_t - x_{t-14}}{14} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_ratio}$$
    Average slope over 14 observations.

29. **`A_grad_E_SAR_ratio_kobs30`**
    Formula:
    $$\frac{x_t - x_{t-30}}{30} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_ratio}$$
    Average slope over 30 observations.

30. **`A_pct_E_SAR_ratio`**
    Formula:
    $$\frac{x_t - x_{t-1}}{x_{t-1} + \epsilon} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_ratio}$$
    One-step percent change.

31. **`A_d_LST_modis_kobs1`**
    Formula:
    $$x_t - x_{t-1} \quad \text{where } x_t \text{ is } \mathrm{LST\_modis}$$
    Captures 1-step change.

32. **`A_d_LST_modis_kobs2`**
    Formula:
    $$x_t - x_{t-2} \quad \text{where } x_t \text{ is } \mathrm{LST\_modis}$$
    Captures 2-step change.

33. **`A_d_LST_modis_kobs5`**
    Formula:
    $$x_t - x_{t-5} \quad \text{where } x_t \text{ is } \mathrm{LST\_modis}$$
    Captures 5-step change.

34. **`A_d_LST_modis_kobs7`**
    Formula:
    $$x_t - x_{t-7} \quad \text{where } x_t \text{ is } \mathrm{LST\_modis}$$
    Captures 7-step change.

35. **`A_d_LST_modis_kobs14`**
    Formula:
    $$x_t - x_{t-14} \quad \text{where } x_t \text{ is } \mathrm{LST\_modis}$$
    Captures 14-step change.

36. **`A_d_LST_modis_kobs30`**
    Formula:
    $$x_t - x_{t-30} \quad \text{where } x_t \text{ is } \mathrm{LST\_modis}$$
    Captures 30-step change.

37. **`A_grad_LST_modis_kobs7`**
    Formula:
    $$\frac{x_t - x_{t-7}}{7} \quad \text{where } x_t \text{ is } \mathrm{LST\_modis}$$
    Average slope over 7 observations.

38. **`A_grad_LST_modis_kobs14`**
    Formula:
    $$\frac{x_t - x_{t-14}}{14} \quad \text{where } x_t \text{ is } \mathrm{LST\_modis}$$
    Average slope over 14 observations.

39. **`A_grad_LST_modis_kobs30`**
    Formula:
    $$\frac{x_t - x_{t-30}}{30} \quad \text{where } x_t \text{ is } \mathrm{LST\_modis}$$
    Average slope over 30 observations.

40. **`A_pct_LST_modis`**
    Formula:
    $$\frac{x_t - x_{t-1}}{x_{t-1} + \epsilon} \quad \text{where } x_t \text{ is } \mathrm{LST\_modis}$$
    One-step percent change.

41. **`A_d_F_NDVI_kobs1`**
    Formula:
    $$x_t - x_{t-1} \quad \text{where } x_t \text{ is } \mathrm{F\_NDVI}$$
    Captures 1-step change.

42. **`A_d_F_NDVI_kobs2`**
    Formula:
    $$x_t - x_{t-2} \quad \text{where } x_t \text{ is } \mathrm{F\_NDVI}$$
    Captures 2-step change.

43. **`A_d_F_NDVI_kobs5`**
    Formula:
    $$x_t - x_{t-5} \quad \text{where } x_t \text{ is } \mathrm{F\_NDVI}$$
    Captures 5-step change.

44. **`A_d_F_NDVI_kobs7`**
    Formula:
    $$x_t - x_{t-7} \quad \text{where } x_t \text{ is } \mathrm{F\_NDVI}$$
    Captures 7-step change.

45. **`A_d_F_NDVI_kobs14`**
    Formula:
    $$x_t - x_{t-14} \quad \text{where } x_t \text{ is } \mathrm{F\_NDVI}$$
    Captures 14-step change.

46. **`A_d_F_NDVI_kobs30`**
    Formula:
    $$x_t - x_{t-30} \quad \text{where } x_t \text{ is } \mathrm{F\_NDVI}$$
    Captures 30-step change.

47. **`A_grad_F_NDVI_kobs7`**
    Formula:
    $$\frac{x_t - x_{t-7}}{7} \quad \text{where } x_t \text{ is } \mathrm{F\_NDVI}$$
    Average slope over 7 observations.

48. **`A_grad_F_NDVI_kobs14`**
    Formula:
    $$\frac{x_t - x_{t-14}}{14} \quad \text{where } x_t \text{ is } \mathrm{F\_NDVI}$$
    Average slope over 14 observations.

49. **`A_grad_F_NDVI_kobs30`**
    Formula:
    $$\frac{x_t - x_{t-30}}{30} \quad \text{where } x_t \text{ is } \mathrm{F\_NDVI}$$
    Average slope over 30 observations.

50. **`A_pct_F_NDVI`**
    Formula:
    $$\frac{x_t - x_{t-1}}{x_{t-1} + \epsilon} \quad \text{where } x_t \text{ is } \mathrm{F\_NDVI}$$
    One-step percent change.

51. **`A_d_E_SAR_diff_kobs1`**
    Formula:
    $$x_t - x_{t-1} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_diff}$$
    Captures 1-step change.

52. **`A_d_E_SAR_diff_kobs2`**
    Formula:
    $$x_t - x_{t-2} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_diff}$$
    Captures 2-step change.

53. **`A_d_E_SAR_diff_kobs5`**
    Formula:
    $$x_t - x_{t-5} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_diff}$$
    Captures 5-step change.

54. **`A_d_E_SAR_diff_kobs7`**
    Formula:
    $$x_t - x_{t-7} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_diff}$$
    Captures 7-step change.

55. **`A_d_E_SAR_diff_kobs14`**
    Formula:
    $$x_t - x_{t-14} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_diff}$$
    Captures 14-step change.

56. **`A_d_E_SAR_diff_kobs30`**
    Formula:
    $$x_t - x_{t-30} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_diff}$$
    Captures 30-step change.

57. **`A_grad_E_SAR_diff_kobs7`**
    Formula:
    $$\frac{x_t - x_{t-7}}{7} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_diff}$$
    Average slope over 7 observations.

58. **`A_grad_E_SAR_diff_kobs14`**
    Formula:
    $$\frac{x_t - x_{t-14}}{14} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_diff}$$
    Average slope over 14 observations.

59. **`A_grad_E_SAR_diff_kobs30`**
    Formula:
    $$\frac{x_t - x_{t-30}}{30} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_diff}$$
    Average slope over 30 observations.

60. **`A_pct_E_SAR_diff`**
    Formula:
    $$\frac{x_t - x_{t-1}}{x_{t-1} + \epsilon} \quad \text{where } x_t \text{ is } \mathrm{E\_SAR\_diff}$$
    One-step percent change.

61. **`A_d_s2_b11_kobs1`**
    Formula:
    $$x_t - x_{t-1} \quad \text{where } x_t \text{ is } \mathrm{s2\_b11}$$
    Captures 1-step change.

62. **`A_d_s2_b11_kobs2`**
    Formula:
    $$x_t - x_{t-2} \quad \text{where } x_t \text{ is } \mathrm{s2\_b11}$$
    Captures 2-step change.

63. **`A_d_s2_b11_kobs5`**
    Formula:
    $$x_t - x_{t-5} \quad \text{where } x_t \text{ is } \mathrm{s2\_b11}$$
    Captures 5-step change.

64. **`A_d_s2_b11_kobs7`**
    Formula:
    $$x_t - x_{t-7} \quad \text{where } x_t \text{ is } \mathrm{s2\_b11}$$
    Captures 7-step change.

65. **`A_d_s2_b11_kobs14`**
    Formula:
    $$x_t - x_{t-14} \quad \text{where } x_t \text{ is } \mathrm{s2\_b11}$$
    Captures 14-step change.

66. **`A_d_s2_b11_kobs30`**
    Formula:
    $$x_t - x_{t-30} \quad \text{where } x_t \text{ is } \mathrm{s2\_b11}$$
    Captures 30-step change.

67. **`A_grad_s2_b11_kobs7`**
    Formula:
    $$\frac{x_t - x_{t-7}}{7} \quad \text{where } x_t \text{ is } \mathrm{s2\_b11}$$
    Average slope over 7 observations.

68. **`A_grad_s2_b11_kobs14`**
    Formula:
    $$\frac{x_t - x_{t-14}}{14} \quad \text{where } x_t \text{ is } \mathrm{s2\_b11}$$
    Average slope over 14 observations.

69. **`A_grad_s2_b11_kobs30`**
    Formula:
    $$\frac{x_t - x_{t-30}}{30} \quad \text{where } x_t \text{ is } \mathrm{s2\_b11}$$
    Average slope over 30 observations.

70. **`A_pct_s2_b11`**
    Formula:
    $$\frac{x_t - x_{t-1}}{x_{t-1} + \epsilon} \quad \text{where } x_t \text{ is } \mathrm{s2\_b11}$$
    One-step percent change.

71. **`A_d_s2_b12_kobs1`**
    Formula:
    $$x_t - x_{t-1} \quad \text{where } x_t \text{ is } \mathrm{s2\_b12}$$
    Captures 1-step change.

72. **`A_d_s2_b12_kobs2`**
    Formula:
    $$x_t - x_{t-2} \quad \text{where } x_t \text{ is } \mathrm{s2\_b12}$$
    Captures 2-step change.

73. **`A_d_s2_b12_kobs5`**
    Formula:
    $$x_t - x_{t-5} \quad \text{where } x_t \text{ is } \mathrm{s2\_b12}$$
    Captures 5-step change.

74. **`A_d_s2_b12_kobs7`**
    Formula:
    $$x_t - x_{t-7} \quad \text{where } x_t \text{ is } \mathrm{s2\_b12}$$
    Captures 7-step change.

75. **`A_d_s2_b12_kobs14`**
    Formula:
    $$x_t - x_{t-14} \quad \text{where } x_t \text{ is } \mathrm{s2\_b12}$$
    Captures 14-step change.

76. **`A_d_s2_b12_kobs30`**
    Formula:
    $$x_t - x_{t-30} \quad \text{where } x_t \text{ is } \mathrm{s2\_b12}$$
    Captures 30-step change.

77. **`A_grad_s2_b12_kobs7`**
    Formula:
    $$\frac{x_t - x_{t-7}}{7} \quad \text{where } x_t \text{ is } \mathrm{s2\_b12}$$
    Average slope over 7 observations.

78. **`A_grad_s2_b12_kobs14`**
    Formula:
    $$\frac{x_t - x_{t-14}}{14} \quad \text{where } x_t \text{ is } \mathrm{s2\_b12}$$
    Average slope over 14 observations.

79. **`A_grad_s2_b12_kobs30`**
    Formula:
    $$\frac{x_t - x_{t-30}}{30} \quad \text{where } x_t \text{ is } \mathrm{s2\_b12}$$
    Average slope over 30 observations.

80. **`A_pct_s2_b12`**
    Formula:
    $$\frac{x_t - x_{t-1}}{x_{t-1} + \epsilon} \quad \text{where } x_t \text{ is } \mathrm{s2\_b12}$$
    One-step percent change.

## Volatility / rolling stats (`V_family`)

1. **`V_rollstd_G_API_kobs7`**
   Formula:
   $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 7,\ \mathrm{ddof}=0$$
   where $x_t$ is $G\_API$.
   Volatility proxy.

2. **`V_rollrng_G_API_kobs7`**
   Formula:
   $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
   where $x_t$ is $G\_API$.
   Range / variability.

3. **`V_rollcv_G_API_kobs7`**
   Formula:
   $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 7$$
   where $x_t$ is $G\_API$.
   Scale-free variability.

4. **`V_rollmean_G_API_kobs7`**
   Formula:
   $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 7$$
   where $x_t$ is $G\_API$.
   Local level.

5. **`V_rollmin_G_API_kobs7`**
   Formula:
   $$\min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
   where $x_t$ is $G\_API$.
   Local low.

6. **`V_rollmax_G_API_kobs7`**
   Formula:
   $$\max\!\left(x_{t-w+1:t}\right), \quad w = 7$$
   where $x_t$ is $G\_API$.
   Local high.

7. **`V_ema_G_API_kobs7`**
   Formula:
   $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{7+1}$$
   where $x_t$ is $G\_API$.
   Smooth trend.

8. **`V_rollstd_G_API_kobs14`**
   Formula:
   $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 14,\ \mathrm{ddof}=0$$
   where $x_t$ is $G\_API$.
   Volatility proxy.

9. **`V_rollrng_G_API_kobs14`**
   Formula:
   $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
   where $x_t$ is $G\_API$.
   Range / variability.

10. **`V_rollcv_G_API_kobs14`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 14$$
    where $x_t$ is $G\_API$.
    Scale-free variability.

11. **`V_rollmean_G_API_kobs14`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $G\_API$.
    Local level.

12. **`V_rollmin_G_API_kobs14`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $G\_API$.
    Local low.

13. **`V_rollmax_G_API_kobs14`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $G\_API$.
    Local high.

14. **`V_ema_G_API_kobs14`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{14+1}$$
    where $x_t$ is $G\_API$.
    Smooth trend.

15. **`V_rollstd_G_API_kobs30`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 30,\ \mathrm{ddof}=0$$
    where $x_t$ is $G\_API$.
    Volatility proxy.

16. **`V_rollrng_G_API_kobs30`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $G\_API$.
    Range / variability.

17. **`V_rollcv_G_API_kobs30`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 30$$
    where $x_t$ is $G\_API$.
    Scale-free variability.

18. **`V_rollmean_G_API_kobs30`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $G\_API$.
    Local level.

19. **`V_rollmin_G_API_kobs30`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $G\_API$.
    Local low.

20. **`V_rollmax_G_API_kobs30`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $G\_API$.
    Local high.

21. **`V_ema_G_API_kobs30`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{30+1}$$
    where $x_t$ is $G\_API$.
    Smooth trend.

22. **`V_rollstd_F_NDMI_kobs7`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 7,\ \mathrm{ddof}=0$$
    where $x_t$ is $F\_NDMI$.
    Volatility proxy.

23. **`V_rollrng_F_NDMI_kobs7`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $F\_NDMI$.
    Range / variability.

24. **`V_rollcv_F_NDMI_kobs7`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 7$$
    where $x_t$ is $F\_NDMI$.
    Scale-free variability.

25. **`V_rollmean_F_NDMI_kobs7`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $F\_NDMI$.
    Local level.

26. **`V_rollmin_F_NDMI_kobs7`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $F\_NDMI$.
    Local low.

27. **`V_rollmax_F_NDMI_kobs7`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $F\_NDMI$.
    Local high.

28. **`V_ema_F_NDMI_kobs7`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{7+1}$$
    where $x_t$ is $F\_NDMI$.
    Smooth trend.

29. **`V_rollstd_F_NDMI_kobs14`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 14,\ \mathrm{ddof}=0$$
    where $x_t$ is $F\_NDMI$.
    Volatility proxy.

30. **`V_rollrng_F_NDMI_kobs14`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $F\_NDMI$.
    Range / variability.

31. **`V_rollcv_F_NDMI_kobs14`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 14$$
    where $x_t$ is $F\_NDMI$.
    Scale-free variability.

32. **`V_rollmean_F_NDMI_kobs14`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $F\_NDMI$.
    Local level.

33. **`V_rollmin_F_NDMI_kobs14`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $F\_NDMI$.
    Local low.

34. **`V_rollmax_F_NDMI_kobs14`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $F\_NDMI$.
    Local high.

35. **`V_ema_F_NDMI_kobs14`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{14+1}$$
    where $x_t$ is $F\_NDMI$.
    Smooth trend.

36. **`V_rollstd_F_NDMI_kobs30`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 30,\ \mathrm{ddof}=0$$
    where $x_t$ is $F\_NDMI$.
    Volatility proxy.

37. **`V_rollrng_F_NDMI_kobs30`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $F\_NDMI$.
    Range / variability.

38. **`V_rollcv_F_NDMI_kobs30`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 30$$
    where $x_t$ is $F\_NDMI$.
    Scale-free variability.

39. **`V_rollmean_F_NDMI_kobs30`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $F\_NDMI$.
    Local level.

40. **`V_rollmin_F_NDMI_kobs30`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $F\_NDMI$.
    Local low.

41. **`V_rollmax_F_NDMI_kobs30`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $F\_NDMI$.
    Local high.

42. **`V_ema_F_NDMI_kobs30`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{30+1}$$
    where $x_t$ is $F\_NDMI$.
    Smooth trend.

43. **`V_rollstd_E_SAR_ratio_kobs7`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 7,\ \mathrm{ddof}=0$$
    where $x_t$ is $E\_SAR\_ratio$.
    Volatility proxy.

44. **`V_rollrng_E_SAR_ratio_kobs7`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $E\_SAR\_ratio$.
    Range / variability.

45. **`V_rollcv_E_SAR_ratio_kobs7`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 7$$
    where $x_t$ is $E\_SAR\_ratio$.
    Scale-free variability.

46. **`V_rollmean_E_SAR_ratio_kobs7`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $E\_SAR\_ratio$.
    Local level.

47. **`V_rollmin_E_SAR_ratio_kobs7`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $E\_SAR\_ratio$.
    Local low.

48. **`V_rollmax_E_SAR_ratio_kobs7`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $E\_SAR\_ratio$.
    Local high.

49. **`V_ema_E_SAR_ratio_kobs7`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{7+1}$$
    where $x_t$ is $E\_SAR\_ratio$.
    Smooth trend.

50. **`V_rollstd_E_SAR_ratio_kobs14`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 14,\ \mathrm{ddof}=0$$
    where $x_t$ is $E\_SAR\_ratio$.
    Volatility proxy.

51. **`V_rollrng_E_SAR_ratio_kobs14`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $E\_SAR\_ratio$.
    Range / variability.

52. **`V_rollcv_E_SAR_ratio_kobs14`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 14$$
    where $x_t$ is $E\_SAR\_ratio$.
    Scale-free variability.

53. **`V_rollmean_E_SAR_ratio_kobs14`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $E\_SAR\_ratio$.
    Local level.

54. **`V_rollmin_E_SAR_ratio_kobs14`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $E\_SAR\_ratio$.
    Local low.

55. **`V_rollmax_E_SAR_ratio_kobs14`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $E\_SAR\_ratio$.
    Local high.

56. **`V_ema_E_SAR_ratio_kobs14`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{14+1}$$
    where $x_t$ is $E\_SAR\_ratio$.
    Smooth trend.

57. **`V_rollstd_E_SAR_ratio_kobs30`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 30,\ \mathrm{ddof}=0$$
    where $x_t$ is $E\_SAR\_ratio$.
    Volatility proxy.

58. **`V_rollrng_E_SAR_ratio_kobs30`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $E\_SAR\_ratio$.
    Range / variability.

59. **`V_rollcv_E_SAR_ratio_kobs30`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 30$$
    where $x_t$ is $E\_SAR\_ratio$.
    Scale-free variability.

60. **`V_rollmean_E_SAR_ratio_kobs30`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $E\_SAR\_ratio$.
    Local level.

61. **`V_rollmin_E_SAR_ratio_kobs30`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $E\_SAR\_ratio$.
    Local low.

62. **`V_rollmax_E_SAR_ratio_kobs30`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $E\_SAR\_ratio$.
    Local high.

63. **`V_ema_E_SAR_ratio_kobs30`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{30+1}$$
    where $x_t$ is $E\_SAR\_ratio$.
    Smooth trend.

---

64. **`V_rollstd_LST_modis_kobs7`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 7,\ \mathrm{ddof}=0$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Volatility proxy.

65. **`V_rollrng_LST_modis_kobs7`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Range / variability.

66. **`V_rollcv_LST_modis_kobs7`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 7$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Scale-free variability.

67. **`V_rollmean_LST_modis_kobs7`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Local level.

68. **`V_rollmin_LST_modis_kobs7`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Local low.

69. **`V_rollmax_LST_modis_kobs7`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Local high.

70. **`V_ema_LST_modis_kobs7`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{7+1}$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Smooth trend.

---

71. **`V_rollstd_LST_modis_kobs14`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 14,\ \mathrm{ddof}=0$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Volatility proxy.

72. **`V_rollrng_LST_modis_kobs14`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Range / variability.

73. **`V_rollcv_LST_modis_kobs14`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 14$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Scale-free variability.

74. **`V_rollmean_LST_modis_kobs14`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Local level.

75. **`V_rollmin_LST_modis_kobs14`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Local low.

76. **`V_rollmax_LST_modis_kobs14`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Local high.

77. **`V_ema_LST_modis_kobs14`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{14+1}$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Smooth trend.

---

78. **`V_rollstd_LST_modis_kobs30`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 30,\ \mathrm{ddof}=0$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Volatility proxy.

79. **`V_rollrng_LST_modis_kobs30`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Range / variability.

80. **`V_rollcv_LST_modis_kobs30`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 30$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Scale-free variability.

81. **`V_rollmean_LST_modis_kobs30`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Local level.

82. **`V_rollmin_LST_modis_kobs30`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Local low.

83. **`V_rollmax_LST_modis_kobs30`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 30$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Local high.

84. **`V_ema_LST_modis_kobs30`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{30+1}$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Smooth trend.

---

85. **`V_rollstd_F_NDVI_kobs7`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 7,\ \mathrm{ddof}=0$$
    where $x_t$ is $F\_NDVI$.
    Volatility proxy.

86. **`V_rollrng_F_NDVI_kobs7`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $F\_NDVI$.
    Range / variability.

87. **`V_rollcv_F_NDVI_kobs7`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 7$$
    where $x_t$ is $F\_NDVI$.
    Scale-free variability.

88. **`V_rollmean_F_NDVI_kobs7`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $F\_NDVI$.
    Local level.

89. **`V_rollmin_F_NDVI_kobs7`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $F\_NDVI$.
    Local low.

90. **`V_rollmax_F_NDVI_kobs7`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $F\_NDVI$.
    Local high.

91. **`V_ema_F_NDVI_kobs7`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{7+1}$$
    where $x_t$ is $F\_NDVI$.
    Smooth trend.

92. **`V_rollstd_F_NDVI_kobs14`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 14,\ \mathrm{ddof}=0$$
    where $x_t$ is $F\_NDVI$.
    Volatility proxy.

93. **`V_rollrng_F_NDVI_kobs14`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $F\_NDVI$.
    Range / variability.

94. **`V_rollcv_F_NDVI_kobs14`**
    Formula:
    $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 14$$
    where $x_t$ is $F\_NDVI$.
    Scale-free variability.

95. **`V_rollmean_F_NDVI_kobs14`**
    Formula:
    $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $F\_NDVI$.
    Local level.

96. **`V_rollmin_F_NDVI_kobs14`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $F\_NDVI$.
    Local low.

97. **`V_rollmax_F_NDVI_kobs14`**
    Formula:
    $$\max\!\left(x_{t-w+1:t}\right), \quad w = 14$$
    where $x_t$ is $F\_NDVI$.
    Local high.

98. **`V_ema_F_NDVI_kobs14`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{14+1}$$
    where $x_t$ is $F\_NDVI$.
    Smooth trend.

99. **`V_rollstd_F_NDVI_kobs30`**
    Formula:
    $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 30,\ \mathrm{ddof}=0$$
    where $x_t$ is $F\_NDVI$.
    Volatility proxy.

100.  **`V_rollrng_F_NDVI_kobs30`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $F\_NDVI$.
      Range / variability.

101.  **`V_rollcv_F_NDVI_kobs30`**
      Formula:
      $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 30$$
      where $x_t$ is $F\_NDVI$.
      Scale-free variability.

102.  **`V_rollmean_F_NDVI_kobs30`**
      Formula:
      $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $F\_NDVI$.
      Local level.

103.  **`V_rollmin_F_NDVI_kobs30`**
      Formula:
      $$\min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $F\_NDVI$.
      Local low.

104.  **`V_rollmax_F_NDVI_kobs30`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $F\_NDVI$.
      Local high.

105.  **`V_ema_F_NDVI_kobs30`**
      Formula:
      $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{30+1}$$
      where $x_t$ is $F\_NDVI$.
      Smooth trend.

106.  **`V_rollstd_E_SAR_diff_kobs7`**
      Formula:
      $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 7,\ \mathrm{ddof}=0$$
      where $x_t$ is $E\_SAR\_diff$.
      Volatility proxy.

107.  **`V_rollrng_E_SAR_diff_kobs7`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $E\_SAR\_diff$.
      Range / variability.

108.  **`V_rollcv_E_SAR_diff_kobs7`**
      Formula:
      $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 7$$
      where $x_t$ is $E\_SAR\_diff$.
      Scale-free variability.

109.  **`V_rollmean_E_SAR_diff_kobs7`**
      Formula:
      $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $E\_SAR\_diff$.
      Local level.

110.  **`V_rollmin_E_SAR_diff_kobs7`**
      Formula:
      $$\min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $E\_SAR\_diff$.
      Local low.

111.  **`V_rollmax_E_SAR_diff_kobs7`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $E\_SAR\_diff$.
      Local high.

112.  **`V_ema_E_SAR_diff_kobs7`**
      Formula:
      $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{7+1}$$
      where $x_t$ is $E\_SAR\_diff$.
      Smooth trend.

113.  **`V_rollstd_E_SAR_diff_kobs14`**
      Formula:
      $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 14,\ \mathrm{ddof}=0$$
      where $x_t$ is $E\_SAR\_diff$.
      Volatility proxy.

114.  **`V_rollrng_E_SAR_diff_kobs14`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $E\_SAR\_diff$.
      Range / variability.

115.  **`V_rollcv_E_SAR_diff_kobs14`**
      Formula:
      $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 14$$
      where $x_t$ is $E\_SAR\_diff$.
      Scale-free variability.

116.  **`V_rollmean_E_SAR_diff_kobs14`**
      Formula:
      $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $E\_SAR\_diff$.
      Local level.

117.  **`V_rollmin_E_SAR_diff_kobs14`**
      Formula:
      $$\min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $E\_SAR\_diff$.
      Local low.

118.  **`V_rollmax_E_SAR_diff_kobs14`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $E\_SAR\_diff$.
      Local high.

119.  **`V_ema_E_SAR_diff_kobs14`**
      Formula:
      $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{14+1}$$
      where $x_t$ is $E\_SAR\_diff$.
      Smooth trend.

120.  **`V_rollstd_E_SAR_diff_kobs30`**
      Formula:
      $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 30,\ \mathrm{ddof}=0$$
      where $x_t$ is $E\_SAR\_diff$.
      Volatility proxy.

121.  **`V_rollrng_E_SAR_diff_kobs30`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $E\_SAR\_diff$.
      Range / variability.

122.  **`V_rollcv_E_SAR_diff_kobs30`**
      Formula:
      $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 30$$
      where $x_t$ is $E\_SAR\_diff$.
      Scale-free variability.

123.  **`V_rollmean_E_SAR_diff_kobs30`**
      Formula:
      $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $E\_SAR\_diff$.
      Local level.

124.  **`V_rollmin_E_SAR_diff_kobs30`**
      Formula:
      $$\min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $E\_SAR\_diff$.
      Local low.

125.  **`V_rollmax_E_SAR_diff_kobs30`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $E\_SAR\_diff$.
      Local high.

126.  **`V_ema_E_SAR_diff_kobs30`**
      Formula:
      $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{30+1}$$
      where $x_t$ is $E\_SAR\_diff$.
      Smooth trend.

127.  **`V_rollstd_s2_b11_kobs7`**
      Formula:
      $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 7,\ \mathrm{ddof}=0$$
      where $x_t$ is $s2\_b11$.
      Volatility proxy.

128.  **`V_rollrng_s2_b11_kobs7`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $s2\_b11$.
      Range / variability.

129.  **`V_rollcv_s2_b11_kobs7`**
      Formula:
      $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 7$$
      where $x_t$ is $s2\_b11$.
      Scale-free variability.

130.  **`V_rollmean_s2_b11_kobs7`**
      Formula:
      $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $s2\_b11$.
      Local level.

131.  **`V_rollmin_s2_b11_kobs7`**
      Formula:
      $$\min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $s2\_b11$.
      Local low.

132.  **`V_rollmax_s2_b11_kobs7`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $s2\_b11$.
      Local high.

133.  **`V_ema_s2_b11_kobs7`**
      Formula:
      $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{7+1}$$
      where $x_t$ is $s2\_b11$.
      Smooth trend.

134.  **`V_rollstd_s2_b11_kobs14`**
      Formula:
      $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 14,\ \mathrm{ddof}=0$$
      where $x_t$ is $s2\_b11$.
      Volatility proxy.

135.  **`V_rollrng_s2_b11_kobs14`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $s2\_b11$.
      Range / variability.

136.  **`V_rollcv_s2_b11_kobs14`**
      Formula:
      $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 14$$
      where $x_t$ is $s2\_b11$.
      Scale-free variability.

137.  **`V_rollmean_s2_b11_kobs14`**
      Formula:
      $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $s2\_b11$.
      Local level.

138.  **`V_rollmin_s2_b11_kobs14`**
      Formula:
      $$\min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $s2\_b11$.
      Local low.

139.  **`V_rollmax_s2_b11_kobs14`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $s2\_b11$.
      Local high.

140.  **`V_ema_s2_b11_kobs14`**
      Formula:
      $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{14+1}$$
      where $x_t$ is $s2\_b11$.
      Smooth trend.

141.  **`V_rollstd_s2_b11_kobs30`**
      Formula:
      $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 30,\ \mathrm{ddof}=0$$
      where $x_t$ is $s2\_b11$.
      Volatility proxy.

142.  **`V_rollrng_s2_b11_kobs30`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $s2\_b11$.
      Range / variability.

143.  **`V_rollcv_s2_b11_kobs30`**
      Formula:
      $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 30$$
      where $x_t$ is $s2\_b11$.
      Scale-free variability.

144.  **`V_rollmean_s2_b11_kobs30`**
      Formula:
      $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $s2\_b11$.
      Local level.

145.  **`V_rollmin_s2_b11_kobs30`**
      Formula:
      $$\min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $s2\_b11$.
      Local low.

146.  **`V_rollmax_s2_b11_kobs30`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $s2\_b11$.
      Local high.

147.  **`V_ema_s2_b11_kobs30`**
      Formula:
      $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{30+1}$$
      where $x_t$ is $s2\_b11$.
      Smooth trend.

148.  **`V_rollstd_s2_b12_kobs7`**
      Formula:
      $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 7,\ \mathrm{ddof}=0$$
      where $x_t$ is $s2\_b12$.
      Volatility proxy.

149.  **`V_rollrng_s2_b12_kobs7`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $s2\_b12$.
      Range / variability.

150.  **`V_rollcv_s2_b12_kobs7`**
      Formula:
      $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 7$$
      where $x_t$ is $s2\_b12$.
      Scale-free variability.

151.  **`V_rollmean_s2_b12_kobs7`**
      Formula:
      $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $s2\_b12$.
      Local level.

152.  **`V_rollmin_s2_b12_kobs7`**
      Formula:
      $$\min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $s2\_b12$.
      Local low.

153.  **`V_rollmax_s2_b12_kobs7`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right), \quad w = 7$$
      where $x_t$ is $s2\_b12$.
      Local high.

154.  **`V_ema_s2_b12_kobs7`**
      Formula:
      $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{7+1}$$
      where $x_t$ is $s2\_b12$.
      Smooth trend.

155.  **`V_rollstd_s2_b12_kobs14`**
      Formula:
      $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 14,\ \mathrm{ddof}=0$$
      where $x_t$ is $s2\_b12$.
      Volatility proxy.

156.  **`V_rollrng_s2_b12_kobs14`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $s2\_b12$.
      Range / variability.

157.  **`V_rollcv_s2_b12_kobs14`**
      Formula:
      $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 14$$
      where $x_t$ is $s2\_b12$.
      Scale-free variability.

158.  **`V_rollmean_s2_b12_kobs14`**
      Formula:
      $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $s2\_b12$.
      Local level.

159.  **`V_rollmin_s2_b12_kobs14`**
      Formula:
      $$\min\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $s2\_b12$.
      Local low.

160.  **`V_rollmax_s2_b12_kobs14`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right), \quad w = 14$$
      where $x_t$ is $s2\_b12$.
      Local high.

161.  **`V_ema_s2_b12_kobs14`**
      Formula:
      $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{14+1}$$
      where $x_t$ is $s2\_b12$.
      Smooth trend.

162.  **`V_rollstd_s2_b12_kobs30`**
      Formula:
      $$\mathrm{std}\!\left(x_{t-w+1:t}\right), \quad w = 30,\ \mathrm{ddof}=0$$
      where $x_t$ is $s2\_b12$.
      Volatility proxy.

163.  **`V_rollrng_s2_b12_kobs30`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right) - \min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $s2\_b12$.
      Range / variability.

164.  **`V_rollcv_s2_b12_kobs30`**
      Formula:
      $$\frac{\mathrm{std}\!\left(x_{t-w+1:t}\right)}{\mathrm{mean}\!\left(x_{t-w+1:t}\right) + \epsilon}, \quad w = 30$$
      where $x_t$ is $s2\_b12$.
      Scale-free variability.

165.  **`V_rollmean_s2_b12_kobs30`**
      Formula:
      $$\mathrm{mean}\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $s2\_b12$.
      Local level.

166.  **`V_rollmin_s2_b12_kobs30`**
      Formula:
      $$\min\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $s2\_b12$.
      Local low.

167.  **`V_rollmax_s2_b12_kobs30`**
      Formula:
      $$\max\!\left(x_{t-w+1:t}\right), \quad w = 30$$
      where $x_t$ is $s2\_b12$.
      Local high.

168.  **`V_ema_s2_b12_kobs30`**
      Formula:
      $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{30+1}$$
      where $x_t$ is $s2\_b12$.
      Smooth trend.

## Memory / lags (`C_family`)

1. **`C_lag_G_API_kobs1`**
   Formula:
   $$x_{t-1}$$
   where $x_t$ is $G\_API$.
   Lagged memory.

2. **`C_lag_G_API_kobs2`**
   Formula:
   $$x_{t-2}$$
   where $x_t$ is $G\_API$.
   Lagged memory.

3. **`C_lag_G_API_kobs5`**
   Formula:
   $$x_{t-5}$$
   where $x_t$ is $G\_API$.
   Lagged memory.

4. **`C_lag_G_API_kobs6`**
   Formula:
   $$x_{t-6}$$
   where $x_t$ is $G\_API$.
   Lagged memory.

5. **`C_lag_G_API_kobs12`**
   Formula:
   $$x_{t-12}$$
   where $x_t$ is $G\_API$.
   Lagged memory.

6. **`C_lag_G_API_kobs30`**
   Formula:
   $$x_{t-30}$$
   where $x_t$ is $G\_API$.
   Lagged memory.

7. **`C_smm_G_API_alpha0.85_n5`**
   Formula:
   $$\sum_{j=1}^{5} 0.85^{j}\,x_{t-j}$$
   where $x_t$ is $G\_API$.
   Exponential lag memory.

8. **`C_lag_F_NDMI_kobs1`**
   Formula:
   $$x_{t-1}$$
   where $x_t$ is $F\_NDMI$.
   Lagged memory.

9. **`C_lag_F_NDMI_kobs2`**
   Formula:
   $$x_{t-2}$$
   where $x_t$ is $F\_NDMI$.
   Lagged memory.

10. **`C_lag_F_NDMI_kobs5`**
    Formula:
    $$x_{t-5}$$
    where $x_t$ is $F\_NDMI$.
    Lagged memory.

11. **`C_lag_F_NDMI_kobs6`**
    Formula:
    $$x_{t-6}$$
    where $x_t$ is $F\_NDMI$.
    Lagged memory.

12. **`C_lag_F_NDMI_kobs12`**
    Formula:
    $$x_{t-12}$$
    where $x_t$ is $F\_NDMI$.
    Lagged memory.

13. **`C_lag_F_NDMI_kobs30`**
    Formula:
    $$x_{t-30}$$
    where $x_t$ is $F\_NDMI$.
    Lagged memory.

14. **`C_smm_F_NDMI_alpha0.85_n5`**
    Formula:
    $$\sum_{j=1}^{5} 0.85^{j}\,x_{t-j}$$
    where $x_t$ is $F\_NDMI$.
    Exponential lag memory.

15. **`C_lag_E_SAR_ratio_kobs1`**
    Formula:
    $$x_{t-1}$$
    where $x_t$ is $E\_SAR\_ratio$.
    Lagged memory.

16. **`C_lag_E_SAR_ratio_kobs2`**
    Formula:
    $$x_{t-2}$$
    where $x_t$ is $E\_SAR\_ratio$.
    Lagged memory.

17. **`C_lag_E_SAR_ratio_kobs5`**
    Formula:
    $$x_{t-5}$$
    where $x_t$ is $E\_SAR\_ratio$.
    Lagged memory.

18. **`C_lag_E_SAR_ratio_kobs6`**
    Formula:
    $$x_{t-6}$$
    where $x_t$ is $E\_SAR\_ratio$.
    Lagged memory.

19. **`C_lag_E_SAR_ratio_kobs12`**
    Formula:
    $$x_{t-12}$$
    where $x_t$ is $E\_SAR\_ratio$.
    Lagged memory.

20. **`C_lag_E_SAR_ratio_kobs30`**
    Formula:
    $$x_{t-30}$$
    where $x_t$ is $E\_SAR\_ratio$.
    Lagged memory.

21. **`C_smm_E_SAR_ratio_alpha0.85_n5`**
    Formula:
    $$\sum_{j=1}^{5} 0.85^{j}\,x_{t-j}$$
    where $x_t$ is $E\_SAR\_ratio$.
    Exponential lag memory.

22. **`C_lag_LST_modis_kobs1`**
    Formula:
    $$x_{t-1}$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Lagged memory.

23. **`C_lag_LST_modis_kobs2`**
    Formula:
    $$x_{t-2}$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Lagged memory.

24. **`C_lag_LST_modis_kobs5`**
    Formula:
    $$x_{t-5}$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Lagged memory.

25. **`C_lag_LST_modis_kobs6`**
    Formula:
    $$x_{t-6}$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Lagged memory.

26. **`C_lag_LST_modis_kobs12`**
    Formula:
    $$x_{t-12}$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Lagged memory.

27. **`C_lag_LST_modis_kobs30`**
    Formula:
    $$x_{t-30}$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Lagged memory.

28. **`C_smm_LST_modis_alpha0.85_n5`**
    Formula:
    $$\sum_{j=1}^{5} 0.85^{j}\,x_{t-j}$$
    where $x_t$ is $\mathrm{LST\_modis}$.
    Exponential lag memory.

29. **`C_lag_F_NDVI_kobs1`**
    Formula:
    $$x_{t-1}$$
    where $x_t$ is $F\_NDVI$.
    Lagged memory.

30. **`C_lag_F_NDVI_kobs2`**
    Formula:
    $$x_{t-2}$$
    where $x_t$ is $F\_NDVI$.
    Lagged memory.

31. **`C_lag_F_NDVI_kobs5`**
    Formula:
    $$x_{t-5}$$
    where $x_t$ is $F\_NDVI$.
    Lagged memory.

32. **`C_lag_F_NDVI_kobs6`**
    Formula:
    $$x_{t-6}$$
    where $x_t$ is $F\_NDVI$.
    Lagged memory.

33. **`C_lag_F_NDVI_kobs12`**
    Formula:
    $$x_{t-12}$$
    where $x_t$ is $F\_NDVI$.
    Lagged memory.

34. **`C_lag_F_NDVI_kobs30`**
    Formula:
    $$x_{t-30}$$
    where $x_t$ is $F\_NDVI$.
    Lagged memory.

35. **`C_smm_F_NDVI_alpha0.85_n5`**
    Formula:
    $$\sum_{j=1}^{5} 0.85^{j}\,x_{t-j}$$
    where $x_t$ is $F\_NDVI$.
    Exponential lag memory.

36. **`C_lag_E_SAR_diff_kobs1`**
    Formula:
    $$x_{t-1}$$
    where $x_t$ is $E\_SAR\_diff$.
    Lagged memory.

37. **`C_lag_E_SAR_diff_kobs2`**
    Formula:
    $$x_{t-2}$$
    where $x_t$ is $E\_SAR\_diff$.
    Lagged memory.

38. **`C_lag_E_SAR_diff_kobs5`**
    Formula:
    $$x_{t-5}$$
    where $x_t$ is $E\_SAR\_diff$.
    Lagged memory.

39. **`C_lag_E_SAR_diff_kobs6`**
    Formula:
    $$x_{t-6}$$
    where $x_t$ is $E\_SAR\_diff$.
    Lagged memory.

40. **`C_lag_E_SAR_diff_kobs12`**
    Formula:
    $$x_{t-12}$$
    where $x_t$ is $E\_SAR\_diff$.
    Lagged memory.

41. **`C_lag_E_SAR_diff_kobs30`**
    Formula:
    $$x_{t-30}$$
    where $x_t$ is $E\_SAR\_diff$.
    Lagged memory.

42. **`C_smm_E_SAR_diff_alpha0.85_n5`**
    Formula:
    $$\sum_{j=1}^{5} 0.85^{j}\,x_{t-j}$$
    where $x_t$ is $E\_SAR\_diff$.
    Exponential lag memory.

43. **`C_lag_s2_b11_kobs1`**
    Formula:
    $$x_{t-1}$$
    where $x_t$ is $s2\_b11$.
    Lagged memory.

44. **`C_lag_s2_b11_kobs2`**
    Formula:
    $$x_{t-2}$$
    where $x_t$ is $s2\_b11$.
    Lagged memory.

45. **`C_lag_s2_b11_kobs5`**
    Formula:
    $$x_{t-5}$$
    where $x_t$ is $s2\_b11$.
    Lagged memory.

46. **`C_lag_s2_b11_kobs6`**
    Formula:
    $$x_{t-6}$$
    where $x_t$ is $s2\_b11$.
    Lagged memory.

47. **`C_lag_s2_b11_kobs12`**
    Formula:
    $$x_{t-12}$$
    where $x_t$ is $s2\_b11$.
    Lagged memory.

48. **`C_lag_s2_b11_kobs30`**
    Formula:
    $$x_{t-30}$$
    where $x_t$ is $s2\_b11$.
    Lagged memory.

49. **`C_smm_s2_b11_alpha0.85_n5`**
    Formula:
    $$\sum_{j=1}^{5} 0.85^{j}\,x_{t-j}$$
    where $x_t$ is $s2\_b11$.
    Exponential lag memory.

50. **`C_lag_s2_b12_kobs1`**
    Formula:
    $$x_{t-1}$$
    where $x_t$ is $s2\_b12$.
    Lagged memory.

51. **`C_lag_s2_b12_kobs2`**
    Formula:
    $$x_{t-2}$$
    where $x_t$ is $s2\_b12$.
    Lagged memory.

52. **`C_lag_s2_b12_kobs5`**
    Formula:
    $$x_{t-5}$$
    where $x_t$ is $s2\_b12$.
    Lagged memory.

53. **`C_lag_s2_b12_kobs6`**
    Formula:
    $$x_{t-6}$$
    where $x_t$ is $s2\_b12$.
    Lagged memory.

54. **`C_lag_s2_b12_kobs12`**
    Formula:
    $$x_{t-12}$$
    where $x_t$ is $s2\_b12$.
    Lagged memory.

55. **`C_lag_s2_b12_kobs30`**
    Formula:
    $$x_{t-30}$$
    where $x_t$ is $s2\_b12$.
    Lagged memory.

56. **`C_smm_s2_b12_alpha0.85_n5`**
    Formula:
    $$\sum_{j=1}^{5} 0.85^{j}\,x_{t-j}$$
    where $x_t$ is $s2\_b12$.
    Exponential lag memory.

## Events / roughness / spikes (`E*`, `I*`)

1. **`E_dVV_1`**
   Formula:
   $$VV_t - VV_{t-1}$$
   One-step VV backscatter change (used for spike / event logic).

2. **`I_ts_spike_s1_vv`**
   Formula:
   $$\Delta t \;\text{since last}\; z_t \ge 2 \;\text{in}\; \Delta VV$$
   (past-only).
   Time since a VV spike; highlights abrupt radar changes.

## Cross-signal coupling (`H_family`)

1. **`H_corr_E_SAR_ratio__F_NDMI_kobs7`**
   Formula:
   $$\mathrm{corr}\!\left(E\_SAR\_ratio,\; F\_NDMI\right), \quad w = 7$$
   Computed over a past-only window.
   Radar–optical coupling.

2. **`H_corr_E_SAR_ratio__F_NDMI_kobs14`**
   Formula:
   $$\mathrm{corr}\!\left(E\_SAR\_ratio,\; F\_NDMI\right), \quad w = 14$$
   Computed over a past-only window.
   Radar–optical coupling.

3. **`H_corr_LST_modis__F_NDMI_kobs7`**
   Formula:
   $$\mathrm{corr}\!\left(\mathrm{LST\_modis},\; F\_NDMI\right), \quad w = 7$$
   Computed over a past-only window.
   Thermal–moisture coupling.

4. **`H_corr_LST_modis__F_NDMI_kobs14`**
   Formula:
   $$\mathrm{corr}\!\left(\mathrm{LST\_modis},\; F\_NDMI\right), \quad w = 14$$
   Computed over a past-only window.
   Thermal–moisture coupling.

## Seasonal + spectral (`D_family`)

1. **`D_sa_F_NDMI`**
   Formula:
   $$x_t - \mu_{m,\mathrm{train}}$$
   where $m$ is month and $x_t$ is $F\_NDMI$.
   Seasonal anomaly (train-only statistics).

2. **`D_z_F_NDMI`**
   Formula:
   $$\frac{x_t - \mu_{m,\mathrm{train}}}{\sigma_{m,\mathrm{train}}}$$
   where $x_t$ is $F\_NDMI$.
   Seasonal z-score (train-only statistics).

3. **`D_sa_E_SAR_ratio`**
   Formula:
   $$x_t - \mu_{m,\mathrm{train}}$$
   where $m$ is month and $x_t$ is $E\_SAR\_ratio$.
   Seasonal anomaly (train-only statistics).

4. **`D_z_E_SAR_ratio`**
   Formula:
   $$\frac{x_t - \mu_{m,\mathrm{train}}}{\sigma_{m,\mathrm{train}}}$$
   where $x_t$ is $E\_SAR\_ratio$.
   Seasonal z-score (train-only statistics).

5. **`D_sa_LST_modis`**
   Formula:
   $$x_t - \mu_{m,\mathrm{train}}$$
   where $m$ is month and $x_t$ is $\mathrm{LST\_modis}$.
   Seasonal anomaly (train-only statistics).

6. **`D_z_LST_modis`**
   Formula:
   $$\frac{x_t - \mu_{m,\mathrm{train}}}{\sigma_{m,\mathrm{train}}}$$
   where $x_t$ is $\mathrm{LST\_modis}$.
   Seasonal z-score (train-only statistics).

7. **`D_fft_dom_F_NDMI_kobs30`**
   Definition:
   Dominant FFT bin index computed over a past-only window with $w = 30$.
   Applied to $F\_NDMI$.
   Captures dominant periodicity.

8. **`D_fft_ent_F_NDMI_kobs30`**
   Formula:
   $$-\frac{\sum_i p_i \log p_i}{\log n}$$
   where $p_i$ are normalized FFT magnitudes (past-only, $w = 30$).
   Applied to $F\_NDMI$.
   Spectral entropy.

9. **`D_fft_dom_E_SAR_ratio_kobs30`**
   Definition:
   Dominant FFT bin index computed over a past-only window with $w = 30$.
   Applied to $E\_SAR\_ratio$.
   Captures dominant periodicity.

10. **`D_fft_ent_E_SAR_ratio_kobs30`**
    Formula:
    $$-\frac{\sum_i p_i \log p_i}{\log n}$$
    where $p_i$ are normalized FFT magnitudes (past-only, $w = 30$).
    Applied to $E\_SAR\_ratio$.
    Spectral entropy.

11. **`D_fft_dom_LST_modis_kobs30`**
    Definition:
    Dominant FFT bin index computed over a past-only window with $w = 30$.
    Applied to $\mathrm{LST\_modis}$.
    Captures dominant periodicity.

12. **`D_fft_ent_LST_modis_kobs30`**
    Formula:
    $$-\frac{\sum_i p_i \log p_i}{\log n}$$
    where $p_i$ are normalized FFT magnitudes (past-only, $w = 30$).
    Applied to $\mathrm{LST\_modis}$.
    Spectral entropy.

---

_Jakob Balkovec & Kerry Cheon_
