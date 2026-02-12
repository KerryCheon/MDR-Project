# Final Set of Features

**Authors:** Jakob Balkovec, Kerry Cheon

> The following features were selected for the final model based on the SHAP analysis conducted on the test data.

---

1. `DOY`: Day of year (1-366).

2. **`V_ema_LST_modis_kobs30`**
   Formula:
   $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{30+1}$$
   where $x_t$ is $\mathrm{LST\_modis}$.
   Smooth trend.

3. **`NDMI`**
   Formula:
   $$\frac{\mathrm{NIR} - \mathrm{SWIR1}}{\mathrm{NIR} + \mathrm{SWIR1} + \epsilon}$$
   Canopy/soil moisture proxy from Sentinel-2 (b8/b11).

4. **`C_smm_G_API_alpha0.85_n5`**
   Formula:
   $$\sum_{j=1}^{5} 0.85^{j}\,x_{t-j}$$
   where $x_t$ is $G\_API$.
   Exponential lag memory.

5. **`F_MSI`**
   Formula:
   $$\frac{\mathrm{SWIR1}}{\mathrm{NIR} + \epsilon}$$
   Moisture stress index (higher can mean drier).

6. `s1_vv`: Sentinel-1 VV backscatter (dB).

7. `precip_mm`: Daily precipitation in mm.

8. **`SAR_ratio`**
   Formula:
   $$\frac{VV}{VH + \epsilon}$$
   Radar polarization ratio, roughness/moisture-sensitive.

9. **`G_API`**
   Formula:
   $$\mathrm{API}_t = P_t + 0.9\,\mathrm{API}_{t-1}$$
   Antecedent precipitation index (rain memory).

10. **`V_ema_G_API_kobs7`**
    Formula:
    $$\mathrm{EMA}_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_{t-1}, \quad \alpha = \frac{2}{7+1}$$
    where $x_t$ is $G\_API$.
    Smooth trend.

11. `s1_vh`: Sentinel-1 VH backscatter (dB).

12. **`V_rollmin_G_API_kobs7`**
    Formula:
    $$\min\!\left(x_{t-w+1:t}\right), \quad w = 7$$
    where $x_t$ is $G\_API$.
    Local low.

13. `s2_b8`: Sentinel-2 band 8 (NIR).

14. **`G_rain_sum_30d`**
    Formula:
    $$\sum_{\tau \in [t-30,\,t]} P_\tau$$
    Short-term rainfall accumulation over calendar days.

---

_Jakob Balkovec & Kerry Cheon_
