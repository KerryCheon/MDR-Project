<!-- All set to KEEP as of Nov 13...will go through them again once we get feedback -->

| **#** | **Feature Name**                       | **Abbreviation**                        | **Keep / Throw Away** | **Source**  |
| ----- | -------------------------------------- | --------------------------------------- | --------------------- | ----------- |
| 1     | First Difference                       | $$\Delta x_t$$                          | ✅ Keep               | Derived     |
| 2     | Second Difference                      | $$\Delta^2 x_t$$                        | ✅ Keep               | Derived     |
| 3     | n-Step Difference                      | $$D_n(t)$$                              | ✅ Keep               | Derived     |
| 4     | Temporal Gradient                      | $$\text{grad}_t(k)$$                    | ✅ Keep               | Derived     |
| 5     | Percent Change                         | $$\text{pct}_t$$                        | ✅ Keep               | Derived     |
| 6     | Moving Average                         | $$\text{MA}_t(k)$$                      | ✅ Keep               | Derived     |
| 7     | Exponential Moving Average             | $$\text{EMA}_t$$                        | ✅ Keep               | Derived     |
| 8     | Rolling Standard Deviation             | $$\sigma_t(k)$$                         | ✅ Keep               | Derived     |
| 9     | Rolling Coefficient of Variation       | $$\text{CV}_t$$                         | ✅ Keep               | Derived     |
| 10    | Rolling Minimum                        | $$\min(x)$$                             | ✅ Keep               | Derived     |
| 11    | Rolling Maximum                        | $$\max(x)$$                             | ✅ Keep               | Derived     |
| 12    | Z-Score Anomaly                        | $$z_t$$                                 | ✅ Keep               | Derived     |
| 13    | Seasonal Anomaly                       | $$\text{SA}_t$$                         | ✅ Keep               | Derived     |
| 14    | Lag-1 Feature                          | $$x_{t-1}$$                             | ✅ Keep               | Derived     |
| 15    | Lag-7 Feature                          | $$x_{t-7}$$                             | ✅ Keep               | Derived     |
| 16    | Lag-30 Feature                         | $$x_{t-30}$$                            | ✅ Keep               | Derived     |
| 17    | Rolling Range                          | $$\text{range}_t$$                      | ✅ Keep               | Derived     |
| 18    | Dominant Fourier Frequency             | $$k^*$$                                 | ✅ Keep               | Derived     |
| 19    | Spectral Entropy                       | $$H$$                                   | ✅ Keep               | Derived     |
| 20    | VV/VH Ratio (Sentinel-1)               | $$\frac{\text{VV}}{\text{VH}}$$         | ✅ Keep               | Sentinel-1  |
| 21    | Backscatter Difference (VV–VH)         | $$\text{VV} - \text{VH}$$               | ✅ Keep               | Sentinel-1  |
| 22    | Radar Coherence                        | $$\gamma$$                              | ✅ Keep               | Sentinel-1  |
| 23    | NDVI Time Series                       | $$\text{NDVI}$$                         | ✅ Keep               | Sentinel-2  |
| 24    | NDMI (Moisture Index)                  | $$\text{NDMI}$$                         | ✅ Keep               | Sentinel-2  |
| 25    | MSI (Moisture Stress Index)            | $$\text{MSI}$$                          | ✅ Keep               | Sentinel-2  |
| 26    | SWIR Reflectance Temporal Curve        | $$\text{SWIR}(t)$$                      | ✅ Keep               | Sentinel-2  |
| 27    | Rainfall Accumulation                  | $$R_k(t)$$                              | ✅ Keep               | Precip Data |
| 28    | Days Since Last Rain                   | $$\text{DSLR}_t$$                       | ✅ Keep               | Precip Data |
| 29    | Temperature Anomaly (LST)              | $$\text{TA}_t$$                         | ✅ Keep               | MODIS / S2  |
| 30    | Radar–Optical Lag Correlation          | $$\text{Corr}(\text{VV}, \text{NDVI})$$ | ✅ Keep               | S1 + S2     |
| 31    | Time Since Last Wetness Spike          | $$\text{TSWS}_t$$                       | ✅ Keep               | Sentinel-1  |
| 32    | Soil Moisture Memory Index             | $$\text{SMM}_t$$                        | ✅ Keep               | Derived     |
| 33    | Radar Temporal Roughness Index         | $$\text{RTI}$$                          | ✅ Keep               | Sentinel-1  |
| 34    | Temperature–Moisture Temporal Coupling | $$C_{TM}(k)$$                           | ✅ Keep               | S1 + LST    |
| 35    | Antecedent Precipitation Index (API)   | $$\text{API}_t$$                        | ✅ Keep               | Precip Data |
