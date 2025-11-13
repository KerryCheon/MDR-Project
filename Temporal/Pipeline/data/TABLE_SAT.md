<!-- All set to KEEP as of Nov 13...will go through them again once we get feedback -->

| Feature Name                           | Abbreviation / Math Symbol              | Keep / Throw Away | Source      |
| -------------------------------------- | --------------------------------------- | ----------------- | ----------- |
| First Difference                       | $$\Delta x_t$$                          | ✅ Keep           | Derived     |
| Second Difference                      | $$\Delta^2 x_t$$                        | ✅ Keep           | Derived     |
| n-Step Difference                      | $$D_n(t)$$                              | ✅ Keep           | Derived     |
| Temporal Gradient                      | $$\text{grad}_t(k)$$                    | ✅ Keep           | Derived     |
| Percent Change                         | $$\text{pct}_t$$                        | ✅ Keep           | Derived     |
| Moving Average                         | $$\text{MA}_t(k)$$                      | ✅ Keep           | Derived     |
| Exponential Moving Average             | $$\text{EMA}_t$$                        | ✅ Keep           | Derived     |
| Rolling Standard Deviation             | $$\sigma_t(k)$$                         | ✅ Keep           | Derived     |
| Rolling Coefficient of Variation       | $$\text{CV}_t$$                         | ✅ Keep           | Derived     |
| Rolling Minimum                        | $$\min(x)$$                             | ✅ Keep           | Derived     |
| Rolling Maximum                        | $$\max(x)$$                             | ✅ Keep           | Derived     |
| Z-Score Anomaly                        | $$z_t$$                                 | ✅ Keep           | Derived     |
| Seasonal Anomaly                       | $$\text{SA}_t$$                         | ✅ Keep           | Derived     |
| Lag-1 Feature                          | $$x_{t-1}$$                             | ✅ Keep           | Derived     |
| Lag-7 Feature                          | $$x_{t-7}$$                             | ✅ Keep           | Derived     |
| Lag-30 Feature                         | $$x_{t-30}$$                            | ✅ Keep           | Derived     |
| Rolling Range                          | $$\text{range}_t$$                      | ✅ Keep           | Derived     |
| Dominant Fourier Frequency             | $$k^*$$                                 | ✅ Keep           | Derived     |
| Spectral Entropy                       | $$H$$                                   | ✅ Keep           | Derived     |
| VV/VH Ratio (Sentinel-1)               | $$\frac{\text{VV}}{\text{VH}}$$         | ✅ Keep           | Sentinel-1  |
| Backscatter Difference (VV–VH)         | $$\text{VV} - \text{VH}$$               | ✅ Keep           | Sentinel-1  |
| Radar Coherence                        | $$\gamma$$                              | ✅ Keep           | Sentinel-1  |
| NDVI Time Series                       | $$\text{NDVI}$$                         | ✅ Keep           | Sentinel-2  |
| NDMI (Moisture Index)                  | $$\text{NDMI}$$                         | ✅ Keep           | Sentinel-2  |
| MSI (Moisture Stress Index)            | $$\text{MSI}$$                          | ✅ Keep           | Sentinel-2  |
| SWIR Reflectance Temporal Curve        | $$\text{SWIR}(t)$$                      | ✅ Keep           | Sentinel-2  |
| Rainfall Accumulation                  | $$R_k(t)$$                              | ✅ Keep           | Precip Data |
| Days Since Last Rain                   | $$\text{DSLR}_t$$                       | ✅ Keep           | Precip Data |
| Temperature Anomaly (LST)              | $$\text{TA}_t$$                         | ✅ Keep           | MODIS / S2  |
| Radar–Optical Lag Correlation          | $$\text{Corr}(\text{VV}, \text{NDVI})$$ | ✅ Keep           | S1 + S2     |
| Time Since Last Wetness Spike          | $$\text{TSWS}_t$$                       | ✅ Keep           | Sentinel-1  |
| Soil Moisture Memory Index             | $$\text{SMM}_t$$                        | ✅ Keep           | Derived     |
| Radar Temporal Roughness Index         | $$\text{RTI}$$                          | ✅ Keep           | Sentinel-1  |
| Temperature–Moisture Temporal Coupling | $$C_{TM}(k)$$                           | ✅ Keep           | S1 + LST    |
| Antecedent Precipitation Index (API)   | $$\text{API}_t$$                        | ✅ Keep           | Precip Data |
