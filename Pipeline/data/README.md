### **Dataset Columns Overview**

| **Column Name**       | **Description**                                                    | **Units / Format**          | **Source**               |
| --------------------- | ------------------------------------------------------------------ | --------------------------- | ------------------------ |
| `date`                | Observation date                                                   | `YYYY-MM-DD`                | USCRN / ECE sensor       |
| `station_id`          | Unique numeric identifier of the observation station               | integer                     | USCRN metadata           |
| `crx_vn`              | Station version number (sensor configuration version)              | numeric code                | USCRN                    |
| `longitude`           | Station longitude (decimal degrees, W is negative)                 | degrees                     | USCRN                    |
| `latitude`            | Station latitude (decimal degrees)                                 | degrees                     | USCRN                    |
| `air_temp_max`        | Maximum daily air temperature                                      | °C                          | USCRN                    |
| `air_temp_min`        | Minimum daily air temperature                                      | °C                          | USCRN                    |
| `air_temp_mean`       | Mean daily air temperature                                         | °C                          | USCRN                    |
| `air_temp_avg`        | Average daily temperature (redundant or averaged from hourly data) | °C                          | USCRN                    |
| `precipitation`       | Daily accumulated precipitation                                    | mm                          | USCRN                    |
| `solar_radiation`     | Daily solar radiation                                              | MJ/m²                       | USCRN                    |
| `sur_temp_type`       | Surface temperature type (Radiometric/Contact/Unknown)             | categorical (`R`, `C`, `U`) | USCRN                    |
| `sur_temp_max`        | Maximum daily surface temperature                                  | °C                          | USCRN                    |
| `sur_temp_min`        | Minimum daily surface temperature                                  | °C                          | USCRN                    |
| `sur_temp_avg`        | Average daily surface temperature                                  | °C                          | USCRN                    |
| `rh_max`              | Maximum daily relative humidity                                    | %                           | Derived / ECE sensor     |
| `rh_min`              | Minimum daily relative humidity                                    | %                           | Derived / ECE sensor     |
| `rh_mean`             | Mean daily relative humidity                                       | %                           | Derived / ECE sensor     |
| `soil_moisture_5cm`   | Volumetric soil moisture at 5 cm depth                             | m³/m³                       | USCRN / ECE sensor       |
| `soil_moisture_10cm`  | Volumetric soil moisture at 10 cm depth                            | m³/m³                       | USCRN / ECE sensor       |
| `soil_moisture_20cm`  | Volumetric soil moisture at 20 cm depth                            | m³/m³                       | USCRN / ECE sensor       |
| `soil_moisture_50cm`  | Volumetric soil moisture at 50 cm depth                            | m³/m³                       | USCRN / ECE sensor       |
| `soil_moisture_100cm` | Volumetric soil moisture at 100 cm depth                           | m³/m³                       | USCRN / ECE sensor       |
| `soil_temp_5cm`       | Soil temperature at 5 cm depth                                     | °C                          | USCRN / ECE sensor       |
| `soil_temp_10cm`      | Soil temperature at 10 cm depth                                    | °C                          | USCRN / ECE sensor       |
| `soil_temp_20cm`      | Soil temperature at 20 cm depth                                    | °C                          | USCRN / ECE sensor       |
| `soil_temp_50cm`      | Soil temperature at 50 cm depth                                    | °C                          | USCRN / ECE sensor       |
| `soil_temp_100cm`     | Soil temperature at 100 cm depth                                   | °C                          | USCRN / ECE sensor       |
| `source_file`         | Name of the original raw input file                                | string                      | Pipeline metadata        |
| `LST`                 | Land Surface Temperature (satellite-derived)                       | Kelvin / °C                 | MODIS / GEE              |
| `NDVI`                | Normalized Difference Vegetation Index                             | unitless (−1 to 1)          | MODIS / GEE              |
| `Rain_sat`            | Satellite-estimated rainfall over the region                       | mm                          | GEE (CHIRPS / GPM)       |
| `DOY`                 | Day of Year derived from `date`                                    | integer (1–366)             | Computed                 |
| `Rain_3d`             | Rolling 3-day rainfall sum                                         | mm                          | Computed                 |
| `SM_prev`             | Previous day’s soil moisture (temporal lag feature)                | m³/m³                       | Computed                 |
| `SM_label`            | Soil moisture classification label (target variable, optional)     | categorical / binary        | Future (model-generated) |
