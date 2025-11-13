### **Dataset Columns Overview**

| **Column Name**       | **Description**                                                | **Units / Format**          | **Source**               | **Keep/Remove for Model Training** |
| --------------------- | -------------------------------------------------------------- | --------------------------- | ------------------------ | ---------------------------------- |
| `date`                | Observation date                                               | `YYYY-MM-DD`                | USCRN / ECE sensor       | ❌ Remove                          |
| `station_id`          | Unique numeric identifier of the observation station           | integer                     | USCRN metadata           | ❌ Remove                          |
| `crx_vn`              | Station version number (sensor configuration version)          | numeric code                | USCRN                    | ❌ Remove                          |
| `longitude`           | Station longitude (decimal degrees, W is negative)             | degrees                     | USCRN                    | ❌ Remove                          |
| `latitude`            | Station latitude (decimal degrees)                             | degrees                     | USCRN                    | ❌ Remove                          |
| `air_temp_max`        | Maximum daily air temperature                                  | °C                          | USCRN                    | ✅ Keep                            |
| `air_temp_min`        | Minimum daily air temperature                                  | °C                          | USCRN                    | ✅ Keep                            |
| `air_temp_mean`       | Mean daily air temperature (hourly)                            | °C                          | USCRN                    | ✅ Keep                            |
| `air_temp_avg`        | Average daily temperature (min vs max -> midpoint)             | °C                          | USCRN                    | ❌ Remove                          |
| `precipitation`       | Daily accumulated precipitation                                | mm                          | USCRN                    | ✅ Keep                            |
| `solar_radiation`     | Daily solar radiation                                          | MJ/m²                       | USCRN                    | ✅ Keep                            |
| `sur_temp_type`       | Surface temperature type (Radiometric/Contact/Unknown)         | categorical (`R`, `C`, `U`) | USCRN                    | ✅ Keep                            |
| `sur_temp_max`        | Maximum daily surface temperature                              | °C                          | USCRN                    | ✅ Keep                            |
| `sur_temp_min`        | Minimum daily surface temperature                              | °C                          | USCRN                    | ✅ Keep                            |
| `sur_temp_avg`        | Average daily surface temperature                              | °C                          | USCRN                    | ✅ Keep                            |
| `rh_max`              | Maximum daily relative humidity                                | %                           | Derived / ECE sensor     | ✅ Keep                            |
| `rh_min`              | Minimum daily relative humidity                                | %                           | Derived / ECE sensor     | ✅ Keep                            |
| `rh_mean`             | Mean daily relative humidity                                   | %                           | Derived / ECE sensor     | ✅ Keep                            |
| `soil_moisture_5cm`   | Volumetric soil moisture at 5 cm depth                         | m³/m³                       | USCRN / ECE sensor       | ✅ Keep                            |
| `soil_moisture_10cm`  | Volumetric soil moisture at 10 cm depth                        | m³/m³                       | USCRN / ECE sensor       | ❌ Remove                          |
| `soil_moisture_20cm`  | Volumetric soil moisture at 20 cm depth                        | m³/m³                       | USCRN / ECE sensor       | ❌ Remove                          |
| `soil_moisture_50cm`  | Volumetric soil moisture at 50 cm depth                        | m³/m³                       | USCRN / ECE sensor       | ❌ Remove                          |
| `soil_moisture_100cm` | Volumetric soil moisture at 100 cm depth                       | m³/m³                       | USCRN / ECE sensor       | ❌ Remove                          |
| `soil_temp_5cm`       | Soil temperature at 5 cm depth                                 | °C                          | USCRN / ECE sensor       | ✅ Keep                            |
| `soil_temp_10cm`      | Soil temperature at 10 cm depth                                | °C                          | USCRN / ECE sensor       | ❌ Remove                          |
| `soil_temp_20cm`      | Soil temperature at 20 cm depth                                | °C                          | USCRN / ECE sensor       | ❌ Remove                          |
| `soil_temp_50cm`      | Soil temperature at 50 cm depth                                | °C                          | USCRN / ECE sensor       | ❌ Remove                          |
| `soil_temp_100cm`     | Soil temperature at 100 cm depth                               | °C                          | USCRN / ECE sensor       | ❌ Remove                          |
| `source_file`         | Name of the original raw input file                            | string                      | Pipeline metadata        | ❌ Remove                          |
| `LST`                 | Land Surface Temperature (satellite-derived)                   | Kelvin / °C                 | MODIS / GEE              | ✅ Keep                            |
| `NDVI`                | Normalized Difference Vegetation Index                         | unitless (−1 to 1)          | MODIS / GEE              | ✅ Keep                            |
| `Rain_sat`            | Satellite-estimated rainfall over the region                   | mm                          | GEE (CHIRPS / GPM)       | ✅ Keep                            |
| `DOY`                 | Day of Year derived from `date`                                | integer (1–366)             | Computed                 | ✅ Keep                            |
| `Rain_3d`             | Rolling 3-day rainfall sum                                     | mm                          | Computed                 | ✅ Keep                            |
| `SM_prev`             | Previous day’s soil moisture (temporal lag feature)            | m³/m³                       | Computed                 | ✅ Keep                            |
| `SM_label`            | Soil moisture classification label (target variable, optional) | categorical / binary        | Future (model-generated) | ✅ Keep                            |
