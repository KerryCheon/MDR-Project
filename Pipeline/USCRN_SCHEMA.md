# USCRN Data Schema Documentation

## Overview
This document describes the USCRN (U.S. Climate Reference Network) daily01 data format used in this pipeline.

## Data Format
USCRN daily01 files contain **28 space-separated columns** with indices **0-27**.

### Column Definitions

| Index | Column Name | USCRN Field Name | Description | Unit |
|-------|-------------|------------------|-------------|------|
| 0 | station_id | WBANNO | Station identifier | - |
| 1 | date | LST_DATE | Date in YYYYMMDD format | - |
| 2 | crx_vn | CRX_VN | Station/version number | - |
| 3 | longitude | LONGITUDE | Longitude (negative for West) | decimal degrees |
| 4 | latitude | LATITUDE | Latitude | decimal degrees |
| 5 | air_temp_max | T_DAILY_MAX | Maximum air temperature | °C |
| 6 | air_temp_min | T_DAILY_MIN | Minimum air temperature | °C |
| 7 | air_temp_mean | T_DAILY_MEAN | Mean air temperature | °C |
| 8 | air_temp_avg | T_DAILY_AVG | Average air temperature | °C |
| 9 | precipitation | P_DAILY_CALC | Precipitation | mm |
| 10 | solar_radiation | SOLARAD_DAILY | Solar radiation | MJ/m² |
| 11 | sur_temp_type | SUR_TEMP_DAILY_TYPE | Surface temp sensor type | R/C/U* |
| 12 | sur_temp_max | SUR_TEMP_DAILY_MAX | Maximum surface temperature | °C |
| 13 | sur_temp_min | SUR_TEMP_DAILY_MIN | Minimum surface temperature | °C |
| 14 | sur_temp_avg | SUR_TEMP_DAILY_AVG | Average surface temperature | °C |
| 15 | rh_max | RH_DAILY_MAX | Maximum relative humidity | % |
| 16 | rh_min | RH_DAILY_MIN | Minimum relative humidity | % |
| 17 | rh_mean | RH_DAILY_AVG | Average relative humidity | % |
| 18 | soil_moisture_5cm | SOIL_MOISTURE_5_DAILY | Soil moisture at 5cm depth | m³/m³ |
| 19 | soil_moisture_10cm | SOIL_MOISTURE_10_DAILY | Soil moisture at 10cm depth | m³/m³ |
| 20 | soil_moisture_20cm | SOIL_MOISTURE_20_DAILY | Soil moisture at 20cm depth | m³/m³ |
| 21 | soil_moisture_50cm | SOIL_MOISTURE_50_DAILY | Soil moisture at 50cm depth | m³/m³ |
| 22 | soil_moisture_100cm | SOIL_MOISTURE_100_DAILY | Soil moisture at 100cm depth | m³/m³ |
| 23 | soil_temp_5cm | SOIL_TEMP_5_DAILY | Soil temperature at 5cm depth | °C |
| 24 | soil_temp_10cm | SOIL_TEMP_10_DAILY | Soil temperature at 10cm depth | °C |
| 25 | soil_temp_20cm | SOIL_TEMP_20_DAILY | Soil temperature at 20cm depth | °C |
| 26 | soil_temp_50cm | SOIL_TEMP_50_DAILY | Soil temperature at 50cm depth | °C |
| 27 | soil_temp_100cm | SOIL_TEMP_100_DAILY | Soil temperature at 100cm depth | °C |

\* Surface temperature sensor type:
- **R** = IR (Infrared) sensor
- **C** = TC (Thermocouple) sensor
- **U** = Unknown/missing

## Missing Data
Missing or invalid values are represented as `-9999` or `-99.000` in the raw data files. These are automatically converted to `NaN` during parsing.

## Important Notes

### Column Alignment
⚠️ **All 28 columns (indices 0-27) must be defined in `config.yaml`** to prevent column misalignment issues.

Historically, columns 11-12 (sur_temp_type and sur_temp_max) were missing from the configuration, causing all subsequent columns to be incorrectly aligned. This has been fixed, but the schema should be validated regularly.

### Validation
Run the schema validation test to ensure the configuration matches the actual data:

```bash
python3 tests/schema_test.py
```

This test verifies:
- All 28 columns are mapped in config.yaml
- Column names match the expected USCRN schema
- Critical columns 11-12 are properly defined
- Actual data files have the expected column count

## References
- Official USCRN documentation: https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/
- USCRN data format specification: https://www.ncei.noaa.gov/data/us-climate-reference-network/
