# Experiment: `ece-input-diagnose-1.0`

Hourly-resolution follow-up to `derived_8.4-ece-model-salvage-1.1`'s `ece_all_sensors_input_overlay_rainfall.png`, where Best-1.1 prediction tracks rainfall while the daily ground truth looks flat. The executed notebook reported all numbers below; reusable parsing/averaging logic is imported from `ece-daily-mean-1.0` (`audit_coverage.py`, `compare_averaging.py`), not redefined.

## 1. Focus windows (notebook stdout)

Three largest pooled daily-max `precip_mm` days from `data/splits/derived_8.4_ece_v3/test.csv`, plus the nonzero neighbor 2026-07-25, extended with dry context days — day-before for event A (07-22), day-after for events B (07-27) and C (08-03) — giving 7 focus days in 3 event windows: A=`[07-22, 07-23]`, B=`[07-25, 07-26, 07-27]`, C=`[08-02, 08-03]`. All three context days read 0.0 pooled precip with full rows (`2026-08-01`, by contrast, has no rows at all — fully missing on all 5 stations.)

```text
pooled daily-max ranking:
date
2026-07-23    10.7
2026-07-26     6.8
2026-08-02     2.3
2026-07-25     0.8
2026-08-15     0.5
2026-07-28     0.3
2026-08-12     0.2
2026-07-22     0.0
2026-07-23: before 2026-07-22 max=0.0, after 2026-07-24 max=0.0
2026-07-26: before 2026-07-25 max=0.8, after 2026-07-27 max=0.0
2026-08-02: before 2026-08-01 max=0.0, after 2026-08-03 max=0.0
```

Only `2026-07-25` qualifies as a nonzero neighbor; 07-22/07-27/08-03 were added explicitly as dry context days (see above).

Daily target vs Best-1.1 (`Global_Single_60_no_smap_fs60`, mean over seeds 42/7/13) on the focus days (notebook stdout):

```text
             station_id       date  soil_moisture_5cm  best_1_1  precip_mm  G_rain_sum_3d
    ECE_BBG_Lost_Meadow 2026-07-22           0.050300  0.073203        0.0            0.0
        ECE_BBG_Main_St 2026-07-22           0.056064  0.080686        0.0            0.0
ECE_Renton_Garden_North 2026-07-22           0.197635  0.069179        0.0            0.0
 ECE_Renton_Garden_Shed 2026-07-22           0.072107  0.069172        0.0            0.0
        ECE_Renton_Home 2026-07-22           0.017618  0.071156        0.0            0.0
    ECE_BBG_Lost_Meadow 2026-07-23           0.059340  0.076211        6.1            6.1
        ECE_BBG_Main_St 2026-07-23           0.061891  0.082427        6.1            6.1
ECE_Renton_Garden_North 2026-07-23           0.188672  0.079805       10.7           10.7
 ECE_Renton_Garden_Shed 2026-07-23           0.078603  0.079529       10.7           10.7
        ECE_Renton_Home 2026-07-23           0.018937  0.079942       10.7           10.7
    ECE_BBG_Lost_Meadow 2026-07-25           0.055305  0.102594        0.0            6.1
        ECE_BBG_Main_St 2026-07-25           0.062631  0.107626        0.0            6.1
ECE_Renton_Garden_North 2026-07-25           0.189068  0.108421        0.5           11.2
 ECE_Renton_Garden_Shed 2026-07-25           0.079821  0.107975        0.5           11.2
        ECE_Renton_Home 2026-07-25           0.020834  0.105073        0.8           11.5
    ECE_BBG_Lost_Meadow 2026-07-26           0.054330  0.105030        6.2           12.3
        ECE_BBG_Main_St 2026-07-26           0.062743  0.109505        6.2           12.3
ECE_Renton_Garden_North 2026-07-26           0.193522  0.107096        6.8           18.0
 ECE_Renton_Garden_Shed 2026-07-26           0.081537  0.106478        6.8           18.0
        ECE_Renton_Home 2026-07-26           0.021794  0.102891        6.8           18.0
    ECE_BBG_Lost_Meadow 2026-07-27           0.052119  0.114293        0.0            6.2
        ECE_BBG_Main_St 2026-07-27           0.063988  0.117883        0.0            6.2
ECE_Renton_Garden_North 2026-07-27           0.174678  0.109046        0.0            7.3
 ECE_Renton_Garden_Shed 2026-07-27           0.078150  0.108759        0.0            7.3
        ECE_Renton_Home 2026-07-27           0.018435  0.106497        0.0            7.6
    ECE_BBG_Lost_Meadow 2026-08-02           0.057679  0.096421        0.8            0.8
        ECE_BBG_Main_St 2026-08-02           0.058134  0.098701        0.8            0.8
ECE_Renton_Garden_North 2026-08-02           0.137999  0.089632        2.3            2.3
 ECE_Renton_Garden_Shed 2026-08-02           0.079937  0.089347        2.3            2.3
        ECE_Renton_Home 2026-08-02           0.019381  0.085354        1.3            1.3
    ECE_BBG_Lost_Meadow 2026-08-03           0.060647  0.091667        0.0            0.8
        ECE_BBG_Main_St 2026-08-03           0.050900  0.094284        0.0            0.8
ECE_Renton_Garden_North 2026-08-03           0.135826  0.091558        0.0            2.3
 ECE_Renton_Garden_Shed 2026-08-03           0.078777  0.090854        0.0            2.3
        ECE_Renton_Home 2026-08-03           0.019126  0.085604        0.0            1.3
```

## 2. Sensor-data coverage on focus days (notebook stdout)

Raw files carry no QC/state flag column — coverage means burst-sample availability per Seattle-calendar hour. Every focus station-day has 22–24 distinct hours and passes the 18/24 h rule, so the flat daily line is not a missing-data artifact on these days:

```text
             station_id       date  n_samples  n_hours  meets_18h
    ECE_BBG_Lost_Meadow 2026-07-22        465       24       True
        ECE_BBG_Main_St 2026-07-22        408       24       True
ECE_Renton_Garden_North 2026-07-22        445       24       True
 ECE_Renton_Garden_Shed 2026-07-22        235       22       True
        ECE_Renton_Home 2026-07-22        436       24       True
    ECE_BBG_Lost_Meadow 2026-07-23        456       24       True
        ECE_BBG_Main_St 2026-07-23        422       24       True
ECE_Renton_Garden_North 2026-07-23        403       24       True
 ECE_Renton_Garden_Shed 2026-07-23        242       24       True
        ECE_Renton_Home 2026-07-23        405       24       True
    ECE_BBG_Lost_Meadow 2026-07-25        437       24       True
        ECE_BBG_Main_St 2026-07-25        437       24       True
ECE_Renton_Garden_North 2026-07-25        422       24       True
 ECE_Renton_Garden_Shed 2026-07-25        222       24       True
        ECE_Renton_Home 2026-07-25        409       24       True
    ECE_BBG_Lost_Meadow 2026-07-26        434       24       True
        ECE_BBG_Main_St 2026-07-26        418       24       True
ECE_Renton_Garden_North 2026-07-26        443       24       True
 ECE_Renton_Garden_Shed 2026-07-26        247       23       True
        ECE_Renton_Home 2026-07-26        375       23       True
    ECE_BBG_Lost_Meadow 2026-07-27        451       24       True
        ECE_BBG_Main_St 2026-07-27        419       24       True
ECE_Renton_Garden_North 2026-07-27        424       24       True
 ECE_Renton_Garden_Shed 2026-07-27        275       24       True
        ECE_Renton_Home 2026-07-27        409       24       True
    ECE_BBG_Lost_Meadow 2026-08-02        420       24       True
        ECE_BBG_Main_St 2026-08-02        431       24       True
ECE_Renton_Garden_North 2026-08-02        510       24       True
 ECE_Renton_Garden_Shed 2026-08-02        324       24       True
        ECE_Renton_Home 2026-08-02        384       24       True
    ECE_BBG_Lost_Meadow 2026-08-03        299       24       True
        ECE_BBG_Main_St 2026-08-03        474       24       True
ECE_Renton_Garden_North 2026-08-03        432       24       True
 ECE_Renton_Garden_Shed 2026-08-03        284       24       True
        ECE_Renton_Home 2026-08-03        424       24       True
min distinct hours on focus days: 22; days below full 24h: 3 (shown as line breaks, never interpolated)
```

Missing hours are kept as line breaks in the figures, never interpolated.

## 3. Daily-mean definition on focus days (notebook stdout)

Simple burst-mean (current `ece_pipe.py`) vs hourly-weighted mean (USCRN equal-hour weighting) vs median, in fraction m3/m3:

```text
             station_id       date   n  n_hours  simple_mean  hourly_weighted_mean  median  diurnal_range
    ECE_BBG_Lost_Meadow 2026-07-22 465       24     0.050300              0.049046 0.05730       0.037055
        ECE_BBG_Main_St 2026-07-22 408       24     0.056064              0.056160 0.05410       0.046379
ECE_Renton_Garden_North 2026-07-22 445       24     0.197635              0.197115 0.20010       0.067168
 ECE_Renton_Garden_Shed 2026-07-22 235       22     0.072107              0.072966 0.07250       0.050988
        ECE_Renton_Home 2026-07-22 436       24     0.017618              0.017446 0.01860       0.027206
    ECE_BBG_Lost_Meadow 2026-07-23 456       24     0.059340              0.058394 0.06280       0.026409
        ECE_BBG_Main_St 2026-07-23 422       24     0.061891              0.061911 0.06060       0.016236
ECE_Renton_Garden_North 2026-07-23 403       24     0.188672              0.189245 0.19250       0.053580
 ECE_Renton_Garden_Shed 2026-07-23 242       24     0.078603              0.079244 0.08065       0.035764
        ECE_Renton_Home 2026-07-23 405       24     0.018937              0.018968 0.02110       0.020925
    ECE_BBG_Lost_Meadow 2026-07-25 437       24     0.055305              0.054894 0.05630       0.026863
        ECE_BBG_Main_St 2026-07-25 437       24     0.062631              0.062364 0.06310       0.024244
ECE_Renton_Garden_North 2026-07-25 422       24     0.189068              0.188455 0.18655       0.062709
 ECE_Renton_Garden_Shed 2026-07-25 222       24     0.079821              0.080223 0.07990       0.034915
        ECE_Renton_Home 2026-07-25 409       24     0.020834              0.020936 0.02190       0.013206
    ECE_BBG_Lost_Meadow 2026-07-26 434       24     0.054330              0.053633 0.05830       0.026326
        ECE_BBG_Main_St 2026-07-26 418       24     0.062743              0.062721 0.06625       0.028405
ECE_Renton_Garden_North 2026-07-26 443       24     0.193522              0.192206 0.19710       0.041724
 ECE_Renton_Garden_Shed 2026-07-26 247       23     0.081537              0.082085 0.08480       0.028898
        ECE_Renton_Home 2026-07-26 375       23     0.021794              0.021781 0.02330       0.016446
    ECE_BBG_Lost_Meadow 2026-07-27 451       24     0.052119              0.051756 0.05560       0.028645
        ECE_BBG_Main_St 2026-07-27 419       24     0.063988              0.063906 0.06680       0.025608
ECE_Renton_Garden_North 2026-07-27 424       24     0.174678              0.174467 0.17350       0.067709
 ECE_Renton_Garden_Shed 2026-07-27 275       24     0.078150              0.078066 0.07970       0.037257
        ECE_Renton_Home 2026-07-27 409       24     0.018435              0.018670 0.01980       0.026577
    ECE_BBG_Lost_Meadow 2026-08-02 420       24     0.057679              0.054576 0.06350       0.035989
        ECE_BBG_Main_St 2026-08-02 431       24     0.058134              0.058477 0.06040       0.016512
ECE_Renton_Garden_North 2026-08-02 510       24     0.137999              0.137674 0.14015       0.065445
 ECE_Renton_Garden_Shed 2026-08-02 324       24     0.079937              0.079146 0.08245       0.038962
        ECE_Renton_Home 2026-08-02 384       24     0.019381              0.018790 0.02015       0.025725
    ECE_BBG_Lost_Meadow 2026-08-03 299       24     0.060647              0.058415 0.06540       0.070212
        ECE_BBG_Main_St 2026-08-03 474       24     0.050900              0.051663 0.05255       0.036989
ECE_Renton_Garden_North 2026-08-03 432       24     0.135826              0.135786 0.13520       0.065649
 ECE_Renton_Garden_Shed 2026-08-03 284       24     0.078777              0.078543 0.08120       0.038982
        ECE_Renton_Home 2026-08-03 424       24     0.019126              0.018977 0.02050       0.025309
```

## 4. Smoothing summary: the sensor sees the event, the daily mean hides it (notebook stdout)

Per focus station-day: within-day swing (`hourly_range`, `spike_above_daily`) vs day-to-day steps of the daily target and of Best-1.1. (`NaN` day-steps on `2026-08-02` are the `2026-08-01` gap — no previous day exists.)

```text
             station_id       date  precip_mm  daily_mean  hourly_range  spike_above_daily  d_target_vs_prev_day  d_pred_vs_prev_day
    ECE_BBG_Lost_Meadow 2026-07-22        0.0    0.050300      0.037055           0.015955             -0.000259            0.000138
        ECE_BBG_Main_St 2026-07-22        0.0    0.056064      0.046379           0.027019             -0.001398            0.000581
ECE_Renton_Garden_North 2026-07-22        0.0    0.197635      0.067168           0.037465              0.002664            0.000253
 ECE_Renton_Garden_Shed 2026-07-22        0.0    0.072107      0.050988           0.026410             -0.007333            0.000029
        ECE_Renton_Home 2026-07-22        0.0    0.017618      0.027206           0.009588             -0.002493           -0.000552
    ECE_BBG_Lost_Meadow 2026-07-23        6.1    0.059340      0.026409           0.008386              0.009040            0.003008
        ECE_BBG_Main_St 2026-07-23        6.1    0.061891      0.016236           0.009950              0.005827            0.001740
ECE_Renton_Garden_North 2026-07-23       10.7    0.188672      0.053580           0.022474             -0.008963            0.010626
 ECE_Renton_Garden_Shed 2026-07-23       10.7    0.078603      0.035764           0.014230              0.006496            0.010357
        ECE_Renton_Home 2026-07-23       10.7    0.018937      0.020925           0.005216              0.001319            0.008786
    ECE_BBG_Lost_Meadow 2026-07-25        0.0    0.055305      0.026863           0.008820              0.000363            0.017827
        ECE_BBG_Main_St 2026-07-25        0.0    0.062631      0.024244           0.009969             -0.000932            0.017101
ECE_Renton_Garden_North 2026-07-25        0.5    0.189068      0.062709           0.043879              0.020918            0.017815
 ECE_Renton_Garden_Shed 2026-07-25        0.5    0.079821      0.034915           0.016613              0.003432            0.017487
        ECE_Renton_Home 2026-07-25        0.8    0.020834      0.013206           0.004824              0.002458            0.004169
    ECE_BBG_Lost_Meadow 2026-07-26        6.2    0.054330      0.026326           0.006329             -0.000974            0.002435
        ECE_BBG_Main_St 2026-07-26        6.2    0.062743      0.028405           0.008094              0.000112            0.001879
ECE_Renton_Garden_North 2026-07-26        6.8    0.193522      0.041724           0.014489              0.004454           -0.001325
 ECE_Renton_Garden_Shed 2026-07-26        6.8    0.081537      0.028898           0.012343              0.001717           -0.001497
        ECE_Renton_Home 2026-07-26        6.8    0.021794      0.016446           0.004112              0.000959           -0.002182
    ECE_BBG_Lost_Meadow 2026-07-27        0.0    0.052119      0.028645           0.010035             -0.002211            0.009263
        ECE_BBG_Main_St 2026-07-27        0.0    0.063988      0.025608           0.008885              0.001245            0.008379
ECE_Renton_Garden_North 2026-07-27        0.0    0.174678      0.067709           0.031472             -0.018845            0.001951
 ECE_Renton_Garden_Shed 2026-07-27        0.0    0.078150      0.037257           0.016043             -0.003387            0.002280
        ECE_Renton_Home 2026-07-27        0.0    0.018435      0.026577           0.008141             -0.003358            0.003606
    ECE_BBG_Lost_Meadow 2026-08-02        0.8    0.057679      0.035989           0.009310                   NaN                 NaN
        ECE_BBG_Main_St 2026-08-02        0.8    0.058134      0.016512           0.005534                   NaN                 NaN
ECE_Renton_Garden_North 2026-08-02        2.3    0.137999      0.065445           0.030555                   NaN                 NaN
 ECE_Renton_Garden_Shed 2026-08-02        2.3    0.079937      0.038962           0.017707                   NaN                 NaN
        ECE_Renton_Home 2026-08-02        1.3    0.019381      0.025725           0.008544                   NaN                 NaN
    ECE_BBG_Lost_Meadow 2026-08-03        0.0    0.060647      0.070212           0.037553              0.002969           -0.004755
        ECE_BBG_Main_St 2026-08-03        0.0    0.050900      0.036989           0.014112             -0.007234           -0.004417
ECE_Renton_Garden_North 2026-08-03        0.0    0.135826      0.065649           0.032805             -0.002172            0.001925
 ECE_Renton_Garden_Shed 2026-08-03        0.0    0.078777      0.038982           0.018695             -0.001159            0.001507
        ECE_Renton_Home 2026-08-03        0.0    0.019126      0.025309           0.008759             -0.000255            0.000250

median hourly_range on focus days: 0.0349
median |d_target_vs_prev_day|: 0.0025
median |d_pred_vs_prev_day|: 0.0024
max |simple_minus_hourly|: 0.0031
rainy station-days: 18 of 35
median hourly_range on rainy days: 0.0287
median |d_target_vs_prev_day| on rainy days: 0.0034
```

Reading: the median within-day sensor swing (0.0349 over all 7 days, 0.0287 on rainy days) is ~10–14x the median day-to-day target step (0.0025). Crucially, the dry context days show equal or *larger* diurnal ranges than the rain days (up to 0.070 on 08-03) — the midday dry-down / overnight recovery cycle exists independent of rain, so on rain days the daily mean averages a wet night against a dry afternoon and comes out flat. The averaging *definition* is second-order (`|simple − hourly-weighted| ≤ 0.0031`); the smoothing is inherent to daily averaging itself.

Limitation: `precip_mm` is a daily Open-Meteo sum — no hourly rainfall exists in-repo — so the exact within-day timing of rain vs the sensor dip cannot be attributed here. The shapes are consistent with brief rain followed by fast peak-summer dry-down, but that causal step needs hourly precipitation.

## 5. Hourly rainfall: rain falls at night, sensors dry by day (notebook stdout)

`hourly_rain_reference.csv` (Open-Meteo archive `rain,precipitation`, fetched by `fetch_hourly_rain.py` with the same `timezone: auto` parameters as `weather_pipe.py`; 5 stations x 33 days x 24 h = 3960 rows) is the identical source as the model's daily input — its daily sums reproduce `test.csv::precip_mm` exactly:

```text
max |hourly-sum - test.csv precip_mm| = 5.55e-17
```

Per focus station-day rain timing (notebook stdout):

```text
             station_id       date  rain_total_mm  rain_peak_hour  rain_peak_mm  n_rain_hours
    ECE_BBG_Lost_Meadow 2026-07-22            0.0               0           0.0             0
        ECE_BBG_Main_St 2026-07-22            0.0               0           0.0             0
ECE_Renton_Garden_North 2026-07-22            0.0               0           0.0             0
 ECE_Renton_Garden_Shed 2026-07-22            0.0               0           0.0             0
        ECE_Renton_Home 2026-07-22            0.0               0           0.0             0
    ECE_BBG_Lost_Meadow 2026-07-23            6.1               0           3.8             7
        ECE_BBG_Main_St 2026-07-23            6.1               0           3.8             7
ECE_Renton_Garden_North 2026-07-23           10.7               0           6.7             6
 ECE_Renton_Garden_Shed 2026-07-23           10.7               0           6.7             6
        ECE_Renton_Home 2026-07-23           10.7               0           6.7             7
    ECE_BBG_Lost_Meadow 2026-07-25            0.0               0           0.0             0
        ECE_BBG_Main_St 2026-07-25            0.0               0           0.0             0
ECE_Renton_Garden_North 2026-07-25            0.5              11           0.2             3
 ECE_Renton_Garden_Shed 2026-07-25            0.5              11           0.2             3
        ECE_Renton_Home 2026-07-25            0.8              11           0.3             3
    ECE_BBG_Lost_Meadow 2026-07-26            6.2               8           0.8            18
        ECE_BBG_Main_St 2026-07-26            6.2               8           0.8            18
ECE_Renton_Garden_North 2026-07-26            6.8               8           1.0            16
 ECE_Renton_Garden_Shed 2026-07-26            6.8               8           1.0            16
        ECE_Renton_Home 2026-07-26            6.8               8           1.0            16
    ECE_BBG_Lost_Meadow 2026-07-27            0.0               0           0.0             0
        ECE_BBG_Main_St 2026-07-27            0.0               0           0.0             0
ECE_Renton_Garden_North 2026-07-27            0.0               0           0.0             0
 ECE_Renton_Garden_Shed 2026-07-27            0.0               0           0.0             0
        ECE_Renton_Home 2026-07-27            0.0               0           0.0             0
    ECE_BBG_Lost_Meadow 2026-08-02            0.8              13           0.2             7
        ECE_BBG_Main_St 2026-08-02            0.8              13           0.2             7
ECE_Renton_Garden_North 2026-08-02            2.3              16           0.6            10
 ECE_Renton_Garden_Shed 2026-08-02            2.3              16           0.6            10
        ECE_Renton_Home 2026-08-02            1.3              14           0.3             7
    ECE_BBG_Lost_Meadow 2026-08-03            0.0               0           0.0             0
        ECE_BBG_Main_St 2026-08-03            0.0               0           0.0             0
ECE_Renton_Garden_North 2026-08-03            0.0               0           0.0             0
 ECE_Renton_Garden_Shed 2026-08-03            0.0               0           0.0             0
        ECE_Renton_Home 2026-08-03            0.0               0           0.0             0
```

This resolves the original puzzle: on 2026-07-23 the bulk of the rain (6.7 of 10.7 mm at Renton) falls in hours 00–01, wetting the sensors overnight; the sensors then dry steadily through the day to a midday/early-afternoon minimum despite light midday drizzle. The 24 h mean averages the wet night against the dry afternoon and comes out flat. Same pattern on 07-26 (morning rain peak at 08h, sensor minima at 13–14h). The BBG station pair shares one reanalysis grid cell (identical series), as do Garden_North/Garden_Shed; Renton_Home differs slightly.

Rain-to-sensor lag table (notebook stdout, saved to `rain_lag_summary.csv`):

```text
             station_id       date  rain_total_mm  rain_peak_hour  sensor_max_hour  sensor_min_hour  min_lag_after_peak_h
    ECE_BBG_Lost_Meadow 2026-07-22            0.0             NaN               22               13                   NaN
        ECE_BBG_Main_St 2026-07-22            0.0             NaN               21               13                   NaN
ECE_Renton_Garden_North 2026-07-22            0.0             NaN                4               15                   NaN
 ECE_Renton_Garden_Shed 2026-07-22            0.0             NaN                5               15                   NaN
        ECE_Renton_Home 2026-07-22            0.0             NaN                5               14                   NaN
    ECE_BBG_Lost_Meadow 2026-07-23            6.1             0.0                4               14                  14.0
        ECE_BBG_Main_St 2026-07-23            6.1             0.0               23                0                   0.0
ECE_Renton_Garden_North 2026-07-23           10.7             0.0                3               15                  15.0
 ECE_Renton_Garden_Shed 2026-07-23           10.7             0.0                4               15                  15.0
        ECE_Renton_Home 2026-07-23           10.7             0.0                7               14                  14.0
    ECE_BBG_Lost_Meadow 2026-07-25            0.0             NaN                5               13                   NaN
        ECE_BBG_Main_St 2026-07-25            0.0             NaN                5               13                   NaN
ECE_Renton_Garden_North 2026-07-25            0.5            11.0                5               16                   5.0
 ECE_Renton_Garden_Shed 2026-07-25            0.5            11.0                5               13                   2.0
        ECE_Renton_Home 2026-07-25            0.8            11.0                5               13                   2.0
    ECE_BBG_Lost_Meadow 2026-07-26            6.2             8.0               23               13                   5.0
        ECE_BBG_Main_St 2026-07-26            6.2             8.0               23               13                   5.0
ECE_Renton_Garden_North 2026-07-26            6.8             8.0                7               14                   6.0
ECE_Renton_Garden_Shed 2026-07-26            6.8             8.0                5               14                   6.0
        ECE_Renton_Home 2026-07-26            6.8             8.0                7               14                   6.0
    ECE_BBG_Lost_Meadow 2026-07-27            0.0             NaN                2               14                   NaN
        ECE_BBG_Main_St 2026-07-27            0.0             NaN                1               13                   NaN
ECE_Renton_Garden_North 2026-07-27            0.0             NaN                0               15                   NaN
 ECE_Renton_Garden_Shed 2026-07-27            0.0             NaN                0               14                   NaN
        ECE_Renton_Home 2026-07-27            0.0             NaN                3               14                   NaN
    ECE_BBG_Lost_Meadow 2026-08-02            0.8            13.0               23               16                   3.0
        ECE_BBG_Main_St 2026-08-02            0.8            13.0                6               18                   5.0
ECE_Renton_Garden_North 2026-08-02            2.3            16.0                3               10                  18.0
 ECE_Renton_Garden_Shed 2026-08-02            2.3            16.0                4                9                  17.0
        ECE_Renton_Home 2026-08-02            1.3            14.0                7               15                   1.0
    ECE_BBG_Lost_Meadow 2026-08-03            0.0             NaN                8               13                   NaN
        ECE_BBG_Main_St 2026-08-03            0.0             NaN                4               12                   NaN
ECE_Renton_Garden_North 2026-08-03            0.0             NaN                5               15                   NaN
 ECE_Renton_Garden_Shed 2026-08-03            0.0             NaN                5               13                   NaN
        ECE_Renton_Home 2026-08-03            0.0             NaN                6               14                   NaN
```

Note the dry context days replicate the same timing signature with no rain at all — sensor minima at 12–15h, maxima overnight/morning — confirming the cycle is diurnal, not rain-driven.

Pooled hour-to-hour sensor change vs lagged rain is now essentially zero at every lag (notebook stdout):

```text
pooled corr(d_sensor[h], rain[h-lag]) over focus station-hours:
  lag 0h: r=+0.019 (n=831)
  lag 1h: r=+0.010 (n=826)
  lag 2h: r=-0.001 (n=821)
  lag 3h: r=+0.010 (n=816)
  lag 4h: r=+0.007 (n=811)
  lag 5h: r=+0.004 (n=806)
  lag 6h: r=+0.018 (n=801)
```

So the sensor has no fast hour-scale kick from rain; the diurnal dry-down cycle dominates hour-to-hour moves, and rain shows up as the overnight wet state that the day then dries out. Caveat: Open-Meteo precipitation is ERA5-based reanalysis (~11 km grid), not a rain gauge — authoritative for *timing* relative to the model's own input, but not independent ground truth.

## Figures (solely generated by the notebook)

- `figures/hourly_event_A_20260722_20260723.png`: 07-22→07-23, all 5 stations — hourly sensor mean, daily-mean segment (model input), hourly-weighted segment, Best-1.1/daily-target noon markers, per-day precip/coverage labels.
- `figures/hourly_event_B_20260725_20260726_20260727.png`: 07-25→07-27 three-day window, same layout.
- `figures/hourly_event_C_20260802_20260803.png`: 08-02→08-03, same layout.
- `figures/hourly_rain_event_A_20260722_20260723.png`, `figures/hourly_rain_event_B_20260725_20260726_20260727.png`, `figures/hourly_rain_event_C_20260802_20260803.png`: same windows with hourly Open-Meteo precipitation bars on a twin rain axis — rain timing vs sensor dry-down.
- `fetch_hourly_rain.py`: versioned fetch script for `hourly_rain_reference.csv` (same Open-Meteo params as `weather_pipe.py`).

## Reproduction

Fetch the hourly rain reference (cached; uses `--refresh` to refetch):

```bash
uv run python experiment/ece-input-diagnose-1.0/fetch_hourly_rain.py
```

Then, from `notebooks/`, run:

```bash
nb execute experiment/ece-input-diagnose-1.0/ece-input-diagnose-1.0.ipynb --uv
```

Versioned outputs are `hourly_comparison.csv` (836 hourly rows), `focus_coverage.csv`, `smoothing_summary.csv`, `hourly_rain_reference.csv` (3960 hourly rows), `rain_lag_summary.csv`. Config is `config.yaml`.
