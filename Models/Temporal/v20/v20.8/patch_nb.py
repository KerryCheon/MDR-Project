"""Patch MDR-v20.8.ipynb: cells 27, 29, 31, 33, 34, 35, 37."""
import json, sys

nb_path = 'Models/Temporal/v20/v20.8/MDR-v20.8.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

def set_cell(idx, src):
    cells[idx]['source'] = [src]

# ── Cell 27: full data download + caching ─────────────────────────────────────
set_cell(27, r"""# ── Initialise GEE ────────────────────────────────────────────────────────────
init_gee(GEE_PROJECT)

MANUAL_DIR = OUTPUT_ROOT / 'manual_sm'
MANUAL_DIR.mkdir(exist_ok=True)

station_featured = {}   # station_id -> featured DataFrame (DATE_START to DATE_END)
skipped = []

for _, row in eval_meta.iterrows():
    sid      = row['station_id']
    lat, lon = row['lat'], row['lon']
    state    = row['state']
    network  = row['network']

    print(f'\n{"="*60}\n  {sid}  [{network}]\n{"="*60}')

    # ── Check feature cache (must span full eval period) ──────────────────────
    cache_path = CACHE_DIR / f'{sid}_featured.csv'
    if cache_path.exists():
        _c = pd.read_csv(cache_path, parse_dates=['date'])
        _c['date'] = _c['date'].dt.strftime('%Y-%m-%d')
        if _c['date'].min() <= DATE_START and _c['date'].max() >= DATE_END:
            station_featured[sid] = _c
            n_lab = _c[TARGET_COL].notna().sum()
            print(f'  Loaded from cache: {len(_c)} rows | labelled: {n_lab}')
            continue
        else:
            print('  Cache stale — re-fetching')

    # ── 1. Ground truth SM ─────────────────────────────────────────────────────
    sm_df = pd.DataFrame()
    if network in ('SNOTEL', 'SCAN'):
        triplet = awdb_find_station(sid, state, network)
        if triplet:
            print(f'  AWDB triplet: {triplet}')
            sm_df = awdb_get_soil_moisture(triplet, FETCH_START, DATE_END)
        else:
            print(f'  WARNING: could not find {sid} in AWDB — trying shorter name')
            for name_try in [sid.replace('_', ' '), sid.split('_')[0]]:
                triplet = awdb_find_station(name_try, state, network)
                if triplet:
                    sm_df = awdb_get_soil_moisture(triplet, FETCH_START, DATE_END)
                    break

    elif network == 'USCRN':
        code  = USCRN_STATION_CODES.get(sid, sid)
        sm_df = uscrn_get_soil_moisture(code, FETCH_START, DATE_END)

    elif network in ('COSMOS', 'PBO-H2O'):
        manual_path = MANUAL_DIR / f'{sid}.csv'
        if manual_path.exists():
            sm_df = pd.read_csv(manual_path, parse_dates=['date'])
            sm_df['date'] = sm_df['date'].dt.strftime('%Y-%m-%d')
            print(f'  Loaded manual SM from {manual_path.name}')
        else:
            print(f'  SKIP: {sid} ({network}) — place CSV in {MANUAL_DIR}')
            skipped.append(sid)
            continue

    if sm_df.empty:
        print(f'  SKIP: no soil moisture data for {sid}')
        skipped.append(sid)
        continue

    n_sm = sm_df['soil_moisture_5cm'].notna().sum() if 'soil_moisture_5cm' in sm_df.columns else len(sm_df)
    print(f'  SM rows: {len(sm_df)} ({sm_df["date"].min()} -> {sm_df["date"].max()})  valid: {n_sm}')

    # ── 2. Satellite features ─────────────────────────────────────────────────
    print('  Fetching SMAP...')
    smap_df = fetch_smap(lat, lon, FETCH_START, DATE_END)
    print('  Fetching Sentinel-1...')
    s1_df   = fetch_sentinel1(lat, lon, FETCH_START, DATE_END)
    print('  Fetching MODIS LST...')
    lst_df  = fetch_modis_lst(lat, lon, FETCH_START, DATE_END)
    print('  Fetching Sentinel-2...')
    s2_df   = fetch_sentinel2(lat, lon, FETCH_START, DATE_END)

    # ── 3. Precipitation ──────────────────────────────────────────────────────
    print('  Fetching precipitation...')
    prec_df = fetch_precipitation(lat, lon, FETCH_START, DATE_END)

    # ── 4. Static features ────────────────────────────────────────────────────
    print('  Fetching static features...')
    static  = fetch_static_features(lat, lon)
    print(f'  elev={static["elev"]:.0f}m  slope={static["slope"]:.1f}  '
          f'clay={static["J_clay_wfrac_b0"]:.0f}  sand={static["J_sand_wfrac_b0"]:.0f}')

    # ── 5. Build daily scaffold and merge ─────────────────────────────────────
    date_range = pd.date_range(FETCH_START, DATE_END, freq='D')
    scaffold   = pd.DataFrame({'date': date_range.strftime('%Y-%m-%d')})

    def _merge(base, src):
        return base if src.empty else base.merge(src, on='date', how='left')

    daily = scaffold.copy()
    daily = _merge(daily, sm_df)
    daily = _merge(daily, smap_df)
    daily = _merge(daily, lst_df)
    daily = _merge(daily, s2_df)
    daily = _merge(daily, s1_df)
    daily = _merge(daily, prec_df)

    # ── 6. Feature engineering ────────────────────────────────────────────────
    daily = engineer_features(daily, static)
    daily['station_id'] = sid
    daily['latitude']   = lat
    daily['longitude']  = lon

    # ── Feature coverage diagnostic ───────────────────────────────────────────
    sat_check = {
        'SMAP': ['SMAP_sm_pm_interp_ema02', 'SMAP_sm_interp_grad7'],
        'LST':  ['V_ema_LST_modis_kobs7'],
        'S2':   ['V_rollmean_s2_b11_kobs7'],
        'SAR':  ['A_d_E_SAR_diff_kobs14'],
    }
    for _src, _cols in sat_check.items():
        _valid_cols = [c for c in _cols if c in daily.columns]
        if not _valid_cols:
            continue
        _nan_pct = daily[_valid_cols].isna().mean().mean()
        _flag = '  *** >50% NaN ***' if _nan_pct > 0.5 else ''
        print(f'  {_src} coverage: {100*(1-_nan_pct):.0f}%{_flag}')

    # ── 7. Clip to eval period and check label count ──────────────────────────
    full_df    = daily[(daily['date'] >= DATE_START) & (daily['date'] <= DATE_END)].copy()
    n_labelled = full_df[TARGET_COL].notna().sum()
    print(f'  Eval-period rows: {len(full_df)}  |  labelled: {n_labelled}')

    if n_labelled < 30:
        print(f'  SKIP: fewer than 30 labelled days')
        skipped.append(sid)
        continue

    # ── 8. Cache ──────────────────────────────────────────────────────────────
    full_df.to_csv(cache_path, index=False)
    station_featured[sid] = full_df
    print(f'  Cached -> {cache_path.name}')

print(f'\n--- Done ---  stations ready: {len(station_featured)}  skipped: {skipped}')
""")

# ── Cell 29: LOSO training loop ───────────────────────────────────────────────
set_cell(29, r"""# ── Load full WA data (train + val + test) ─────────────────────────────────────
wa_df = pd.concat([
    pd.read_csv(SPLIT_ROOT / SPLIT / 'train.csv'),
    pd.read_csv(SPLIT_ROOT / SPLIT / 'val.csv'),
    pd.read_csv(SPLIT_ROOT / SPLIT / 'test.csv'),
], ignore_index=True)
print(f'WA data: {len(wa_df):,} rows | stations: {sorted(wa_df["station_id"].unique())}')

missing_wa = [f for f in ALL_FEATS if f not in wa_df.columns]
print('WA missing features:', missing_wa if missing_wa else 'NONE ✓')

# ── LOSO loop ─────────────────────────────────────────────────────────────────
loso_results = []
loso_preds   = {}   # station_id -> (y_true, y_pred, dates)
eval_sids    = list(station_featured.keys())

print(f'\nLOSO over {len(eval_sids)} stations')

for sid in eval_sids:
    print(f'\n{"="*60}\n  LOSO: holding out {sid}\n{"="*60}')

    test_df = station_featured[sid].dropna(subset=[TARGET_COL]).copy()

    # Train = WA + all other eval stations
    other_dfs = [
        station_featured[oid].dropna(subset=[TARGET_COL])
        for oid in eval_sids if oid != sid
    ]
    other_concat = pd.concat(other_dfs, ignore_index=True) if other_dfs else pd.DataFrame()
    train_df = pd.concat(
        [wa_df, other_concat] if not other_concat.empty else [wa_df],
        ignore_index=True
    ).dropna(subset=[TARGET_COL])

    print(f'  Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows')

    # Train regime ensemble
    y_tv      = train_df[TARGET_COL].values
    X_tv_base = train_df[FEATURE_COLS_BASE]
    X_tv_wet  = train_df[FEATURE_COLS_WET]

    reg_tv        = label_regime(y_tv)
    mask_trans_tv = reg_tv == 1
    mask_wet_tv   = reg_tv == 2
    print(f'  Regime — dry:{(reg_tv==0).sum():,}  trans:{mask_trans_tv.sum():,}  wet:{mask_wet_tv.sum():,}')

    _xgb_base = XGBRegressor(**XGB_PARAMS_DRY)
    _xgb_base.fit(X_tv_base, y_tv, verbose=0)
    _pred_base_tv = _xgb_base.predict(X_tv_base)

    X_tv_aug   = np.column_stack([X_tv_wet, _pred_base_tv])
    _xgb_trans = XGBRegressor(**XGB_PARAMS_TRANSITION)
    _xgb_trans.fit(X_tv_aug[mask_trans_tv], y_tv[mask_trans_tv], verbose=0)

    _xgb_wet = XGBRegressor(**XGB_PARAMS_WET)
    _xgb_wet.fit(X_tv_aug[mask_wet_tv], y_tv[mask_wet_tv], verbose=0)

    # Predict on held-out station
    X_test_base = test_df[FEATURE_COLS_BASE].values
    X_test_wet  = test_df[FEATURE_COLS_WET].values
    _pb         = _xgb_base.predict(X_test_base)
    X_test_aug  = np.column_stack([X_test_wet, _pb])
    _pt         = _xgb_trans.predict(X_test_aug)
    _pw         = _xgb_wet.predict(X_test_aug)
    y_pred      = make_final_pred(_pb, _pt, _pw, np.zeros(len(test_df)))
    y_true      = test_df[TARGET_COL].values

    m = get_metrics(y_true, y_pred)
    print(f'  R2={m["r2"]:+.4f}  RMSE={m["rmse"]:.4f}  ubRMSE={m["ubrmse"]:.4f}  Bias={m["bias"]:+.4f}  n={m["n"]}')

    row = {'station': sid, **m}
    reg_h = label_regime(y_true)
    for rname, ridx in [('dry', 0), ('transition', 1), ('wet', 2)]:
        mask = reg_h == ridx
        if mask.sum() > 1:
            rm = get_metrics(y_true[mask], y_pred[mask])
            for k, v in rm.items():
                row[f'{rname}_{k}'] = v

    loso_results.append(row)
    loso_preds[sid] = (y_true, y_pred, pd.to_datetime(test_df['date'].values))

print(f'\nLOSO complete — {len(loso_results)} stations')
""")

# ── Cells 31, 33, 34, 35, 37: rename spatial_* -> loso_* ─────────────────────
renames = [
    ('spatial_results', 'loso_results'),
    ('spatial_preds',   'loso_preds'),
    ('spatial_eval_summary.csv', 'loso_eval_summary.csv'),
    ('spatial_scatter.png', 'loso_scatter.png'),
    ('spatial_r2_bar.png',  'loso_r2_bar.png'),
    ('spatial_timeseries.png', 'loso_timeseries.png'),
    ('Zero-shot', 'LOSO'),
    ('zero-shot', 'LOSO'),
    ('v20.8 LOSO (OR/ID/MT)', 'v20.8 LOSO (cross-state)'),
    ("'v20.8 Zero-shot (OR/ID/MT)'", "'v20.8 LOSO (cross-state)'"),
]
for idx in [31, 33, 34, 35, 37]:
    src = ''.join(cells[idx]['source'])
    for old, new in renames:
        src = src.replace(old, new)
    set_cell(idx, src)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('All cells patched and saved.')
