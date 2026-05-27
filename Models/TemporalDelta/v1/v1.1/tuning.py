# --- Round 2: Additional hyperparameter tuning ---
GRID2 = [
    dict(n_estimators=1600, max_depth=None, min_samples_split=4, min_samples_leaf=2, max_features="sqrt"),
    dict(n_estimators=800,  max_depth=None, min_samples_split=4, min_samples_leaf=3, max_features="sqrt"),
    dict(n_estimators=800,  max_depth=None, min_samples_split=4, min_samples_leaf=4, max_features="sqrt"),
    dict(n_estimators=800,  max_depth=None, min_samples_split=3, min_samples_leaf=2, max_features="sqrt"),
    dict(n_estimators=800,  max_depth=25,   min_samples_split=4, min_samples_leaf=2, max_features="sqrt"),
    dict(n_estimators=800,  max_depth=30,   min_samples_split=4, min_samples_leaf=2, max_features="sqrt"),
    dict(n_estimators=800,  max_depth=None, min_samples_split=4, min_samples_leaf=2, max_features="log2"),
    dict(n_estimators=800,  max_depth=None, min_samples_split=4, min_samples_leaf=2, max_features=0.6),
    dict(n_estimators=400,  max_depth=None, min_samples_split=4, min_samples_leaf=2, max_features="sqrt"),
    dict(n_estimators=800,  max_depth=None, min_samples_split=4, min_samples_leaf=2, max_features="sqrt", bootstrap=False),
]

rows2 = []

for i, params in enumerate(GRID2):
    label = f"[{i+1}/{len(GRID2)}] n_est={params['n_estimators']}, depth={params['max_depth']}, split={params['min_samples_split']}, leaf={params['min_samples_leaf']}, mf={params['max_features']}"
    if 'bootstrap' in params:
        label += f", bootstrap={params['bootstrap']}"
    print(f"\n{label}")
    t0 = time.time()
    rf = RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train_delta)
    t1 = time.time()
    print(f"  trained in {t1-t0:.1f}s")

    vdp = rf.predict(X_val)
    tdp = rf.predict(X_test)

    vdr2 = r2_score(y_val_delta, vdp)
    tdr2 = r2_score(y_test_delta, tdp)

    vrr2 = r2_score(val_df[TARGET_COL], y_val_lag1 + vdp)
    trr2 = r2_score(test_df[TARGET_COL], y_test_lag1 + tdp)

    print(f"  Val \u0394SM R\u00b2={vdr2:.4f} | Test \u0394SM R\u00b2={tdr2:.4f}")
    print(f"  Val Recon R\u00b2={vrr2:.4f} | Test Recon R\u00b2={trr2:.4f}")

    row = {**params, 'val_delta_r2': vdr2, 'test_delta_r2': tdr2, 'val_recon_r2': vrr2, 'test_recon_r2': trr2}
    row['bootstrap'] = params.get('bootstrap', True)
    rows2.append(row)

results2_df = pd.DataFrame(rows2)
results2_df.to_csv(f"{OUTPUT_ROOT}/hyperparameter_tuning_round2.csv", index=False)

print("\n\n=== ROUND 2 GRID SEARCH RESULTS ===")
print("|n_estimators|max_depth|min_samples_split|min_samples_leaf|max_features|bootstrap|Val \u0394SM R^2|Test \u0394SM R^2|Val Recon SM R^2|Test Recon SM R^2|")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows2:
    bs = r.get('bootstrap', True)
    print(f"|{r['n_estimators']}|{r['max_depth']}|{r['min_samples_split']}|{r['min_samples_leaf']}|{r['max_features']}|{bs}|{r['val_delta_r2']:.4f}|{r['test_delta_r2']:.4f}|{r['val_recon_r2']:.4f}|{r['test_recon_r2']:.4f}|")

best_all = max(rows2, key=lambda r: r['val_delta_r2'])
print(f"\nBest config in round 2 by Val \u0394SM R\u00b2 ({best_all['val_delta_r2']:.4f}):")
print(f"  n_estimators={best_all['n_estimators']}, max_depth={best_all['max_depth']}, min_samples_split={best_all['min_samples_split']}, min_samples_leaf={best_all['min_samples_leaf']}, max_features={best_all['max_features']}, bootstrap={best_all.get('bootstrap', True)}")
