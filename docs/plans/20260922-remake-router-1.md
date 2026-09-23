## W2 guard spec — `derived_8.4-routing-optimize-1.0/`

### 0. Objective / non-goals
* Keep **full-covariate routing narrative**: router features = exactly `shared_backbone_54` (no hand-picked subset, no V0-50 import).
* Add a deterministic, trainval-only **station-consistency guard** that guarantees `same station → same regime` for known stations while staying automatic for new regions.
* Non-goals: no delta-feature search, no K>2 campaign, no OOS/ECE retraining, no edits to `writeup/`, `paper/`, archives, or existing experiments. One new config family only: `Guarded_Backbone54_k2 (0,0)` vs existing `Backbone54 (0,0)` + `V0 (0,0)` references.

### 1. Variants to implement (exactly 2, pick one as primary)
* **V-A — Majority-vote guard (recommended primary):** per-sample `Backbone54Router` fit as today (`mean-impute → StandardScaler → KMeans(2, seed 42, n_init 10)` on 54 backbone, fit on fold-trainval only); then per-station majority label computed on the **fit frame only**, broadcast at predict time.
* **V-B — Station-mean guard (challenger):** same fit machinery, but router decision made on station-mean feature vector (mean of 54 backbone cols per `station_id` on fit frame → single `kmeans.predict` per station → broadcast). Tests whether "route the site, not the sample" is cleaner/more stable across LOSO folds.
* Both reuse `SalvagedKMeansRouter` patterns from `formal-eval-2.1-ece-v3/eval_formal/routers.py:170-263` for `tau=0.10` availability gate + WA-p5 margin; no new thresholds invented on test.

### 2. Formal fit/predict semantics
```
fit(train: DataFrame with station_id + 54 backbone):
  1. values = train[backbone54]; means = col means; X = scaler.fit_transform(fillna(means))
  2. kmeans.fit(X); aux GAPI router.fit(train)
  3. margins = |d0-d1| on X; p5 = percentile(margins,5); median_margin = median(margins)
  4a. V-A: labels = kmeans.predict(X); station_majority[s] = mode(labels[train.station_id==s])
      tie → smaller cluster id after canonicalization (see §3); record majority_strength[s] = max share
  4b. V-B: station_means[s] = mean(values[station==s]); station_label[s] = kmeans.predict(scaler.transform(station_means[s]))
  5. persist {features, means, scaler, kmeans, aux_threshold, p5, median_margin, station_majority/means, version}

predict(frame: DataFrame with optional station_id):
  1. static = kmeans.predict(scaler.transform(fillna(means)))
  2. gated = full-SMAP-block-missing OR miss_rate > tau → aux GAPI label (same availability_gate as salvage)
  3. if station_id present AND s in station_majority/means AND not gated:
       V-A → station_majority[s]; V-B → station_label[s]
     else (unseen station / missing id / gated):
       per-sample static label; if margin < p5 → route to Global_Single_54 fallback expert (margin_fallback policy)
  4. return hard int labels; optional predict_weights mirrors salvage softmax T=0.25 (diagnostic only)
```
* LOSO rule: refit per fold on 6-station fold-trainval; held-out station is **unseen by construction** → always takes unseen path (tests transfer). No held-out leakage into `means/scaler/centroids/majority/p5`.
* Temporal rule: fit once on full 7-station trainval; test rows of known stations take majority path.

### 3. Label canonicalization (required for cross-fold comparison)
* After each fit, canonicalize cluster ids: `c1 = cluster with lower mean SMAP_sm_pm_interp_rollmean30` (eastern/drier group per `gating-analysis README:173-195`); swap labels if needed. Record `label_flip_applied: bool`.
* Prevents ARI/agreement artifacts from arbitrary KMeans numbering; does not touch geometry.

### 4. K-selection (trainval-only, automatic story)
* Ship `k_sweep = {2,3,4}` diagnostic in notebook (same recipe, `n_clusters=K`): report trainval-only silhouette / Calinski-Harabasz / Davies-Bouldin / purity / balance / val-period expert R²; test purity shown as "out-of-sample confirmation only" (per `audit.md:23-24`). Paper claims K=2 from trainval; no test in selection.

### 5. New experiment layout (mirror `eval-1.4` / `formal-eval-1.0` conventions)
```
notebooks/experiment/derived_8.4-routing-optimize-1.0/
  config.yaml          # data paths (=derived_8.4), shared_backbone_54 (import, not redefined),
                       # pinned configs: GuardedV-A (0,0), GuardedV-B (0,0); refs to Backbone54/V0/Global
                       # XGBoost exact_params (=2500-tree SOTA 1.5), seeds {temporal 30, LOSO 5×7},
                       # tau=0.10, T=0.25, router_seed=42
  routing_opt/
    routers.py         # GuardedBackbone54Router(V-A), StationMeanBackbone54Router(V-B) + get_router
    data.py            # reuse eval14/eval_formal loader (train/val/test, v0+backbone lists)
    jobs.py            # resumable per-(config,seed,station) jobs, data_version, meta.json
  run_temporal.py / run_loso.py / run_worker.py  # same driver format as formal-eval-1.0
  analyze_agreement.py # V0 vs Backbone vs GuardedV-A vs GuardedV-B: ARI/agree, sizes, crosstabs, p5/margins
  derived_8.4-routing-optimize-1.0.ipynb  # sole figure/table generator
  README.md            # stdout-only tables (see §6)
  pinned_configurations.json  # written before any run (audit trail)
```

### 6. Evaluation + README tables (stdout only via `nb execute --uv`)
1. Agreement: full-trainval ARI/crosstab (expect GuardedV-A ≡ Backbone ARI=1 on full fit; V-B near-1); per-LOSO-fold ARI/agree trainval + held-out test.
2. Purity: trainval/test station purity (Guarded = 1.000 by construction on known stations; report held-out separately).
3. Temporal 30-seed: `GuardedV-A/B (0,0)` vs `Backbone (0,0)` vs `V0 (0,0)` vs `Global_54` — R²/RMSE/MAE/bias + paired t/Wilcoxon + BH-FDR + (station,month) bootstrap.
4. LOSO 5-seed: mean R², win counts vs Global (sign test n=7), per-station bars; seed-42 must reproduce `Backbone (0,0) 0.6171 / V0 (0,0) 0.6343` baselines.
5. Margin/gate diagnostics: p5 values per fold, % gated, % fallback, majority_strength distribution.
6. K-sweep trainval table (§4).

### 7. Acceptance / promotion rule
* Promote `GuardedV-A` to paper primary iff: temporal within ±0.003 of `V0 (0,0)` AND LOSO within ±0.010 or better AND purity 1.000 on known stations AND unseen-path fallback rate sane (<15% gated/ambiguous). Else keep unguarded `Backbone54` primary, report guard as null-method result — automatability claim still holds.
* `V-B` promotes only if it beats `V-A` on LOSO by >0.005 with equal temporal; else appendix.

### 8. Risks
* Majority guard is identity on full-trainval (already pure) — gain only materializes across LOSO refits/OOD; possible null on temporal.
* Station-mean needs warmup rule for new sites (~30-obs `kobs` windows + SMAP latency); document as deployment requirement, snow/SWE sites out-of-scope.
* Shared-backbone test-selection caveat (R4) unchanged — relative-claims framing stays.

Build-mode entry: copy `eval14/routers.py:_KMeansRouter` + `ece-v3/routers.py:availability_gate/margin` into `routing_opt/routers.py`, add §2–§3 logic, smoke-test (`--smoke`, CPU, `data_version=-1`), then full GPU run + `nb execute experiment/derived_8.4-routing-optimize-1.0/... --uv`.