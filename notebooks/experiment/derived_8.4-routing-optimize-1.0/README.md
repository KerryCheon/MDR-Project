# Experiment: `derived_8.4-routing-optimize-1.0` — Guarded backbone routing

## Objective

Decide the paper-wide primary router without reintroducing a hand-picked feature
story: keep routing features exactly the shared 54-feature backbone
(full-covariate routing) and add a deterministic, trainval-only
station-consistency guard. Compares `Backbone54` (unguarded reference),
`GuardedV-A` (majority-vote guard, primary candidate), `StationMeanV-B`
(station-mean guard, challenger), and `V0_Full` (legacy sensitivity only).

Scope is routing diagnostics ONLY (CPU, no XGBoost training): agreement /
purity / K-sweep / margin-fallback diagnostics. Expert training (temporal /
LOSO) reuses `derived_8.4-formal-eval-1.0` drivers and is out of scope here.

## Design (W2 spec)

See `routing_opt/routers.py` for the exact semantics:

- Fit: `mean-impute -> StandardScaler -> KMeans(2, seed 42, n_init 10)` on the
  54 backbone, fit frame only (full trainval, or 6-station fold trainval under
  LOSO). Canonicalize labels so `c1` is the drier regime (lower fit-frame mean
  `SMAP_sm_pm_interp_rollmean30`). Calibrate `margin_p5` + GAPI aux threshold
  on the fit frame only (`tau=0.10`, salvage pattern).
- `GuardedV-A.predict`: known, non-gated stations get their fit-frame majority
  label (deterministic tie-break: smaller canonical id); unseen / missing /
  gated rows fall back to per-sample static prediction (`fallback_mask`
  additionally flags unseen + gated/ambiguous rows for global-expert fallback).
- `StationMeanV-B.predict`: known stations get one label from a KMeans predict
  on the station-mean vector, broadcast to all rows.
- LOSO: router refit per fold on fold-trainval; held-out station is unseen by
  construction (no leakage into means/scaler/centroids/majority/p5).

## Run

From `notebooks/` (uv env):

```bash
uv run python experiment/derived_8.4-routing-optimize-1.0/analyze_agreement.py
nb execute experiment/derived_8.4-routing-optimize-1.0/derived_8.4-routing-optimize-1.0.ipynb --uv --timeout 600
```

## Results (stdout of the executed notebook, `nb execute --uv`)

Executed: `nb execute experiment/derived_8.4-routing-optimize-1.0/derived_8.4-routing-optimize-1.0.ipynb --uv --timeout 600`
from `notebooks/`. Split sizes: train=9803 / val=4805 / test=6620 / trainval=14608.
All routers `KMeans(2, seed 42, n_init 10)`, canonicalized so `c1` = drier regime.
Calibrated margins (fit frame only): V0 `margin_p5=1.956`, Backbone/Guarded/StationMean `1.5481`
— reproducing the salvage thresholds (`ece-router-salvage` WA `{'v0': 1.956, 'backbone': 1.548}`).

### R1. Feature overlap (V0=50, backbone54=54, overlap=17)

V0-only (33, legacy `derived_8.2` inheritance dropped by the guard story):

```text
A_d_E_SAR_diff_kobs30, A_d_LST_modis_kobs14, A_d_LST_modis_kobs30, A_d_SMAP_sm_interp_kobs5,
A_d_s2_b11_kobs30, A_grad_LST_modis_kobs14, A_grad_LST_modis_kobs30, A_grad_SMAP_sm_interp_kobs30,
C_lag_F_NDMI_kobs30, D_sa_F_NDMI, D_sa_LST_modis, D_z_E_SAR_ratio, F_MSI, J_bio_bio15,
J_clay_wfrac_b100, K_aspect_cos, K_aspect_sin, V_ema_G_API_kobs30, V_ema_SMAP_sm_interp_kobs30,
V_rollmax_G_API_kobs30, V_rollmax_G_API_kobs7, V_rollmean_F_NDMI_kobs30, V_rollmin_G_API_kobs14,
V_rollrng_E_SAR_ratio_kobs30, V_rollrng_F_NDVI_kobs14, V_rollrng_F_NDVI_kobs30, V_rollrng_G_API_kobs30,
V_rollrng_LST_modis_kobs30, aspect, latitude, lia_std_asc_deg, s2_b11, slope
```

Backbone-only (37):

```text
A_d_E_SAR_ratio_kobs30, D_cos_DOY, D_fft_dom_LST_modis_kobs30, D_fft_ent_LST_modis_kobs30,
D_sin_DOY, D_z_F_NDMI, D_z_LST_modis, E_rough_s1_vh_kobs14, G_API, G_DSLR, G_rain_sum_3d,
G_rain_sum_7d, J_bio_bio13, J_lc_code, J_soil_texture_usda_b0, SMAP_ampm_diff_interp,
SMAP_sm_interp_rollrange7, SMAP_sm_pm_interp, SMAP_sm_pm_interp_lag7, SMAP_sm_pm_interp_rollrange7,
V_rollmax_E_SAR_diff_kobs14, V_rollmax_E_SAR_diff_kobs30, V_rollmax_F_NDMI_kobs30,
V_rollmax_F_NDVI_kobs14, V_rollmax_F_NDVI_kobs30, V_rollmin_E_SAR_ratio_kobs30,
V_rollmin_LST_modis_kobs30, V_rollmin_SMAP_sm_interp_kobs14, V_rollmin_SMAP_sm_interp_kobs30,
V_rollmin_s2_b11_kobs30, V_rollrng_E_SAR_diff_kobs30, V_rollrng_G_API_kobs7, V_rollrng_s2_b11_kobs30,
cos_year, precip_mm, s2_b4, sin_year
```

### R2. Full-trainval partitions (all identical — reproduces the gating-analysis anchor)

```text
--- trainval n=14608 ---
V0_Full        sizes=[10624, 3984] mean_purity=1.0000
Backbone       sizes=[10624, 3984] mean_purity=1.0000
GuardedV-A     sizes=[10624, 3984] mean_purity=1.0000
StationMeanV-B sizes=[10624, 3984] mean_purity=1.0000
ARI V0_Full vs Backbone: 1.0000 agree=1.0000
ARI V0_Full vs GuardedV-A: 1.0000 agree=1.0000
ARI V0_Full vs StationMeanV-B: 1.0000 agree=1.0000
ARI Backbone vs GuardedV-A: 1.0000 agree=1.0000
ARI Backbone vs StationMeanV-B: 1.0000 agree=1.0000
ARI GuardedV-A vs StationMeanV-B: 1.0000 agree=1.0000
--- test n=6620 ---
V0_Full        sizes=[4817, 1803] mean_purity=1.0000
Backbone       sizes=[4817, 1803] mean_purity=0.9997
GuardedV-A     sizes=[4817, 1803] mean_purity=1.0000
StationMeanV-B sizes=[4817, 1803] mean_purity=1.0000
ARI V0_Full vs Backbone: 0.9987 agree=0.9997
ARI V0_Full vs GuardedV-A: 1.0000 agree=1.0000
ARI V0_Full vs StationMeanV-B: 1.0000 agree=1.0000
ARI Backbone vs GuardedV-A: 0.9987 agree=0.9997
ARI Backbone vs StationMeanV-B: 0.9987 agree=0.9997
ARI GuardedV-A vs StationMeanV-B: 1.0000 agree=1.0000
```

Reading: on the full 7-station fit every router produces the same station-pure
partition (10,624/3,984). On test, unguarded Backbone mixes 2 rows
(purity 0.9997); both guards recover the V0 partition exactly (ARI=1.0).

### R3. Per-LOSO-fold agreement + fallback rates

| held_out | ARI_tr V0 vs Backbone | ARI_te V0 vs Backbone | ARI_tr Backbone vs GuardedV-A | ARI_te Backbone vs GuardedV-A | ARI_tr Backbone vs StationMeanV-B | ARI_te Backbone vs StationMeanV-B | ARI_tr GuardedV-A vs StationMeanV-B | ARI_te GuardedV-A vs StationMeanV-B | fallback_rate_te | n_te |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| BeaverPass_WA_990 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 626 |
| CayusePass_WA | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1081 |
| Darrington | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 999 |
| Paradise_WA | 0.9902 | 0 | 0.9902 | 1 | 0.9902 | 1 | 1 | 1 | 1 | 1067 |
| Quinault | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1044 |
| SourdoughGulch_WA_985 | 0.1315 | 1 | 0.3491 | 1 | 0.3383 | 1 | 0.4199 | 1 | 1 | 906 |
| Spokane | 0.1919 | 0 | 0.3297 | 1 | 0.3297 | 1 | 1 | 1 | 1 | 897 |

Caveats: `ARI_te` is degenerate on single-station test slices (chance agreement
≈ 1, so near-identical assignments score 0 — see R3b for the honest metric).
`fallback_rate_te = 1` everywhere is by construction: the held-out station is
unseen, so `fallback_mask` flags every row. The mask is advisory (marks rows
where a global-expert fallback policy may apply); `predict` still returns a
valid per-sample regime for unseen rows, which is the default `as_routed` policy.

### R3b. Held-out regime assignment per fold (share in c1 = drier regime)

| held_out | n_te | V0_Full_share_c1 | Backbone_share_c1 | GuardedV-A_share_c1 | StationMeanV-B_share_c1 |
|:--|--:|--:|--:|--:|--:|
| BeaverPass_WA_990 | 626 | 0 | 0 | 0 | 0 |
| CayusePass_WA | 1081 | 0 | 0.0139 | 0.0139 | 0.0139 |
| Darrington | 999 | 0 | 0 | 0 | 0 |
| Paradise_WA | 1067 | 0 | 0.0009 | 0.0009 | 0.0009 |
| Quinault | 1044 | 0 | 0 | 0 | 0 |
| SourdoughGulch_WA_985 | 906 | 1 | 1 | 1 | 1 |
| Spokane | 897 | 0.9989 | 1 | 1 | 1 |

Reading: held-out assignment is essentially identical across routers (≤15 rows
differ on CayusePass). The LOSO gap (V0 0.634 vs Backbone 0.617) therefore comes
from specialist-training composition on eastern-holdout folds (R3, fold-trainval
ARI 0.13/0.19), not from misrouting the held-out station.

### R4. K-sweep quality, trainval-only (K in {2,3,4})

| K | silhouette | calinski_harabasz | davies_bouldin | inertia | mean_station_purity_trainval | min_cluster_share | cluster_sizes |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 2 | 0.218834 | 3880.86 | 1.57925 | 623236 | 1 | 0.272727 | [10624, 3984] |
| 3 | 0.225664 | 3589.38 | 1.81548 | 528875 | 0.833006 | 0.264033 | [6513, 3857, 4238] |
| 4 | 0.187591 | 2855 | 2.01503 | 497221 | 0.695141 | 0.146632 | [2142, 4053, 3826, 4587] |

Reproduces `gating-analysis-1.0` exactly (CH/DB best at K=2, silhouette
marginally higher at K=3, purity 1.000/0.833/0.695). K-selection evidence a new
region recomputes from its own trainval.

Guard maps (full-trainval fit, all strengths 1.0):

| station | GuardedV-A majority | StationMeanV-B label |
|:--|--:|--:|
| BeaverPass_WA_990 | 0 | 0 |
| CayusePass_WA | 0 | 0 |
| Darrington | 0 | 0 |
| Paradise_WA | 0 | 0 |
| Quinault | 0 | 0 |
| SourdoughGulch_WA_985 | 1 | 1 |
| Spokane | 1 | 1 |

## Decision rule

Promote `GuardedV-A` to paper primary iff temporal stays within ±0.003 of
`V0 (0,0)` and LOSO within ±0.010 or better in the follow-up formal-eval run,
with purity 1.000 on known stations and sane fallback rates; else keep
unguarded `Backbone54` primary and report the guard as a null-method result.

## Caveats

- Shared 54 backbone is test-selected upstream (see methodology audit); relative
  comparisons only.
- K-selection uses trainval-only indices; test purity is confirmation only.
- New-region deployment must recompute imputation means, scaler, centroids,
  majority map, and p5 on local trainval; K=2 is not universal; snow/SWE sites
  out of scope; SMAP latency/missingness documented per the provenance table.

## Training runs (T1 temporal + L1 LOSO) — status: COMPLETE

Slurm job `2155747` (`routing-opt-1.0`, H100): COMPLETED 2026-09-22, wall
1:50:45, exit 0. 3 pinned no-delta configs × 30 temporal seeds (90 jobs) + 3 × 5
LOSO seeds × 7 stations (105 jobs), 2500-tree experts, same hyperparameters and
resume semantics as `formal-eval-1.0` (`data_version=1`).

Replication gates (slurm log): Backbone54 temporal seed-42 `0.814205` vs
eval-1.3 `0.814205` [OK]; Backbone54 LOSO seed-42 mean `0.6171` vs eval-1.3
`0.6171` [OK] — harness validated. Guard/StationMean seed-42 temporal
`0.814334` reproduces eval-1.1's V0_Full-backbone number exactly (guards ==
V0 test partition, §R2), not Backbone's `0.814205`; `config.yaml` anchors were
corrected accordingly (the slurm-log MISMATCH lines for the guards predate that
correction and are superseded by it).

```bash
cd notebooks/experiment/derived_8.4-routing-optimize-1.0
mkdir -p artifacts/slurm && sbatch run_slurm.sh   # job 2155747
# CPU smoke first (never reused: data_version=-1, n_estimators=100):
uv run python run_temporal.py --smoke --max-configs 1 --seeds 42 --n-parallel 1
uv run python run_loso.py --smoke --max-configs 1 --seeds 42 --max-stations 1 --n-parallel 1
```

Smoke results (CPU, 100 trees — sanity only, not comparable to GPU anchors):
temporal GuardedV-A s42 R² 0.5140 RMSE 0.0710 (20s); LOSO GuardedV-A s42
BeaverPass fold ok (8s).

### T1. Temporal seed-level summary (30 seeds; stdout of the executed notebook)

| config | n | mean_R2 | std_R2 | mean_RMSE | mean_MAE | mean_bias |
|:--|--:|--:|--:|--:|--:|--:|
| Clustering_Backbone54_k2_c0_0_c1_0 | 30 | 0.811724 | 0.001369 | 0.044201 | 0.033985 | 0.005975 |
| Guarded_Backbone54_k2_c0_0_c1_0 | 30 | 0.811843 | 0.001369 | 0.044187 | 0.033979 | 0.005958 |
| StationMean_Backbone54_k2_c0_0_c1_0 | 30 | 0.811843 | 0.001369 | 0.044187 | 0.033979 | 0.005958 |

Guarded-Backbone paired R² diff: +0.000119, better on 30/30 seeds (paired t
p=1.42e-28 — significant only because seed noise is tiny; honestly a tie).
Guarded vs formal `V0_Full (0,0)` 0.8118: +0.000043 (W2 bar ±0.003: PASS).
Backbone reproduces formal-eval's 0.8117 ± 0.0014 exactly.

### L1. LOSO summary (5 seeds × 7 stations; stdout of the executed notebook)

| config | loso_mean_R2 | seed_means |
|:--|--:|:--|
| Clustering_Backbone54_k2_c0_0_c1_0 | 0.6199 | [0.6234, 0.6225, 0.6194, 0.6171, 0.6169] |
| Guarded_Backbone54_k2_c0_0_c1_0 | 0.6379 | [0.6391, 0.6419, 0.6371, 0.6356, 0.6357] |
| StationMean_Backbone54_k2_c0_0_c1_0 | 0.6336 | [0.6356, 0.6366, 0.6316, 0.6306, 0.6337] |

Guarded-Backbone per-held-out-station mean R² diff (5 seeds):

| station | mean | std |
|:--|--:|--:|
| BeaverPass_WA_990 | 0 | 0 |
| CayusePass_WA | 0 | 0 |
| Darrington | 0 | 0 |
| Paradise_WA | 0.0154 | 0.0097 |
| Quinault | 0 | 0 |
| SourdoughGulch_WA_985 | 0.0591 | 0.0085 |
| Spokane | 0.0514 | 0.0046 |

Guarded wins 5/5 seeds overall; gains concentrate exactly on the folds §R3
flagged (eastern holdouts + Paradise), other folds bit-identical. Guarded vs
formal `V0_Full (0,0)` 0.6372: +0.0007 (W2 bar ±0.010: PASS).

### W2 verdict: PROMOTE `Guarded_Backbone54_k2 (0,0)` to paper primary

Both bars pass with margin: temporal +0.00004 (bar ±0.003), LOSO +0.0007 (bar
±0.010). The guard closes the full Backbone→V0 LOSO gap (+0.018) while routing
on exactly the shared 54 backbone — no legacy 50-feature story needed.
`StationMeanV-B` (0.6336) is kept as challenger/appendix; `V0_Full` demoted to
legacy sensitivity reference.

## Files

- `config.yaml` — splits, pinned 54 list, router seed/tau, K-sweep, anchors.
- `pinned_configurations.json` — written before any run (audit trail).
- `routing_opt/routers.py` — guard implementations + `get_router`.
- `routing_opt/data.py` — split loader (same semantics as eval14).
- `routing_opt/agreement.py` — ARI/agreement/purity/K-sweep helpers.
- `analyze_agreement.py` — CLI printing R1–R4.
- `derived_8.4-routing-optimize-1.0.ipynb` — executed report notebook.
