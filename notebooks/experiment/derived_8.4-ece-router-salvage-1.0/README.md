# Experiment: `derived_8.4-ece-router-salvage-1.0`

Router-only salvage for `Clustering_V0_Full_k2` and `Clustering_Backbone54_k2`
(`c0_0_c1_0`, 54 backbone features, no deltas) on the 5 in-situ ECE stations.
Experts are frozen (fit on WA `trainval` only); routing policies are
inference-time label overrides. ECE targets are used for evaluation only;
the margin threshold comes from WA `trainval` only.

WA margin thresholds: `{'v0': 1.9559823212899237, 'backbone': 1.5480887296641908}`.

## Routing policies

All policies reuse the SAME frozen experts (fit on WA `trainval` only);
only the per-row expert assignment changes at inference time.
C0 is the dry specialist, C1 the wet-mountain specialist.

1. `as_routed` (baseline, no fix): the family's own static KMeans router
   decides per row. This is the published-model failure — it sends
   `Lost_Meadow` (100%) and `Renton_Home` (90%) to the wet expert.
2. `c0_only` (force dry): every ECE row goes to the dry expert.
   Blunt but effective; recovers to dynamic-router error levels.
3. `c1_only` (force wet, diagnostic): every row goes to the wet expert.
   Deliberately terrible — proves C1 is the poison and the experts
   themselves are transferable.
4. `gapi_transplant`: labels from the `G_API` rainfall-index router
   (fit on WA only), predictions from the family's frozen experts.
   July–August is bone dry, so all 150 rows land in C0.
5. `dynamic_transplant`: labels from the dynamic 3-feature KMeans router
   (`SMAP_sm_pm_interp_lag1`, `G_API`, `LST_modis`), predictions from the
   family's frozen experts. Also 100% C0 on this window.
6. `seasonal`: calendar router (May–Oct dry, Nov–Apr wet). The ECE window
   falls entirely in the dry season, so all rows land in C0 — the
   cheapest possible rule, using no features at all.
7. `margin_fallback`: keep the static decision when confident, otherwise
   fall back to the `Global_Single_54` expert (also fit on WA only).
   Confidence is the KMeans margin (gap between nearest and
   second-nearest centroid); rows below the WA 5th percentile fall back.
   Best mean RMSE on native-missing input for both families.

Note: transplant cluster labels index the host family's experts, whose
cluster semantics differ by construction — this mismatch is documented,
not hidden. R2 stays negative throughout (variance compression,
`Var(y) ~ 1e-05`); judge policies by RMSE / bias / ubRMSE.

## Pooled summary (mean over seeds)

| family                   | ece_input   | policy             |   rmse_mean |    rmse_std |   mae_mean |   bias_mean |   ubrmse_mean |    r2_mean |   pearson_mean |   rmse_change_vs_as_routed |
|:-------------------------|:------------|:-------------------|------------:|------------:|-----------:|------------:|--------------:|-----------:|---------------:|---------------------------:|
| Clustering_V0_Full_k2    | zero        | as_routed          |   0.117805  | 0.00125947  |  0.0938316 |   0.0707025 |     0.0942262 |  -5.26995  |   -0.486731    |                 0          |
| Clustering_V0_Full_k2    | zero        | c0_only            |   0.0513448 | 0.000214697 |  0.0446738 |   0.0197976 |     0.0473656 |  -0.190965 |    0.141216    |                -0.0664598  |
| Clustering_V0_Full_k2    | zero        | c1_only            |   0.155617  | 0.00198143  |  0.148323  |   0.148323  |     0.0470762 |  -9.9413   |    0.1328      |                 0.0378121  |
| Clustering_V0_Full_k2    | zero        | gapi_transplant    |   0.0513448 | 0.000214697 |  0.0446738 |   0.0197976 |     0.0473656 |  -0.190965 |    0.141216    |                -0.0664598  |
| Clustering_V0_Full_k2    | zero        | dynamic_transplant |   0.0513448 | 0.000214697 |  0.0446738 |   0.0197976 |     0.0473656 |  -0.190965 |    0.141216    |                -0.0664598  |
| Clustering_V0_Full_k2    | zero        | seasonal           |   0.0513448 | 0.000214697 |  0.0446738 |   0.0197976 |     0.0473656 |  -0.190965 |    0.141216    |                -0.0664598  |
| Clustering_V0_Full_k2    | zero        | margin_fallback    |   0.0539197 | 0.000940114 |  0.0459548 |   0.0155183 |     0.0516083 |  -0.313713 |    0.0441687   |                -0.0638849  |
| Clustering_Backbone54_k2 | zero        | as_routed          |   0.145876  | 0.00174498  |  0.135067  |   0.127709  |     0.0704996 |  -8.61435  |   -0.205982    |                 0          |
| Clustering_Backbone54_k2 | zero        | c0_only            |   0.0513448 | 0.000214697 |  0.0446738 |   0.0197976 |     0.0473656 |  -0.190965 |    0.141216    |                -0.0945316  |
| Clustering_Backbone54_k2 | zero        | c1_only            |   0.155617  | 0.00198143  |  0.148323  |   0.148323  |     0.0470762 |  -9.9413   |    0.1328      |                 0.00974034 |
| Clustering_Backbone54_k2 | zero        | gapi_transplant    |   0.0513448 | 0.000214697 |  0.0446738 |   0.0197976 |     0.0473656 |  -0.190965 |    0.141216    |                -0.0945316  |
| Clustering_Backbone54_k2 | zero        | dynamic_transplant |   0.0513448 | 0.000214697 |  0.0446738 |   0.0197976 |     0.0473656 |  -0.190965 |    0.141216    |                -0.0945316  |
| Clustering_Backbone54_k2 | zero        | seasonal           |   0.0513448 | 0.000214697 |  0.0446738 |   0.0197976 |     0.0473656 |  -0.190965 |    0.141216    |                -0.0945316  |
| Clustering_Backbone54_k2 | zero        | margin_fallback    |   0.0561207 | 0.000920032 |  0.0478675 |   0.0192539 |     0.0526871 |  -0.423115 |    0.0384242   |                -0.0897556  |
| Clustering_V0_Full_k2    | native      | as_routed          |   0.0754312 | 0.00150144  |  0.0661495 |   0.0541376 |     0.0525073 |  -1.57123  |    2.71895e-06 |                 0          |
| Clustering_V0_Full_k2    | native      | c0_only            |   0.0691922 | 0.00170403  |  0.0627014 |   0.0506894 |     0.0470804 |  -1.16384  |    0.139703    |                -0.00623907 |
| Clustering_V0_Full_k2    | native      | c1_only            |   0.183449  | 0.00344241  |  0.176782  |   0.176782  |     0.0489945 | -14.2073   |   -0.143219    |                 0.108018   |
| Clustering_V0_Full_k2    | native      | gapi_transplant    |   0.0691922 | 0.00170403  |  0.0627014 |   0.0506894 |     0.0470804 |  -1.16384  |    0.139703    |                -0.00623907 |
| Clustering_V0_Full_k2    | native      | dynamic_transplant |   0.0691922 | 0.00170403  |  0.0627014 |   0.0506894 |     0.0470804 |  -1.16384  |    0.139703    |                -0.00623907 |
| Clustering_V0_Full_k2    | native      | seasonal           |   0.0691922 | 0.00170403  |  0.0627014 |   0.0506894 |     0.0470804 |  -1.16384  |    0.139703    |                -0.00623907 |
| Clustering_V0_Full_k2    | native      | margin_fallback    |   0.0588943 | 0.00108189  |  0.0519288 |   0.0278064 |     0.0518739 |  -0.567342 |    0.038321    |                -0.016537   |
| Clustering_Backbone54_k2 | native      | as_routed          |   0.127926  | 0.00133332  |  0.102571  |   0.0912697 |     0.0896239 |  -6.39364  |   -0.38279     |                 0          |
| Clustering_Backbone54_k2 | native      | c0_only            |   0.0691922 | 0.00170403  |  0.0627014 |   0.0506894 |     0.0470804 |  -1.16384  |    0.139703    |                -0.0587342  |
| Clustering_Backbone54_k2 | native      | c1_only            |   0.183449  | 0.00344241  |  0.176782  |   0.176782  |     0.0489945 | -14.2073   |   -0.143219    |                 0.0555225  |
| Clustering_Backbone54_k2 | native      | gapi_transplant    |   0.0691922 | 0.00170403  |  0.0627014 |   0.0506894 |     0.0470804 |  -1.16384  |    0.139703    |                -0.0587342  |
| Clustering_Backbone54_k2 | native      | dynamic_transplant |   0.0691922 | 0.00170403  |  0.0627014 |   0.0506894 |     0.0470804 |  -1.16384  |    0.139703    |                -0.0587342  |
| Clustering_Backbone54_k2 | native      | seasonal           |   0.0691922 | 0.00170403  |  0.0627014 |   0.0506894 |     0.0470804 |  -1.16384  |    0.139703    |                -0.0587342  |
| Clustering_Backbone54_k2 | native      | margin_fallback    |   0.058715  | 0.000951561 |  0.052015  |   0.0278926 |     0.0516333 |  -0.557722 |    0.0407855   |                -0.0692113  |

## Station RMSE (mean over seeds)

|                                                              |   ECE_BBG_Lost_Meadow |   ECE_BBG_Main_St |   ECE_Renton_Garden_North |   ECE_Renton_Garden_Shed |   ECE_Renton_Home |
|:-------------------------------------------------------------|----------------------:|------------------:|--------------------------:|-------------------------:|------------------:|
| ('Clustering_V0_Full_k2', 'zero', 'as_routed')               |              0.163857 |          0.048967 |                  0.061687 |                 0.028708 |          0.188440 |
| ('Clustering_V0_Full_k2', 'zero', 'c0_only')                 |              0.036338 |          0.035448 |                  0.063735 |                 0.022692 |          0.077614 |
| ('Clustering_V0_Full_k2', 'zero', 'c1_only')                 |              0.163857 |          0.170277 |                  0.070456 |                 0.145785 |          0.197529 |
| ('Clustering_V0_Full_k2', 'zero', 'gapi_transplant')         |              0.036338 |          0.035448 |                  0.063735 |                 0.022692 |          0.077614 |
| ('Clustering_V0_Full_k2', 'zero', 'dynamic_transplant')      |              0.036338 |          0.035448 |                  0.063735 |                 0.022692 |          0.077614 |
| ('Clustering_V0_Full_k2', 'zero', 'seasonal')                |              0.036338 |          0.035448 |                  0.063735 |                 0.022692 |          0.077614 |
| ('Clustering_V0_Full_k2', 'zero', 'margin_fallback')         |              0.052145 |          0.034191 |                  0.073523 |                 0.022093 |          0.068833 |
| ('Clustering_V0_Full_k2', 'native', 'as_routed')             |              0.073270 |          0.076921 |                  0.036734 |                 0.050547 |          0.115125 |
| ('Clustering_V0_Full_k2', 'native', 'c0_only')               |              0.064655 |          0.065244 |                  0.036734 |                 0.050547 |          0.107654 |
| ('Clustering_V0_Full_k2', 'native', 'c1_only')               |              0.190230 |          0.200980 |                  0.097042 |                 0.171799 |          0.229672 |
| ('Clustering_V0_Full_k2', 'native', 'gapi_transplant')       |              0.064655 |          0.065244 |                  0.036734 |                 0.050547 |          0.107654 |
| ('Clustering_V0_Full_k2', 'native', 'dynamic_transplant')    |              0.064655 |          0.065244 |                  0.036734 |                 0.050547 |          0.107654 |
| ('Clustering_V0_Full_k2', 'native', 'seasonal')              |              0.064655 |          0.065244 |                  0.036734 |                 0.050547 |          0.107654 |
| ('Clustering_V0_Full_k2', 'native', 'margin_fallback')       |              0.065552 |          0.045161 |                  0.062140 |                 0.028330 |          0.079532 |
| ('Clustering_Backbone54_k2', 'zero', 'as_routed')            |              0.126676 |          0.170277 |                  0.078108 |                 0.127424 |          0.197529 |
| ('Clustering_Backbone54_k2', 'zero', 'c0_only')              |              0.036338 |          0.035448 |                  0.063735 |                 0.022692 |          0.077614 |
| ('Clustering_Backbone54_k2', 'zero', 'c1_only')              |              0.163857 |          0.170277 |                  0.070456 |                 0.145785 |          0.197529 |
| ('Clustering_Backbone54_k2', 'zero', 'gapi_transplant')      |              0.036338 |          0.035448 |                  0.063735 |                 0.022692 |          0.077614 |
| ('Clustering_Backbone54_k2', 'zero', 'dynamic_transplant')   |              0.036338 |          0.035448 |                  0.063735 |                 0.022692 |          0.077614 |
| ('Clustering_Backbone54_k2', 'zero', 'seasonal')             |              0.036338 |          0.035448 |                  0.063735 |                 0.022692 |          0.077614 |
| ('Clustering_Backbone54_k2', 'zero', 'margin_fallback')      |              0.054977 |          0.037931 |                  0.071339 |                 0.028002 |          0.073482 |
| ('Clustering_Backbone54_k2', 'native', 'as_routed')          |              0.064655 |          0.184914 |                  0.040523 |                 0.060316 |          0.195350 |
| ('Clustering_Backbone54_k2', 'native', 'c0_only')            |              0.064655 |          0.065244 |                  0.036734 |                 0.050547 |          0.107654 |
| ('Clustering_Backbone54_k2', 'native', 'c1_only')            |              0.190230 |          0.200980 |                  0.097042 |                 0.171799 |          0.229672 |
| ('Clustering_Backbone54_k2', 'native', 'gapi_transplant')    |              0.064655 |          0.065244 |                  0.036734 |                 0.050547 |          0.107654 |
| ('Clustering_Backbone54_k2', 'native', 'dynamic_transplant') |              0.064655 |          0.065244 |                  0.036734 |                 0.050547 |          0.107654 |
| ('Clustering_Backbone54_k2', 'native', 'seasonal')           |              0.064655 |          0.065244 |                  0.036734 |                 0.050547 |          0.107654 |
| ('Clustering_Backbone54_k2', 'native', 'margin_fallback')    |              0.064732 |          0.045161 |                  0.062140 |                 0.028330 |          0.079532 |

## Reproduction

From `notebooks/`, run:

```powershell
nb execute experiment/derived_8.4-ece-router-salvage-1.0/derived_8.4-ece-router-salvage-1.0.ipynb --uv --timeout 3600
```

Or run the tracked script directly (same code the notebook imports):

```powershell
uv run --project . python notebooks/experiment/derived_8.4-ece-router-salvage-1.0/run_salvage.py
```

Tables above are transcribed from the executed notebook stdout / CSVs.
Versioned outputs are `summary.csv`, `seed_metrics.csv`,
`station_metrics.csv`, and `routing_audit.json`.
