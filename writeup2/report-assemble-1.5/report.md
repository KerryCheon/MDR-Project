# Regional Models for Daily Soil-Moisture Estimation in Washington

## Technical report assembly 1.5 — Paper 2 handoff

> **Purpose.** This technical report supports drafting the second paper. It records the methods, results, interpretation, evidence boundaries, and source trail in one place. The assembled Markdown is the canonical text; `report.pdf` is its reading copy. It is written so an external reader can follow it without opening any other file in the repository.
>
> **Evidence rule.** This report presents the Washington temporal and leave-one-station-out (LOSO) evaluation together with the ECE station evaluation. The main comparison uses the same protocol and the same random seeds throughout, so every main gap below compares matched runs. The four simpler grouping rules are shown for context.

## 1. Executive synthesis and paper thesis

**Study question.** Does grouping daily observations by their satellite, weather, and site features, then fitting one soil-moisture predictor per group, improve estimates across Washington stations? All multi-regime models in this report share one shape — a router with k=2 sends each station-day observation to exactly one of two XGBoost predictors, with no blending — and the global comparator fits one XGBoost predictor for all observations (§4). The method under test is the **station-majority shared-feature cluster-routed multi-regime model**. It is tested twice: against the same model without the station consistency guarantee (isolating the station-majority rule), and against the existing single-regime global model (isolating grouping itself). The study evaluates which grouping signals are useful on this dataset; it does not propose a new model architecture or a universal number of groups.

**Primary finding.** The station-majority shared-feature cluster-routed multi-regime model has temporal R² **0.8118** (seed SD 0.0014; RMSE 0.0442) over 30 predictor seeds on the 2023–2025 development test period. Against the single-regime global model at R² **0.7798** (seed SD 0.0013), the difference is **+0.0320 R²**, with the station-majority model higher on all 30 seeds (30/0/0 wins/ties/losses). Against the same model without the station consistency guarantee at R² **0.8117**, the difference is **+0.0001 R²**, a near-tie. In leave-one-station-out evaluation, the station-majority model averages **0.6379 R²** versus **0.5797** for the global model, a difference of **+0.0581** across five seeds and seven held-out stations, with gains on 4 folds and losses on 3; versus **0.6199** for the model without the guarantee, the difference is **+0.0180**, with gains on 3 folds and ties on 4.

**Comparison context.** The four simpler grouping rules in Table 1 score lower in this comparison, shown for context. A grouping whose gate is trained on the in-situ target and then frozen has temporal R² **0.7354**; prediction uses features only, but because the gate is supervised on the observed target it is shown only as context. These results describe the specific methods and data evaluated here.

**ECE result.** On five ECE collaboration stations — new in-situ soil-moisture sensor sites deployed by the collaborating ECE team, disjoint from the seven Washington training stations (§8) — the station-majority shared-feature cluster-routed multi-regime model's standard K-means assignment has pooled RMSE **0.1674** because the SMAP satellite inputs it expects are entirely missing there. With an automatic availability-gated fallback based on the available antecedent-precipitation index, pooled RMSE is **0.0578**, close to the global model's **0.0586**. Section 8 uses these sites to interpret what the regional labels mean: the groups separate mainly on wetness, not elevation, and the error at the new lowland sites follows the expert a row is routed to rather than the site location.

**Most important constraint for the paper.** The shared 54-feature set and XGBoost settings were influenced by the same test period used here, as were later group-assignment and model choices. Every 2023–2025 result is development-split evidence rather than untouched confirmation. A future independent time period or external dataset is needed to estimate the true advantage without this selection bias.

**Suggested working title.** “Regional Models for Soil-Moisture Estimation in Washington.” In technical sections, the method can be described as two XGBoost predictors selected by feature-based groups. The station-majority rule is one defined part of that approach.

## 2. Relationship to the first paper and literature

The first paper, [*Enhancing Spatial and Temporal Coverage of Soil Moisture Estimation Using Satellite and Weather-Driven Machine Learning*](../../paper/Enhancing%20Spatial%20and%20Temporal%20Coverage%20of%20Soil%20Moisture%20Estimation%20Using%20Satellite%20and%20Weather-%20Driven%20Machine%20Learning.pdf), established the multi-source satellite/weather pipeline, imputation, feature-engineering approach, and a global XGBoost model. Its reported R² of 0.822 is from five public Washington stations under a different dataset and evaluation protocol; it should not be ranked against the seven-station results here. The paper does not report the project's later regional-group comparisons; those remain project history rather than findings of the first paper. The first paper's DOI is [10.1109/AIIoT68874.2026.11569136](https://doi.org/10.1109/AIIoT68874.2026.11569136).

**Contribution relative to the first paper.** The first paper established the multi-source pipeline, the imputation and feature-engineering approach, and a single global XGBoost estimate at five public stations; it treated regime-based modeling as future work and reported no regional grouping. This work answers that open question under one fixed protocol. It holds the feature backbone and the XGBoost experts constant, compares grouping and assignment rules on matched seeds, adds the station-majority consistency rule and an input-availability gate, and evaluates the result on held-out Washington stations and at the ECE deployment sites. Because regionally specialized and mixture-of-experts soil-moisture models already exist, the contribution is not a new architecture; it is a controlled, paired characterization of which grouping covariates are available when a prediction is made, how assignment is evaluated, and when specialization helps or fails.

Soil-moisture MoE and regional specialization already exist. [Yang et al. (2025)](https://doi.org/10.1016/j.jhydrol.2025.133763) combine CNN–LSTM experts with adaptive weighting across Tibetan Plateau climate zones. [Zhang et al. (2026)](https://doi.org/10.3390/rs18183125) combine an antecedent-precipitation water-balance prior, a time-aware Mamba encoder, and a context-conditioned MoE for forecasting. [Singh et al. (2026)](https://doi.org/10.1029/2025JH001039) use wetness-specialized agents for surface-to-subsurface estimation; their target and protocol differ from this daily surface-estimation benchmark. [Wu et al. (2026)](https://doi.org/10.1016/j.jag.2026.105597) apply an RTE-guided MoE to passive-microwave retrieval. These are direct overlap in the use of specialized components, so architectural novelty is not the claim here.

Clustered soil-moisture models also precede this work: [Chakrabarti et al. (2016)](https://doi.org/10.1109/TGRS.2016.2547389) use soft cluster memberships and kernel regression for disaggregation; [Ding et al. (2024)](https://doi.org/10.1016/j.jag.2024.104003) use K-means-defined subregions with specialized gap-filling models; and [Stahl and McColl (2022)](https://doi.org/10.1175/JCLI-D-21-0780.1) identify global seasonal-cycle regimes. In broader hydrology, [Moges et al. (2016)](https://doi.org/10.1002/2015WR018266) use indicator-driven hierarchical model weighting, while [Rong et al. (2025)](https://doi.org/10.1016/j.jhydrol.2025.132737) compare ways of assigning observations to specialists for streamflow prediction. These studies motivate asking **which inputs determine group assignment, whether they are available when a prediction is made, and how that assignment is evaluated**, rather than claiming that clustering or specialized predictors are new. Full verified reference metadata is in §11.

## 3. Data, target, and evaluation protocol

The study combines in-situ soil-moisture observations with satellite, weather, and site information. It uses 9803 training rows (2017–2020), 4805 validation rows (2021–2022), and 6620 later-period test rows (2023–2025). Training plus validation is 14608 rows fitted together. The prediction target is daily soil moisture at 5 cm.

**Seven Washington study stations.** The network column identifies the source of the in-situ observations. Coordinates, elevation, and training-plus-validation row counts provide site context. No station is hand-assigned to a group anywhere in this report; grouping always comes from the fitted routers described in §4.

| Station | Data network | Lat. | Lon. | Elevation m | Trainval rows |
| --- | --- | --- | --- | --- | --- |
| Beaver Pass | SNOTEL | 48.88 | -121.26 | 1125 | 2185 |
| Cayuse Pass | SNOTEL | 46.87 | -121.53 | 1588 | 2042 |
| Darrington | NOAA USCRN | 48.54 | -121.45 | 166 | 2048 |
| Paradise | SNOTEL | 46.78 | -121.75 | 1564 | 2189 |
| Quinault | NOAA USCRN | 47.51 | -123.81 | 88 | 2160 |
| Sourdough Gulch | SNOTEL | 46.23 | -117.40 | 1164 | 2191 |
| Spokane | NOAA USCRN | 47.42 | -117.53 | 705 | 1793 |

Every model compared in this report predicts the same target — daily volumetric soil moisture at 5 cm depth (m³/m³) — from the same **54** satellite, weather, and site features (full list in Appendix A) and with the same XGBoost predictor settings (§4); the multi-regime models differ from the global model only in that daily observations are grouped before predictors are fitted, and from each other only in the assignment rule. The station-majority shared-feature cluster-routed multi-regime model uses those same features to form its groups; it does not require a separate hand-picked grouping feature set. Inputs include SMAP soil-moisture proxies, Sentinel-derived indices, weather and antecedent precipitation, terrain, climate, land cover, and temporal summaries. The grouping does not use in-situ target labels, though it does use satellite soil-moisture information. The complete feature list appears in Appendix A. The shared feature set makes the model comparison easier to interpret, while its selection using the 2023–2025 test period remains a limitation.

Temporal evaluation trains on training plus validation and evaluates the later test years at the seven known stations. In leave-one-station-out (LOSO) evaluation, the model is refit using six stations and evaluated on the seventh (seven held-out folds total); information from the held-out station is excluded from imputation, scaling, grouping, and thresholds. The global comparator used the same protocol and the same random seeds, so temporal gaps compare matched seeds and LOSO gaps compare matched seed-and-station combinations. Temporal results average 30 XGBoost seeds; LOSO results use five seeds per held-out station. The K-means router itself is fixed at seed 42 throughout.

For all analyses, R², RMSE, MAE, and bias refer to daily volumetric soil moisture at 5 cm; error metrics use m³/m³. R² measures explained variance, RMSE and MAE measure error size, bias is signed mean prediction error, and ubRMSE removes the mean bias component. Seed intervals measure variation across XGBoost predictor random seeds only (the K-means router stays at seed 42). They do not measure uncertainty from station sampling, feature selection, model selection, or reuse of the test period during development. Seed-level 95% t intervals and matched-seed differences answer different questions and should not be interchanged. Seven-station LOSO win counts are descriptive and have little inferential power.

## 4. How the multi-regime models work

The regional-model family studied here treats regional specialization as a routing problem: partition station-day observations into a small number of covariate-defined groups and fit one predictor per group, so that each observation is estimated by a predictor trained on its own group. A router maps each row's feature vector to a group index, and the estimate is produced by that group's predictor; assignment is hard, so exactly one predictor handles each row and there is no blending. The family under test is the **shared-feature cluster-routed multi-regime model**, and the comparator is the **Existing single-regime global model**, which fits one predictor on all observations and performs no grouping. The router and the experts deliberately share one feature backbone — the 54 satellite, weather, and site features described in §3 and listed in Appendix A. That set was assembled for the single-regime global model and reused unchanged here so the information available to every variant is held constant; it is not claimed to be a purpose-built or optimal feature set for routing. Sharing the backbone means the multi-regime and single-regime models differ only in whether observations are partitioned before fitting, not in what they observe. Because the contribution studied here is the grouping step rather than a new architecture, the soft-membership and adaptive-weighting routers that exist in the literature (§2) are outside this study's scope.

**Router.** The router is a K-means model over the shared features, and it is fitted on the fit frame only — never on the evaluation rows. Feature columns are mean-imputed with fit-frame means, standardized with a fit-frame scaler, and clustered with k=2, seed 42, and `n_init=10`. After each fit the two cluster labels are canonicalized by one feature-indexing rule: the group with the lower fit-frame mean of the 30-day rolling SMAP feature `SMAP_sm_pm_interp_rollmean30` is labeled canonical group 1 and the other group 0. This keeps labels comparable across refits without attaching physical wet/dry meaning; group numbers are model identifiers, not climate classes. The group count k=2 is a Washington-specific setting rather than a finding, and the fit-frame grouping-quality comparison behind it is reported in §7 (Table 5); deployments to other regions should rerun that comparison rather than assume two groups.

On the Washington fit data the two groups hold 10624 and 3984 rows. The low-SMAP group contains Spokane and Sourdough Gulch; the other contains Beaver Pass, Cayuse Pass, Darrington, Paradise, and Quinault. On this fit every station's rows fall entirely in one group (minimum station purity 1.000). For reference, the fit-frame means of that SMAP feature are 0.4360 for group 0 and 0.1801 for group 1, with corresponding target means 0.2216 and 0.2094. The groups are well separated on the satellite feature but nearly equal in in-situ mean, so the partition is best read as a covariate grouping rather than a claim about distinct soil-moisture levels. Table 6 (§7) profiles which features separate the groups.

**No target leakage.** Group assignment in this family is computed from the shared feature set and, for known stations, the station identifier alone. The in-situ soil-moisture target is never an input to the router, the label canonicalization, or the availability and margin gates, and no held-out information enters any routing quantity. The satellite SMAP features that drive the grouping are gap-filled remote-sensing estimates, not the observed 5 cm target. The experts are supervised predictors and are fitted on the target, as any regressor is; this statement concerns the routing decision, not expert fitting. For known stations the station identifier is used only as a consistency key, never as a predictive feature. The one routing variant in this report that uses in-situ labels is the target-threshold grouping in the table below, which trains a gate on the observed target and then freezes it, so at prediction time that gate uses only features, like the other routers. It uses in-situ labels during training, so it sits outside the shared-feature family and is reported only as context.

**Station consistency.** A per-observation K-means label can split one station's rows across groups, so each expert would train on a shifting fraction of that station. The station-majority rule removes that ambiguity: each station present in the fit frame is assigned the group held by the majority of its fit-frame rows, with ties broken toward the smaller canonical group id, and that label is applied to all of that station's rows at both fit and predict time. The rule keeps a known station's observations **within a single regime** and changes which training rows each expert sees; it does not assign a group to a station that was not in the fit frame. The model with this rule is the **station-majority shared-feature cluster-routed multi-regime model**; the otherwise identical model without it — every observation assigned by its own nearest centroid — is the shared-feature cluster-routed multi-regime model without station consistency guarantee. The rule is the only difference between the two, giving a controlled comparison of station consistency. For a station not present in the fit frame, a row without a station id, or a row that fails the availability check, assignment falls back to the row-level nearest-centroid label. The availability check is input-only and fires when the row's SMAP block is entirely missing or when its overall missing-feature rate exceeds a fixed 0.10 threshold; the router also marks rows whose centroid margin falls below a fit-frame fifth-percentile threshold as ambiguous for an optional global-predictor fallback.

**Fit and predict.** Given a fit frame (training plus validation, or the fold's six stations in leave-one-station-out evaluation): (1) fit the imputation means, the feature scaler, and K-means on that frame, and canonicalize the two labels; (2) compute each known station's majority group; (3) label every fit row — station majority where the station is known, nearest centroid otherwise — and fit one XGBoost expert per group on that group's rows. To predict a row, apply the frozen imputation and scaling, resolve its group with the same rule (including the availability and margin checks), and return the estimate from that group's expert. In leave-one-station-out evaluation the router, imputation, and scaling are refitted on the six training stations alone, so the held-out station is unseen and its rows are assigned by the row-level rule. The shared XGBoost experts use 2,500 histogram trees, learning rate 0.005, depth 9, minimum child weight 8, gamma 0, alpha 0.03, lambda 0.75, row subsampling 0.9, and column subsampling 0.8; these settings were tuned during the same test era and contribute to the development-split limitation. Only the XGBoost `random_state` varies across seeds; the router is fixed at seed 42.

**Scope and conventions.** The family uses hard assignment, so each row is handled by exactly one expert; an optional soft blend is developed for the station-shortfall setting in §8 and is not evaluated on the Washington benchmark. The shared feature set was selected for the global model and reused here, and its selection used the 2023–2025 period, so all results are development evidence rather than independent confirmation. The global model's single-regime form must not be ranked against the first paper's R² of 0.822, which used five stations and a different protocol (§2).

| Model | Grouping inputs | Assignment rule | Target used? |
| --- | --- | --- | --- |
| station-majority shared-feature cluster-routed multi-regime model | The same 54 shared features as the global model, including current, lagged, and rolling SMAP proxies (reused unchanged; not claimed to be optimal for routing) | 54-feature K-means (mean-impute, standardize, seed 42, `n_init=10`) **plus station-majority rule**: a fitted station gets its most common training-data group; new/missing/gated rows use the per-day label | No |
| shared-feature cluster-routed multi-regime model without station consistency guarantee | Same 54 features | Same 54-feature K-means, per-day assignment for every row (no station-majority rule) | No |
| Existing single-regime global model | No grouping (SMAP only inside the predictor, not a grouping step) | One XGBoost on all 54 features | No |
| Three-feature K-means grouping | 3 time-varying features: `SMAP_sm_pm_interp_lag1`, `G_API`, `LST_modis` | K-means (k=2, standardized, mean-imputed, seed 42, `n_init=10`) | No |
| Seasonal grouping | Calendar month | May–Oct → group 0 (dry season), Nov–Apr → group 1 (wet season); no fitting | No |
| Precipitation-index grouping | Single feature `G_API` (antecedent precipitation index) | Split at the training-data median, low → group 0, high → group 1 | No |
| Target-threshold grouping | In-situ labels to train the gate; the gate is frozen before prediction | XGB classifier predicts whether that day's in-situ soil moisture is below 0.16 m³/m³; at predict time it uses only features | Yes — training only |

The first three rows use the same protocol and the same random seeds, so the comparisons in §§5–6 are matched; the four simpler grouping rules are shown as context. Each of the first three models shares the same target, feature set, and XGBoost configuration and differs only in its grouping or assignment rule.

## 5. Primary temporal results

**Table 1. Temporal results on the 2023–2025 test period.** Seed SD and 95% CI show variation across the 30 XGBoost random seeds (router fixed). The first three rows use the same protocol; the four simpler-grouping rows are shown for context. RMSE, MAE, and bias are m³/m³.

| Model | Seeds | R² | Seed SD | 95% seed CI | RMSE | MAE | Bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| station-majority shared-feature cluster-routed multi-regime model | 30 | 0.8118 | 0.0014 | [0.8113, 0.8124] | 0.0442 | 0.0340 | 0.0060 |
| shared-feature cluster-routed multi-regime model without station consistency guarantee | 30 | 0.8117 | 0.0014 | [0.8112, 0.8122] | 0.0442 | 0.0340 | 0.0060 |
| Three-feature K-means grouping | 30 | 0.7855 | 0.0010 | [0.7851, 0.7858] | 0.0472 | 0.0363 | 0.0096 |
| Existing single-regime global model | 30 | 0.7798 | 0.0013 | [0.7793, 0.7803] | 0.0478 | 0.0369 | 0.0100 |
| Seasonal grouping | 30 | 0.7700 | 0.0016 | [0.7694, 0.7706] | 0.0489 | 0.0377 | 0.0107 |
| Precipitation-index grouping | 30 | 0.7676 | 0.0009 | [0.7672, 0.7679] | 0.0491 | 0.0381 | 0.0106 |
| Target-threshold grouping | 30 | 0.7354 | 0.0011 | [0.7350, 0.7358] | 0.0524 | 0.0388 | 0.0145 |

**Analysis.** The station-majority shared-feature cluster-routed multi-regime model beats the single-regime global model by +0.0320 mean R², higher on all 30 seeds (30/0/0 wins/ties/losses) — a gap roughly twenty-five times the seed SD shown above, so it is not seed noise under this development protocol. The same model without the station consistency guarantee is nearly tied with the station-majority model temporally: their mean R² difference is +0.0001, smaller than the seed SD shown above, though its direction is consistent across the 30 seeds. On the full training-plus-validation fit this is expected: every station sits entirely in one group, so the station-majority rule has almost nothing to reassign (§4). The simpler-grouping rows score lower in this comparison, shown for context.

![Temporal multi-seed comparison and LOSO fold differences](figures/main_temporal_and_loso.png)

*Figure 1.* Model comparison: the first three rows use the same protocol and seeds; simpler-grouping rows are shown for context. Error bars show predictor-seed variation, not selection uncertainty.

## 6. In-state spatial generalization: LOSO

**Table 2. Station-mean leave-one-station-out results.** Each of the first three models uses five XGBoost seeds across seven held-out Washington stations. The first three rows use the same protocol; the four simpler-grouping rows are shown for context.

| Model | Seeds × held-out sites | Station-mean R² | Station-mean RMSE |
| --- | --- | --- | --- |
| station-majority shared-feature cluster-routed multi-regime model | 5 × 7 | 0.6379 | 0.0558 |
| shared-feature cluster-routed multi-regime model without station consistency guarantee | 5 × 7 | 0.6199 | 0.0571 |
| Existing single-regime global model | 5 × 7 | 0.5797 | 0.0610 |
| Three-feature K-means grouping | 5 × 7 | 0.5627 | 0.0624 |
| Seasonal grouping | 5 × 7 | 0.5744 | 0.0619 |
| Precipitation-index grouping | 5 × 7 | 0.5490 | 0.0634 |
| Target-threshold grouping | 5 × 7 | 0.4906 | 0.0667 |

**Analysis.** The station-majority shared-feature cluster-routed multi-regime model's mean advantage over the global model is +0.0581 R², with gains on 4 held-out stations and losses on 3 (24/0/11 wins/ties/losses across the 35 seed-fold combinations). Its mean advantage over the same model without the station consistency guarantee is +0.0180 R², with improvements on 3 held-out stations and ties on 4. The simpler-grouping rows score lower on station-mean R² in this comparison, shown for context. With seven stations, fold counts describe this set of locations and are not population estimates.

**Table 3. LOSO results by held-out station, station-majority versus global.** Each row compares the two models on the same five seeds at one held-out station; wins/ties/losses count seeds where the station-majority model scored higher, equal, or lower.

| Held-out station | Station-majority shared-feature cluster-routed multi-regime model R² | Existing single-regime global model R² | Difference (5-seed mean) | Wins / ties / losses (5 seeds) |
| --- | --- | --- | --- | --- |
| Cayuse Pass | 0.6862 | 0.3309 | +0.3553 | 5/0/0 |
| Paradise | 0.7800 | 0.7384 | +0.0415 | 5/0/0 |
| Beaver Pass | 0.7531 | 0.7135 | +0.0396 | 5/0/0 |
| Quinault | 0.5729 | 0.5619 | +0.0110 | 5/0/0 |
| Darrington | 0.6949 | 0.6970 | -0.0021 | 3/0/2 |
| Spokane | 0.6083 | 0.6271 | -0.0189 | 0/0/5 |
| Sourdough Gulch | 0.3699 | 0.3894 | -0.0195 | 1/0/4 |

**Analysis.** The largest gain is the Cayuse Pass holdout (+0.3553, 5/0/0), where the global predictor transfers poorly and the regional predictor holds; Beaver Pass, Paradise, and Quinault also favor the station-majority model on all five seeds. Three holdouts favor the global model: Spokane (0/0/5) and Sourdough Gulch (1/0/4), the two eastern-Washington stations whose group loses its training representatives when either is held out, and Darrington narrowly (3/0/2). The station-majority rule changes which training observations each group-specific predictor sees; it does not assign a fixed group to a station that was left out. These comparisons describe model behavior and do not establish a geophysical cause.

**Table 4. LOSO results by held-out station, station-majority versus without-guarantee.** The agreement column is an adjusted Rand index (ARI) score between alternative training-set partitions: higher means the two procedures divided the six training stations more similarly. Lower agreement means the group assignments changed more after refitting.

| Held-out station | Station-majority shared-feature cluster-routed multi-regime model R² | shared-feature cluster-routed multi-regime model without station consistency guarantee R² | Difference | Wins / ties (5 seeds) | Train-group agreement: with vs without station consistency guarantee |
| --- | --- | --- | --- | --- | --- |
| Sourdough Gulch | 0.3699 | 0.3108 | +0.0591 | 5/0 | 0.349 |
| Spokane | 0.6083 | 0.5568 | +0.0514 | 5/0 | 0.330 |
| Paradise | 0.7800 | 0.7645 | +0.0154 | 5/0 | 0.990 |
| Beaver Pass | 0.7531 | 0.7531 | +0.0000 | 0/5 | 1.000 |
| Cayuse Pass | 0.6862 | 0.6862 | +0.0000 | 0/5 | 1.000 |
| Darrington | 0.6949 | 0.6949 | +0.0000 | 0/5 | 1.000 |
| Quinault | 0.5729 | 0.5729 | +0.0000 | 0/5 | 1.000 |

**Analysis.** The largest gains occur for the Sourdough Gulch and Spokane holdouts, where refitting changes how the six training stations are divided. For a held-out station, predictions still use its row-level features. Paradise has a smaller gain. The 4 tied folds show locations where the rule did not change the observed result. These comparisons describe model behavior and do not establish a geophysical cause.

## 7. Partition diagnostics and feature interpretation

**Table 5. Number-of-groups comparison using training and validation data.** Lower Davies–Bouldin and higher Calinski–Harabasz favor two groups here; silhouette is slightly higher at three. The Calinski–Harabasz index compares between-group spread to within-group spread (higher means more compact, better-separated groups); the Davies–Bouldin index averages each group's similarity to its most similar group (lower means better separated). These summary measures do not establish physical meaning or the best choice for future data. Deployments to other regions should re-run this comparison on local data rather than assume two groups.

| Number of groups | Silhouette | Calinski–Harabasz | Davies–Bouldin | Trainval station agreement | Smallest group share |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.219 | 3880.9 | 1.579 | 1.000 | 0.273 |
| 3 | 0.226 | 3589.4 | 1.815 | 0.833 | 0.264 |
| 4 | 0.188 | 2855.0 | 2.015 | 0.695 | 0.147 |

For the seven Washington stations (membership stated in §4), this is a sample-specific pattern across the Cascade side and eastern Washington. The five-station group includes both lower-elevation sites and mountain sites, so the groups are not a strict mountain-versus-lowland split. The station-majority shared-feature cluster-routed multi-regime model keeps known stations in one group by design; this consistency is a model rule rather than independent evidence of a physical classification.

**Table 6. Features that differ most between the two groups.** Separation index is a 0–1 group-difference score (1 = groups do not overlap on that feature); the target itself is excluded from the ranking. Profile attributes need not be router inputs — this table profiles what the groups look like, not what the K-means saw.

| Feature | Separation index | Median c0 | Median c1 |
| --- | --- | --- | --- |
| longitude | 1.000 | -121.534 | -117.400 |
| J_bio_bio13 | 1.000 | 391.000 | 70.000 |
| SMAP_sm_pm_interp | 0.932 | 0.414 | 0.192 |
| latitude | 0.642 | 47.510 | 46.233 |
| G_API | 0.551 | 47.459 | 14.936 |
| J_lc_code | 0.550 | 10.000 | 30.000 |
| G_rain_sum_7d | 0.424 | 30.100 | 9.500 |
| J_soil_texture_usda_b0 | 0.398 | 7.000 | 7.000 |
| LST_modis | 0.384 | 280.944 | 291.242 |
| F_NDVI | 0.326 | 0.532 | 0.330 |

**Analysis.** The groups also differ in site descriptors (longitude, precipitation climatology, land cover), satellite soil-moisture summaries, vegetation and temperature features, and antecedent weather. These associations help explain why the regional predictors see different inputs. In this study, the useful interpretation is a broad western/Cascade-side versus eastern-Washington pattern that also reflects elevation and local conditions. It describes these stations and does not define universal physical regimes.

**Why regional specialization helps, and when more data may not.** Each specialist is fitted on a fraction of the rows the global model sees — 10624 rows in one group and 3984 in the other, against 14608 for the global model — yet the two specialists reach R² 0.7989 and 0.8419 on their own test rows, above the global model's pooled 0.7798. The two groups are not equally difficult, so this comparison is descriptive rather than a controlled test of training-set size; it shows that a specialist can do more with less data when the data it sees is internally consistent.

The same pattern appears for inputs rather than rows. Adding ten features to the larger group lowered pooled R² from 0.8143 to 0.7891, while adding ten features to the smaller group left it essentially unchanged at 0.8150. More rows or more inputs are therefore not uniformly beneficial for a fixed feature set; the benefit depends on whether the added information is informative for that regime. We report this as an interpretation and not as a general law: no controlled experiment that varies only the amount of training data was run, and testing that directly is future work.

Grouping is useful here because the two groups differ systematically — in aridity, precipitation, land cover, and the satellite and weather features that predict moisture (Table 6) — so a single global predictor has to fit two different input-to-moisture relationships at once. The global model also carries a pooled bias of 0.0100 m³/m³, larger than the 0.0089 of the larger-group specialist and the -0.0018 of the smaller-group specialist. These differences are consistent with the specialists fitting more homogeneous behavior. They are presented as interpretation rather than as a causal explanation, and they describe the methods and data evaluated here.

## 8. Regional model interpretation at ECE stations

The station-majority shared-feature cluster-routed multi-regime model (§4) fits one soil-moisture predictor per Washington group. Because the Washington split aligns with longitude, the groups were first read as a western/Cascade-side versus eastern pattern. The ECE sites give an independent way to test what those labels mean at new locations: the model assigns every new observation to a group automatically and then applies that group's predictor, so the assignment decides which of the two expert behaviors is used.

The five ECE collaboration stations are new in-situ sensor sites deployed by the collaborating ECE team, disjoint from and outside the seven-station Washington training set. The ECE set contains 150 daily observations from 5 stations between 2026-07-20 and 2026-08-19, with 30 observations per site. Unlike the Washington stations, all five are low, wet Puget-Sound lowland gardens and homes: 51–157 m elevation with 1018–1227 mm annual-precipitation descriptors (Table 10). It is a short late-summer window, so the results describe how the learned assignment and predictors behave at these sites, not full-season performance.

The ECE observations contain none of the SMAP features used in the standard Washington grouping (100% SMAP missing rate on the 150 rows). For these rows the model's input-availability gate (§4) switches automatically to a fallback assignment based on the available antecedent-precipitation index (`G_API`): split at the Washington-fit median, low → the Washington-drier predictor, high → the other predictor. The class-to-predictor direction was fixed once on SMAP-masked Washington validation data before scoring any ECE target, so routing still requires no manual choice per site; ECE targets were used only to score predictions. We report the standard assignment, the automatic fallback, and two forced single-predictor references; the forced rows (run one Washington predictor on every ECE row) are non-deployable diagnostics that show what each predictor does on its own.

For the station-majority shared-feature cluster-routed multi-regime model, Washington fit data associates the western/Cascade-side stations with one predictor and Spokane plus Sourdough Gulch with the other. In the ECE tables, predictor indices 0 and 1 refer to the two fitted predictors within each model; those indices do not assign universal climate labels to ECE stations.

**Table 7. Mapping the precipitation-based fallback to each model's predictors.** Predictor indices are specific to each model fit; they are not climate labels. The fixed single-predictor rows are non-deployable references.

| Model | Precipitation class 0 → predictor | Class 1 → predictor | Fixed comparator predictor | Other predictor | Comparator usable from inputs? | Index with lower WA SMAP mean? |
| --- | --- | --- | --- | --- | --- | --- |
| shared-feature cluster-routed multi-regime model without station consistency guarantee | index 0 | index 1 | index 0 | index 1 | no | no |
| station-majority shared-feature cluster-routed multi-regime model | index 0 | index 1 | index 1 | index 0 | no | yes |

In the Washington fit data, the mean of `SMAP_sm_pm_interp_rollmean30` is 0.4360 in one group and 0.1801 in the other; corresponding target means are 0.2216 and 0.2094. These summaries describe the fitted groups; group numbers are model labels, not physical class labels.

**Table 8. Fallback mapping chosen using Washington validation.** The lower-RMSE mapping was fixed before scoring ECE observations; an exact tie would select class 0 → predictor index 0.

| Precipitation class 0 → | Precipitation class 1 → | WA validation RMSE | Chosen for ECE |
| --- | --- | --- | --- |
| index 0 | index 1 | 0.0740 | yes |
| index 1 | index 0 | 0.1172 | no |

**Table 9. Pooled ECE results over five predictor seeds (station-majority shared-feature cluster-routed multi-regime model + references).** Error metrics are m³/m³. R² is omitted because target variance is very small over this short window. The deployable rows of the same model without the station consistency guarantee are bit-identical to the station-majority shared-feature cluster-routed multi-regime model's on this ECE set (standard RMSE 0.1674, fallback RMSE 0.0578), so they are stated here rather than repeated as extra rows; the soft-blend fallback matches the hard fallback at reported precision and is omitted. "Usable from observed inputs?" marks deployable assignments versus single-predictor diagnostic references.

| Model | Assignment or reference | Usable from observed inputs? | RMSE ± seed SD | MAE | Bias | ubRMSE |
| --- | --- | --- | --- | --- | --- | --- |
| station-majority shared-feature cluster-routed multi-regime model | Usual fitted assignment | yes | 0.1674 ± 0.0034 | 0.1473 | +0.1419 | 0.0888 |
| station-majority shared-feature cluster-routed multi-regime model | Precipitation-based assignment | yes | 0.0578 ± 0.0006 | 0.0502 | +0.0287 | 0.0502 |
| station-majority shared-feature cluster-routed multi-regime model | Single regime predictor 1 (reference) | no | 0.1925 ± 0.0040 | 0.1857 | +0.1857 | 0.0510 |
| station-majority shared-feature cluster-routed multi-regime model | Single regime predictor 0 (reference) | no | 0.0578 ± 0.0006 | 0.0502 | +0.0287 | 0.0502 |
| Existing single-regime global model | Single global predictor | yes | 0.0586 ± 0.0007 | 0.0506 | +0.0151 | 0.0566 |

**Analysis.** The precipitation-based fallback reduces pooled RMSE relative to the standard K-means assignment by 0.1097 m³/m³. Its RMSE is 0.0578, close to the global model's 0.0586 (difference -0.0009). All 150 rows fall in precipitation class 0. Under the mapping chosen with Washington validation, that class points to the Washington-wetter predictor. Its result matches that predictor run alone (RMSE 0.0578); the other predictor run alone has RMSE 0.1925. In other words, on this short late-summer ECE window the fallback test exercises only one of the two regime predictors — it checks whether that predictor transfers, not whether the two-predictor system beats the global model.

**Table 10. ECE station-level errors and site context.** Elevation and annual precipitation describe the sites; they do not explain model error by themselves. The final column summarizes the fallback assignment's weight on the fixed comparison predictor (0.000 at every site: every ECE row took the same branch).

| ECE station | Elev. m | Annual precip. descriptor mm | Usual-assignment RMSE | Precipitation-assignment RMSE | Single-regime global RMSE | Precipitation-assignment bias | Weight on comparator predictor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bellevue Botanical Garden Lost Meadow | 52 | 1019 | 0.0480 | 0.0480 | 0.0619 | +0.0415 | 0.000 |
| Bellevue Botanical Garden Main Street | 51 | 1018 | 0.2105 | 0.0490 | 0.0414 | +0.0464 | 0.000 |
| Renton Garden North | 157 | 1227 | 0.1009 | 0.0569 | 0.0806 | -0.0537 | 0.000 |
| Renton Garden Shed | 157 | 1227 | 0.1566 | 0.0342 | 0.0232 | +0.0252 | 0.000 |
| Renton Home | 136 | 1181 | 0.2425 | 0.0870 | 0.0678 | +0.0839 | 0.000 |

**Analysis.** Errors vary across the five sites, so the pooled result does not describe every location. This short evaluation helps interpret the regional predictors and their assignments at new stations. Its Washington-selected mapping and late-summer observations do not establish full-season performance or a universal regime definition. ECE targets did not determine the mapping.

**What the ECE sites reveal about the regional labels.** The Washington groups separate strongly on longitude and wetness and only weakly on elevation. In the group profile, longitude and precipitation of the wettest month each reach a separation index of 1.00 and the satellite soil-moisture summary 0.93, while elevation separates by only 0.208 (group medians 1205 m and 1161 m). Longitude is therefore a proxy for a wet, maritime versus drier, more seasonal contrast, with elevation and land-surface setting a secondary axis. The five ECE sites sit well below either group's median elevation, so they lie outside the elevation range on which the groups were fitted even though they are geographically western.

At these new lowland sites the fitted assignment is not reliable. Under the standard assignment the five nearby sites that share these lowland characteristics split across both experts, and the error follows the expert rather than the site: the one site routed to the wetter/Western predictor has RMSE 0.0480, while the four routed mainly to the drier/Eastern predictor reach up to 0.2425 (Table 10). When the automatic fallback routes every row to the wetter/Western predictor (the class-to-expert direction fixed once on Washington validation), pooled RMSE is 0.0578, close to the global model's 0.0586. The grouping signal that looked geographic in Washington is therefore only partly about location: at new lowland sites the assignment rule makes a large difference in accuracy, and a longitude-based label would not identify the better expert. These per-site comparisons are diagnostics that explain which expert suits these sites; the deployed assignment stays automatic, with no manual regime selection.

This connects to the discussion in §7 that more rows or inputs are not uniformly better. The larger western group pools coastal-lowland and high-elevation sites, so its additional rows do not make the learned assignment reliable at the new lowland sites, even though its expert transfers well once it is selected. We present this as an interpretation of a short, single-branch evaluation. The mapping was fixed using Washington validation, all 150 rows took the same branch, and the group indices remain model-specific conventions rather than established physical classes; the results do not define a universal regime or establish full-season or out-of-region transfer.

![Station-majority shared-feature cluster-routed multi-regime model assignment options and ECE station results](figures/ece_policies_and_stations.png)

*Figure 2.* ECE pooled (A) and station-level (B) RMSE, labeled with the model names stated in the text. The station-level panel shows that the error follows the expert a site is routed to. Forced single-predictor bars are model-specific diagnostic references, not deployable assignments.

A separate hourly analysis found median within-day sensor range 0.0349 versus median absolute day-to-day daily-target step 0.0025 in its focus windows. These observations help describe the daily measurement setting; the ECE results here remain a short, site-specific view of the Washington models.

## 9. Robustness, negative evidence, and limits

The station-majority shared-feature cluster-routed multi-regime model uses the shared feature set described in §4.

**Out-of-state comparison.** Multi-regime and comparator configurations evaluated on ten out-of-state stations all have negative station-mean R²; the two-group variants show no reliable transfer beyond the region. These are not results for the station-majority shared-feature cluster-routed multi-regime model, so they appear here as a scope boundary, not a result table.

| Model or grouping | 10-station mean R² | 10-station mean RMSE | Pooled R² |
| --- | --- | --- | --- |
| shared-feature cluster-routed multi-regime model without station consistency guarantee (54 features) | -0.527 | 0.088 | 0.1325 ± 0.0044 |
| Seasonal grouping | -0.517 | 0.087 | 0.1613 ± 0.0059 |
| Precipitation-index grouping | -0.729 | 0.090 | 0.0899 ± 0.0087 |
| Existing single-regime global model (54 features) | -0.429 | 0.085 | 0.2060 ± 0.0047 |
| Three-feature K-means grouping | -0.561 | 0.089 | 0.1380 ± 0.0063 |
| Target-threshold grouping | -0.765 | 0.095 | 0.0485 ± 0.0060 |

**Interpretation.** The results support evaluating regional predictors for within-Washington differences under this development protocol. The out-of-state table shows that the current models do not establish transfer to other regions. Snowpack-dominated sites and missing snow-water-equivalent information are areas for future model development.

**Selection and inference limits.** Seven held-out stations provide a small spatial sample. Temporal seed intervals measure XGBoost fitting variation; they do not include uncertainty from station sampling or feature and model selection. The shared features, XGBoost settings, and some model choices were informed by the 2023–2025 test period, so the scores are development evidence. The station-majority rule applies to known stations; for new stations, predictions use row-level grouping. The ECE evaluation covers five separate stations over a short late-summer window, with a fallback mapping selected using Washington validation. It helps interpret the model behavior at those sites but does not establish broader seasonal or geographic transfer.

## 10. Paper contribution decisions and writing guidance

1. **Lead with the two main results.** Report the temporal and LOSO differences for the station-majority shared-feature cluster-routed multi-regime model versus the global model together with the comparison against the same model without the station-consistency rule: grouping itself gains +0.0320 temporal R² (30/0 seeds) and +0.0581 LOSO R²; the station-majority rule adds +0.0001 temporal and +0.0180 LOSO over the same model without it. Present the simpler grouping rows as additional context. Report fold-level wins and losses honestly — three LOSO folds favor the global model.
2. **Explain the Washington groups.** Describe the five west/Cascade-side stations and two eastern stations as a sample-specific pattern. State that the separation is dominated by wetness and precipitation seasonality (which longitude proxies), with elevation and land-surface setting a weaker secondary axis.
3. **Use the ECE section to interpret model behavior.** Explain the SMAP shortfall, the Washington-selected precipitation fallback, pooled and site-level errors, and the fact that the error at the new lowland sites follows the expert a row is routed to automatically rather than the site location — including that all 150 ECE rows took the same fallback branch. Keep the small sample and short window in view.
4. **State the development-split status in abstract, results, and limitations.** Independent confirmation requires new dates or sites and a selection protocol that keeps final evaluation untouched. Future work can test the number of groups, an evaluated global fallback for ECE routing, more station types and snowpack features, full-season ECE data, and broader geographic transfer.
5. **Include the specialization-versus-data discussion.** Explain that each specialist trains on a subset of the rows yet matches or beats the all-data global predictor on its own group, and that additional rows or inputs are not uniformly beneficial for a fixed feature set. Report this as interpretation and keep the absence of a controlled data-quantity experiment explicit; do not compare station sets across dataset versions.

**Proposed paper spine.** Introduction: the estimation question after the first paper. Related work: regional soil-moisture models. Methods: data, shared predictors, the multi-regime model, and the global comparison. Results: temporal performance, held-out station performance, and group interpretation. Discussion: behavior at ECE stations, reliance on satellite and weather inputs, selection limits, and scope. Conclusion: a bounded finding about regional models for Washington stations.

## 11. References

1. Balkovec, J., et al. *Enhancing Spatial and Temporal Coverage of Soil Moisture Estimation Using Satellite and Weather-Driven Machine Learning.* In *2026 IEEE World AI IoT Congress (AIIoT)*. [doi:10.1109/AIIoT68874.2026.11569136](https://doi.org/10.1109/AIIoT68874.2026.11569136); [repository PDF](../../paper/Enhancing%20Spatial%20and%20Temporal%20Coverage%20of%20Soil%20Moisture%20Estimation%20Using%20Satellite%20and%20Weather-%20Driven%20Machine%20Learning.pdf). Historical context for the pipeline and global model.
2. Yang, J., Yang, Q., Hu, F., and Shao, J. (2025). “An Interpretable Framework of Soil Moisture Estimation Based on Mixture-of-Experts (ISMoE): A Case Study on the Tibetan Plateau.” *Journal of Hydrology* 661, 133763. [doi:10.1016/j.jhydrol.2025.133763](https://doi.org/10.1016/j.jhydrol.2025.133763).
3. Zhang, Z., Mao, K., Yuan, Z., and Bateni, S. M. (2026). “An Antecedent-Precipitation-Informed Soil Water Balance and Time-Aware Mamba–MoE Framework for Surface Soil Moisture Forecasting.” *Remote Sensing* 18(18), 3125. [doi:10.3390/rs18183125](https://doi.org/10.3390/rs18183125).
4. Singh, A., Singh, V., and Gaurav, K. (2026). “Weak Physics-Guided Multi-Agent Learning for Surface to Subsurface Moisture Estimation Across Diverse Climate and Soil Conditions.” *JGR: Machine Learning and Computation* 3, e2025JH001039. [doi:10.1029/2025JH001039](https://doi.org/10.1029/2025JH001039).
5. Wu, Y., Mao, K., Shi, J., Guo, Z., and Bateni, S. M. (2026). “An RTE-Guided Mixture-of-Experts Framework for AMSR2 Passive Microwave Soil Moisture Retrieval.” *International Journal of Applied Earth Observation and Geoinformation* 154, 105597. [doi:10.1016/j.jag.2026.105597](https://doi.org/10.1016/j.jag.2026.105597).
6. Chakrabarti, S., Judge, J., Bongiovanni, T., Rangarajan, A., and Ranka, S. (2016). “Disaggregation of Remotely Sensed Soil Moisture in Heterogeneous Landscapes Using Holistic Structure-Based Models.” *IEEE TGRS* 54(8), 4629–4641. [doi:10.1109/TGRS.2016.2547389](https://doi.org/10.1109/TGRS.2016.2547389).
7. Ding, T., Zhao, W., and Yang, Y. (2024). “Addressing Spatial Gaps in ESA CCI Soil Moisture Product: A Hierarchical Reconstruction Approach Using Deep Learning Model.” *International Journal of Applied Earth Observation and Geoinformation* 132, 104003. [doi:10.1016/j.jag.2024.104003](https://doi.org/10.1016/j.jag.2024.104003).
8. Stahl, M. O., and McColl, K. A. (2022). “The Seasonal Cycle of Surface Soil Moisture.” *Journal of Climate* 35, 4997–5012. [doi:10.1175/JCLI-D-21-0780.1](https://doi.org/10.1175/JCLI-D-21-0780.1).
9. Moges, E., Demissie, Y., and Li, H.-Y. (2016). “Hierarchical Mixture of Experts and Diagnostic Modeling Approach to Reduce Hydrologic Model Structural Uncertainty.” *Water Resources Research* 52, 2551–2570. [doi:10.1002/2015WR018266](https://doi.org/10.1002/2015WR018266).
10. Rong, Z., Sun, W., Xie, Y., Huang, Z., and Chen, X. (2025). “Mixture of Experts Leveraging Informer and LSTM Variants for Enhanced Daily Streamflow Forecasting.” *Journal of Hydrology*, 132737. [doi:10.1016/j.jhydrol.2025.132737](https://doi.org/10.1016/j.jhydrol.2025.132737).
11. Chen, T., and Guestrin, C. (2016). “XGBoost: A Scalable Tree Boosting System.” *Proceedings of KDD*. [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).

## Appendix A. Exact shared 54-feature backbone

The 54 features are grouped below by theme so the list reads as model inputs rather than a name dump. Names are exact pipeline identifiers (backticked); the one-line gloss says what each feature measures. Name prefixes encode the transform family: `V_` rolling-window statistics, `A_` changes and slopes, `C_` lags and memory, `D_` seasonal or spectral transforms, `E_` radar physics, `G_` hydrologic and weather inputs, `J_` static site descriptors, `F_` optical indices, and bare `SMAP_` satellite soil moisture. A `_kobsK` window covers the last K valid observations, not calendar days, which keeps rolling statistics stable across satellite revisit and cloud gaps.

**Precipitation and hydrologic memory (6 features).** Rain actually observed at the station, plus decaying memory of past rain.

- `precip_mm`: daily ERA5-Land precipitation at the station point (mm)
- `G_API`: antecedent precipitation index: exponentially decayed rain memory
- `G_DSLR`: days since last rain: dry-down duration
- `G_rain_sum_3d`: calendar 3-day cumulative rainfall (true days)
- `G_rain_sum_7d`: calendar 7-day cumulative rainfall (true days)
- `V_rollrng_G_API_kobs7`: rolling range of API over the last 7 observations

**Satellite soil moisture - SMAP (12 features).** Coarse satellite moisture and its recent history, the grouping's main signal.

- `SMAP_sm_pm_interp`: SMAP evening-pass soil moisture, gap-filled to daily
- `SMAP_sm_pm_interp_lag7`: evening SMAP value shifted back 7 days
- `SMAP_sm_pm_interp_lag30`: evening SMAP value shifted back 30 days
- `SMAP_sm_pm_interp_rollrange7`: 7-day rolling range of evening SMAP
- `SMAP_sm_pm_interp_rollmean30`: 30-day rolling mean of evening SMAP
- `SMAP_sm_pm_interp_rollrange30`: 30-day rolling range of evening SMAP
- `SMAP_sm_interp_rollrange7`: 7-day rolling range of combined AM+PM SMAP
- `SMAP_ampm_diff_interp`: morning-minus-evening SMAP: diurnal dry-down signal
- `A_d_SMAP_sm_interp_kobs30`: 30-observation first difference of combined SMAP: wetting/drying step
- `V_rollmin_SMAP_sm_interp_kobs14`: rolling minimum of combined SMAP over 14 observations
- `V_rollmin_SMAP_sm_interp_kobs30`: rolling minimum of combined SMAP over 30 observations
- `SMAP_x_year`: SMAP-by-year interaction: long-term sensor drift term

**Radar - Sentinel-1 SAR (10 features).** Microwave backscatter physics and its recent volatility.

- `E_SAR_ratio`: VV/VH backscatter ratio: vegetation/soil-moisture sensitive
- `A_d_E_SAR_ratio_kobs30`: 30-observation first difference of the SAR ratio
- `V_rollmax_E_SAR_ratio_kobs7`: rolling maximum of the SAR ratio over 7 observations
- `V_rollmin_E_SAR_ratio_kobs30`: rolling minimum of the SAR ratio over 30 observations
- `V_rollmax_E_SAR_ratio_kobs30`: rolling maximum of the SAR ratio over 30 observations
- `A_grad_E_SAR_diff_kobs30`: linear slope of the VV-minus-VH difference over 30 observations
- `V_rollmax_E_SAR_diff_kobs14`: rolling maximum of the SAR difference over 14 observations
- `V_rollrng_E_SAR_diff_kobs30`: rolling range of the SAR difference over 30 observations
- `V_rollmax_E_SAR_diff_kobs30`: rolling maximum of the SAR difference over 30 observations
- `E_rough_s1_vh_kobs14`: surface-roughness proxy: rolling variability of VH backscatter over 14 observations

**Vegetation and optical - Sentinel-2 (12 features).** Greenness, canopy water, and shortwave-infrared moisture bands.

- `s2_b4`: Sentinel-2 Red band B4 (665 nm) surface reflectance
- `s2_b8`: Sentinel-2 near-infrared band B8 (842 nm) surface reflectance
- `A_grad_s2_b11_kobs30`: slope of SWIR1 B11 (1610 nm) over 30 observations: moisture trend
- `V_rollrng_s2_b11_kobs30`: rolling range of B11 over 30 observations
- `V_rollmin_s2_b11_kobs30`: rolling minimum of B11 over 30 observations
- `V_rollmin_s2_b12_kobs30`: rolling minimum of SWIR2 B12 (2190 nm) over 30 observations
- `V_rollmax_F_NDMI_kobs30`: rolling maximum of NDMI (NIR-SWIR canopy-water index) over 30 observations
- `V_rollmax_F_NDVI_kobs14`: rolling maximum of NDVI (red-NIR greenness) over 14 observations
- `V_rollmax_F_NDVI_kobs30`: rolling maximum of NDVI over 30 observations
- `V_ema_F_NDVI_kobs30`: exponential moving average of NDVI over 30 observations
- `C_lag_F_NDVI_kobs30`: NDVI value lagged 30 observations
- `D_z_F_NDMI`: seasonal z-score anomaly of NDMI vs its monthly climatology

**Land-surface temperature (5 features).** MODIS temperature level, anomaly, and periodicity.

- `V_rollmin_LST_modis_kobs30`: rolling minimum of MODIS land-surface temperature over 30 observations: cold extreme
- `V_ema_LST_modis_kobs30`: exponential moving average of land-surface temperature over 30 observations
- `D_z_LST_modis`: seasonal z-score anomaly of land-surface temperature
- `D_fft_dom_LST_modis_kobs30`: dominant Fourier frequency of temperature over 30 observations: periodicity
- `D_fft_ent_LST_modis_kobs30`: spectral entropy of temperature over 30 observations: signal complexity

**Calendar and seasonal cycle (4 features).** Where the day sits in the annual and multi-year cycle.

- `D_sin_DOY`: sine of day-of-year: seasonal cycle phase
- `D_cos_DOY`: cosine of day-of-year: seasonal cycle quadrature
- `sin_year`: sine of fractional year: multi-year cyclic trend
- `cos_year`: cosine of fractional year: multi-year quadrature

**Static site descriptors (5 features).** Terrain, climate normals, land cover, and soil at the station (no time variation).

- `J_aspect_deg`: SRTM terrain aspect in degrees at the station
- `J_bio_bio02`: WorldClim mean diurnal temperature range (1970-2000 normals)
- `J_bio_bio13`: WorldClim precipitation of wettest month (1970-2000 normals)
- `J_lc_code`: land-cover class code (ESA WorldCover/NLCD)
- `J_soil_texture_usda_b0`: FAO HWSD USDA soil-texture class at the surface
