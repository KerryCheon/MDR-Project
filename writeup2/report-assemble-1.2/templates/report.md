# Regional Models for Daily Soil-Moisture Estimation in Washington

## Technical report assembly 1.2 — Paper 2 handoff

> **Purpose.** This technical report supports drafting the second paper. It records the methods, results, interpretation, evidence boundaries, and source trail in one place. The assembled Markdown is the canonical text; `report.pdf` is its reading copy. It is written so an external reader can follow it without opening any other file in the repository.
>
> **Evidence rule.** This report presents the Washington temporal and leave-one-station-out (LOSO) evaluation together with the ECE station evaluation as one final evidence batch. They use different station sets and evaluation protocols. Exact source paths, run identities, and hashes are recorded in the claims ledger and provenance manifest. All reported values are inserted from saved artifacts by `scripts/report_data.py`; no models are fitted during report assembly.

### The three main models (read this first)

All models in this report predict the same thing — **daily volumetric soil moisture at 5 cm depth (m³/m³)** — from the same **shared set of 54 satellite, weather, and site features** (§3, full list in Appendix A), using the same XGBoost predictor settings (§4). They differ only in **how they group daily observations before fitting predictors**:

- **Two-regime regional model (the family under test).** A K-means router with k=2 divides each station-day observation into one of two groups, then a separate XGBoost predictor is fitted for each group. Each new observation is handled by exactly one of the two predictors (no blending). The router is fitted on fit data only (mean-impute → standardize → K-means with seed 42, `n_init=10`); cluster labels are canonicalized after each fit so the drier group — lower fit-data mean of the 30-day rolling SMAP feature `SMAP_sm_pm_interp_rollmean30` — has a consistent label.
- **Primary regional model.** The two-regime model **plus a station-majority rule**: a station already seen during fitting is always assigned to whichever group most of its fit-period days fell in (ties go to the smaller group id). Rows from a new station, rows without a station id, or rows failing the missing-data gate fall back to the per-day K-means assignment. We call this the **primary regional model** throughout and report its numbers as the headline result.
- **Unguarded two-regime model.** The identical two-regime model **without** the station-majority rule: every row is assigned from its own features by nearest K-means centroid. This is the only directly paired ablation of the primary model (same features, same predictors, same seeds).
- **Single-regime global model.** One XGBoost predictor fitted on all observations, no grouping. The global model reported here is re-run on the **same seven stations, same 54 features, and same test period** as the regional models, so it is a contemporary comparator. Do not confuse it with the first paper's R² of 0.822, which used five stations and a different protocol (§2) and must not be ranked against the numbers here.

## 1. Executive synthesis and paper thesis

**Study question.** Does grouping daily observations by their satellite, weather, and site features, then fitting one soil-moisture predictor per group, improve estimates across Washington stations? The **primary regional model** (two K-means groups + one XGBoost predictor per group + station-majority rule) is the method under test. We compare it against the unguarded version of the same two-regime model and against the single-regime global model and simpler grouping rules described in §4. The study evaluates which grouping signals are useful on this dataset; it does not propose a new model architecture or a universal number of groups.

**Primary finding.** The primary regional model has temporal R² **{{GUARD_R2}}** (seed SD {{GUARD_SD}}; RMSE {{GUARD_RMSE}}) over 30 predictor seeds on the 2023–2025 development test period. The unguarded two-regime model has R² **{{BACKBONE_R2}}**, a small difference of **{{GUARD_BACKBONE_DIFF}} R²**. In leave-one-station-out evaluation, the primary model averages **{{GUARD_LOSO}} R²** versus **{{BACKBONE_LOSO}}** for the unguarded model, a difference of **{{GUARD_LOSO_DIFF}}** across five seeds and seven held-out stations. The station table locates the gain in {{FOLD_GAINS}} folds; {{FOLD_TIES}} folds tie.

**Comparison context.** The contemporary global model has temporal R² **{{GLOBAL_R2}}** and LOSO R² **{{GLOBAL_LOSO}}**. The apparent primary-model gaps of **{{GUARD_GLOBAL_GAP}}** temporal and **{{GUARD_GLOBAL_LOSO_GAP}}** LOSO are background context from a different saved run, not paired tests — only the primary-vs-unguarded difference above is controlled. A grouping that learns its split from the in-situ target itself has temporal R² **{{GATE_R2}}**; that target is unavailable when making predictions, so it is shown only to illustrate the ceiling/cheating case. These results describe the specific methods and data evaluated here.

**ECE result.** On five ECE collaboration stations, the primary regional model's standard K-means assignment has pooled RMSE **{{ECE_ROUTED_RMSE}}** because the SMAP satellite inputs it expects are entirely missing there. With a fallback assignment based on the available antecedent-precipitation index, pooled RMSE is **{{ECE_AUTO_RMSE}}**, close to the global model's **{{ECE_GLOBAL_RMSE}}**. The next sections explain how the regional groups relate to the Washington sites and how the ECE comparison should be read.

**Most important constraint for the paper.** The shared 54-feature set and XGBoost settings were influenced by the same test period used here, as were later group-assignment and model choices. Every 2023–2025 result is development-split evidence rather than untouched confirmation. A future independent time period or external dataset is needed to estimate the true advantage without this selection bias.

**Suggested working title.** “Regional Models for Soil-Moisture Estimation in Washington.” In technical sections, the method can be described as two XGBoost predictors selected by feature-based groups. The primary model's station-majority rule is one defined part of that approach.

## 2. Relationship to the first paper and literature

The first paper, [*Enhancing Spatial and Temporal Coverage of Soil Moisture Estimation Using Satellite and Weather-Driven Machine Learning*](../../paper/Enhancing%20Spatial%20and%20Temporal%20Coverage%20of%20Soil%20Moisture%20Estimation%20Using%20Satellite%20and%20Weather-%20Driven%20Machine%20Learning.pdf), established the multi-source satellite/weather pipeline, imputation, feature-engineering approach, and a global XGBoost model. Its reported R² of 0.822 is from five public Washington stations under a different dataset and evaluation protocol; it should not be ranked against the seven-station results here. The paper does not report the project's later regional-group comparisons; those remain project history rather than findings of the first paper. The first paper's DOI is [10.1109/AIIoT68874.2026.11569136](https://doi.org/10.1109/AIIoT68874.2026.11569136).

Soil-moisture MoE and regional specialization already exist. [Yang et al. (2025)](https://doi.org/10.1016/j.jhydrol.2025.133763) combine CNN–LSTM experts with adaptive weighting across Tibetan Plateau climate zones. [Zhang et al. (2026)](https://doi.org/10.3390/rs18183125) combine an antecedent-precipitation water-balance prior, a time-aware Mamba encoder, and a context-conditioned MoE for forecasting. [Singh et al. (2026)](https://doi.org/10.1029/2025JH001039) use wetness-specialized agents for surface-to-subsurface estimation; their target and protocol differ from this daily surface-estimation benchmark. [Wu et al. (2026)](https://doi.org/10.1016/j.jag.2026.105597) apply an RTE-guided MoE to passive-microwave retrieval. These are direct overlap in the use of specialized components, so architectural novelty is not the claim here.

Clustered soil-moisture models also precede this work: [Chakrabarti et al. (2016)](https://doi.org/10.1109/TGRS.2016.2547389) use soft cluster memberships and kernel regression for disaggregation; [Ding et al. (2024)](https://doi.org/10.1016/j.jag.2024.104003) use K-means-defined subregions with specialized gap-filling models; and [Stahl and McColl (2022)](https://doi.org/10.1175/JCLI-D-21-0780.1) identify global seasonal-cycle regimes. In broader hydrology, [Moges et al. (2016)](https://doi.org/10.1002/2015WR018266) use indicator-driven hierarchical model weighting, while [Rong et al. (2025)](https://doi.org/10.1016/j.jhydrol.2025.132737) compare ways of assigning observations to specialists for streamflow prediction. These studies motivate asking **which inputs determine group assignment, whether they are available when a prediction is made, and how that assignment is evaluated**, rather than claiming that clustering or specialized predictors are new. Full verified metadata is in `references.bib` and §12.

## 3. Data, target, and evaluation protocol

The study combines in-situ soil-moisture observations with satellite, weather, and site information. It uses {{TRAIN_N}} training rows (2017–2020), {{VAL_N}} validation rows (2021–2022), and {{TEST_N}} later-period test rows (2023–2025). Training plus validation is {{TRAINVAL_N}} rows fitted together (called "trainval" in file names). The prediction target is daily soil moisture at 5 cm. Source and processing details are recorded in the claims ledger and provenance manifest.

**Seven Washington study stations.** The network column identifies the source of the in-situ observations. Coordinates, elevation, and group membership provide site context; group membership is not a causal climate classification. Group numbers are local to this study.

{{WA_STATION_TABLE}}

The global model and each regional predictor use the same **{{FEATURE_COUNT}}** features. The primary regional model uses those same features to form groups; it does not require a separate hand-picked grouping feature set. Inputs include SMAP soil-moisture proxies, Sentinel-derived indices, weather and antecedent precipitation, terrain, climate, land cover, and temporal summaries. The grouping does not use in-situ target labels, though it does use satellite soil-moisture information. The complete feature list appears in Appendix A. The shared feature set makes the model comparison easier to interpret, while its selection using the 2023–2025 test period remains a limitation.

Temporal evaluation trains on training plus validation and evaluates the later test years at the seven known stations. In leave-one-station-out (LOSO) evaluation, the model is refit using six stations and evaluated on the seventh (seven held-out folds total); information from the held-out station is excluded from imputation, scaling, grouping, and thresholds. Temporal results average 30 XGBoost seeds; LOSO results use five seeds per held-out station. The K-means router itself is fixed at seed 42 throughout, because per-group feature additions are tied to one clustering.

For all analyses, R², RMSE, MAE, and bias refer to daily volumetric soil moisture at 5 cm; error metrics use m³/m³. R² measures explained variance, RMSE and MAE measure error size, bias is signed mean prediction error, and ubRMSE removes the mean bias component. Seed intervals measure variation across XGBoost predictor random seeds only (the K-means router stays at seed 42). They do not measure uncertainty from station sampling, feature selection, model selection, or reuse of the test period during development. Seed-level 95% t intervals and paired seed tests answer different questions and should not be interchanged. A seven-station LOSO win count is descriptive and has little inferential power.

## 4. How the regional models work

The question this paper tests is multi-regime versus single-regime: whether fitting one predictor per group beats fitting one predictor for all observations. The regional family therefore pairs a group router with one XGBoost predictor per group, and each station-day observation is handled by exactly one predictor — there is no blending across predictors. Hard assignment is a deployability choice rather than an accuracy claim: a soft-blend fallback policy run on the ECE set matches the hard fallback at reported precision (Table 8), so the simpler deployable rule is kept; that comparison covers only the short ECE window (§8). This contrasts with the soft-membership and adaptive-weighting designs in the literature (§1), which this study does not test.

The group count k=2 is a Washington setting, not a finding: two groups performed best on these seven stations, and the training-and-validation number-of-groups comparison behind that choice is reported in Table 4 (§7). Deployments to other regions should re-run that comparison on local data rather than assume two groups.

On the Washington fit data, the drier-labeled group contains Spokane and Sourdough Gulch; the other group contains Beaver Pass, Cayuse Pass, Darrington, Paradise, and Quinault. Labels are canonicalized after each fit so the drier group — lower fit-data mean of the 30-day rolling SMAP feature `SMAP_sm_pm_interp_rollmean30` — carries a consistent label, but labels can still change across refits. The Washington fit-frame means of that SMAP feature are {{WA_SMAP_C0}} for canonical group 0 and {{WA_SMAP_C1}} for group 1, with corresponding target means {{WA_TARGET_C0}} and {{WA_TARGET_C1}}; group numbers are model labels, not physical class labels. Table 5 (§7) profiles which features differ most between the groups.

Per-day K-means assignment can split one station's rows across groups, so each group-specific predictor would train on a shifting slice of that station. The station-majority rule keeps each fitted station in a single group — the station's most common fit-frame group, with ties broken toward the smaller group id — and keeping known stations together changes which training examples each predictor sees. That rule is the only difference between the primary model and its unguarded twin, so the paired gaps in §§5–6 ({{GUARD_BACKBONE_DIFF}} temporal R², {{GUARD_LOSO_DIFF}} LOSO R²) measure exactly the rule's effect. For a new station or a row without a station id, assignment falls back to the per-day nearest-centroid label. Rows whose SMAP block is entirely missing, or whose overall missing-data rate exceeds the fitted gate, are flagged by an input-only availability check so the ECE shortfall (§8) can be described honestly.

The shared XGBoost predictors use 2,500 histogram trees, learning rate 0.005, depth 9, minimum child weight 8, gamma 0, alpha 0.03, lambda 0.75, row subsampling 0.9, and column subsampling 0.8. These settings were tuned during the same test era and contribute to the development-split limitation. Only the XGBoost `random_state` varies across seeds; the routers are fixed at seed 42.

| Model | Grouping inputs | Assignment rule |
| --- | --- | --- |
| Target-threshold grouping | Earlier 50-feature set, plus in-situ labels to train the gate (undeployable) | XGB classifier (same XGBoost settings) predicts whether that day's in-situ soil moisture is below 0.16 m³/m³; at predict time it uses only features |
| Seasonal grouping | Calendar month | May–Oct → group 0 (dry season), Nov–Apr → group 1 (wet season); no fitting |
| Precipitation-index grouping | Single feature `G_API` (antecedent precipitation index) | Split at the fit-data median (refit on each fit frame), low → group 0, high → group 1 |
| Dynamic-feature grouping | 3 time-varying features: `SMAP_sm_pm_interp_lag1`, `G_API`, `LST_modis` | K-means (k=2, standardized, mean-imputed, seed 42, `n_init=10`) |
| Primary regional model | All 54 shared features, including current, lagged, and rolling SMAP proxies | 54-feature K-means (same recipe as above) **plus station-majority rule**: a fitted station gets its most common fit-frame group; new/missing/gated rows use the per-day label |
| Unguarded two-regime model | Same 54 features | Same 54-feature K-means, per-day assignment for every row (no station-majority rule) |
| Single-regime global model | No grouping (SMAP only inside the predictor, not a grouping step) | One XGBoost on all 54 features |

The main comparison is between the primary regional model and its unguarded twin. The global model and the simpler grouping rules above provide context. The precipitation-index rule is the only SMAP-free grouping, so it doubles as the ECE fallback in §8. Because the shared test period informed feature and model choices, all results here are development evidence rather than independent confirmation. An earlier 50-feature regional variant (V0) is retained as background in Appendix B only.

## 5. Primary temporal results

**Table 1. Temporal results on the 2023–2025 test period.** Seed SD and 95% CI show variation across the 30 XGBoost random seeds (router fixed). Only the primary-vs-unguarded gap is a paired test; every other gap in this table is contemporary context from a different saved run — do not read them as formal wins. RMSE, MAE, and bias are m³/m³.

{{TEMPORAL_TABLE}}

**Analysis.** The primary regional model and the unguarded two-regime model are nearly tied temporally: their mean R² difference is {{GUARD_BACKBONE_DIFF}}, smaller than the seed SD shown above, though its direction is consistent across the 30 paired seeds. The global and simpler-grouping rows score lower in this saved comparison, but those are context rows rather than paired tests against the primary model. The earlier 50-feature variant is kept out of this table; see Appendix B.

![Temporal multi-seed comparison and paired LOSO fold differences](figures/main_temporal_and_loso.png)

*Figure 1.* Model comparison from saved evaluation tables. In the carried-forward figure, "Guarded primary" is the primary regional model defined above, "Backbone ablation" is the unguarded two-regime model, and "V0 sensitivity" is the earlier 50-feature model detailed in Appendix B. Error bars show predictor-seed variation, not selection uncertainty.

## 6. In-state spatial generalization: LOSO

**Table 2. Station-mean leave-one-station-out results.** Each model uses five XGBoost seeds across seven held-out Washington stations. Only the first two rows are paired; the global rows are contemporary context from a different saved run.

{{LOSO_TABLE}}

**Analysis.** The primary model's paired mean advantage over the unguarded two-regime model is {{GUARD_LOSO_DIFF}} R², with improvements on {{FOLD_GAINS}} held-out stations and ties on {{FOLD_TIES}}. This is the clearest spatial result in the paired comparison. The global-model gap is context only; it does not supply a direct paired comparison or a test of fold-by-fold wins. With seven stations, the fold count describes this set of locations and is not a population estimate.

**Table 3. Paired LOSO results by held-out station.** The two agreement columns are adjusted Rand index (ARI) scores between alternative training-set partitions: higher means the two procedures divided the six training stations more similarly. Lower agreement means the group assignments changed more after refitting.

{{T8_TABLE}}

**Analysis.** The largest gains occur for the Sourdough Gulch and Spokane holdouts, where refitting changes how the six training stations are divided. For a held-out station, predictions still use its row-level features. The station-majority rule changes which training observations each group-specific predictor sees; it does not assign a fixed group to a station that was left out. Paradise has a smaller gain. The {{FOLD_TIES}} tied folds show locations where the rule did not change the observed result. These comparisons describe model behavior and do not establish a geophysical cause.

## 7. Partition diagnostics and feature interpretation

**Table 4. Number-of-groups comparison using training and validation data.** Lower Davies–Bouldin and higher Calinski–Harabasz favor two groups here; silhouette is slightly higher at three. These summary measures do not establish physical meaning or the best choice for future data. Deployments to other regions should re-run this comparison on local data rather than assume two groups.

{{K_TABLE}}

For the seven Washington stations (membership stated in §4), this is a sample-specific pattern across the Cascade side and eastern Washington. The five-station group includes both lower-elevation sites and mountain sites, so the groups are not a strict mountain-versus-lowland split. The primary model keeps known stations in one group by design; this consistency is a model rule rather than independent evidence of a physical classification.

**Table 5. Features that differ most between the two groups.** Separation index is a saved 0–1 group-difference score from the earlier gating analysis of the same Washington split (1 = groups do not overlap on that feature); the target itself is excluded from the ranking. Profile attributes need not be router inputs — this table profiles what the groups look like, not what the K-means saw.

{{PROFILE_TABLE}}

**Analysis.** The groups also differ in site descriptors (longitude, precipitation climatology, land cover), satellite soil-moisture summaries, vegetation and temperature features, and antecedent weather. These associations help explain why the regional predictors see different inputs. In this study, the useful interpretation is a broad western/Cascade-side versus eastern-Washington pattern that also reflects elevation and local conditions. It describes these stations and does not define universal physical regimes.

## 8. Regional model interpretation at ECE stations

The primary regional model (§4) fits one soil-moisture predictor per Washington group. This is a sample-specific pattern that overlaps imperfectly with elevation: the five-station group includes both lower-elevation and mountain stations. The groups should not be read as a general mountain-versus-lowland classification.

The five ECE collaboration stations are outside the seven-station Washington training set, so the results show how the learned predictors behave at new sites. The ECE set contains {{ECE_ROWS}} daily observations from {{ECE_STATIONS}} stations between {{ECE_FIRST_DATE}} and {{ECE_LAST_DATE}}, with {{ECE_PER_STATION}} observations per site. It is a short evaluation window; the site-level results show how errors vary across those locations.

The ECE observations contain none of the SMAP features used in the standard Washington grouping (100% SMAP missing rate on the 150 rows). We therefore report the standard K-means assignment alongside a fallback assignment based on the available antecedent-precipitation index (`G_API`): split at the Washington-fit median, low → the Washington-drier predictor, high → the other predictor. Which fallback direction to use was fixed on SMAP-masked Washington validation data before scoring any ECE target; ECE targets were used only to score predictions. Single-predictor rows (run one Washington predictor on every ECE row) are non-deployable references that show what each predictor does on its own.

For the primary regional model, Washington fit data associates the western/Cascade-side stations with one predictor and Spokane plus Sourdough Gulch with the other. In the ECE tables, predictor indices 0 and 1 refer to the two fitted predictors within each model; those indices do not assign universal climate labels to ECE stations.

**Table 6. Mapping the precipitation-based fallback to each model's predictors.** Predictor indices are specific to each model fit; they are not climate labels. The fixed single-predictor rows are non-deployable references.

{{ECE_CROSSWALK}}

In the Washington fit data, the mean of `SMAP_sm_pm_interp_rollmean30` is {{WA_SMAP_C0}} in one group and {{WA_SMAP_C1}} in the other; corresponding target means are {{WA_TARGET_C0}} and {{WA_TARGET_C1}}. These summaries describe the fitted groups; group numbers are model labels, not physical class labels.

**Table 7. Fallback mapping chosen using Washington validation.** The lower-RMSE mapping was fixed before scoring ECE observations; an exact tie would select class 0 → predictor index 0.

{{ECE_CALIBRATION_TABLE}}

**Table 8. Pooled ECE results over five predictor seeds (primary model + references).** Error metrics are m³/m³. R² is omitted because target variance is very small over this short window. The unguarded two-regime model's deployable rows are bit-identical to the primary model's on this ECE set (standard RMSE {{ECE_ROUTED_RMSE}}, fallback RMSE {{ECE_AUTO_RMSE}}), so they are stated here rather than repeated as extra rows; the soft-blend fallback matches the hard fallback at reported precision and is omitted. "Usable from observed inputs?" marks deployable assignments versus single-predictor diagnostic references.

{{ECE_TABLE}}

**Analysis.** The precipitation-based fallback reduces pooled RMSE relative to the standard K-means assignment by {{ECE_AUTO_ROUTED_REDUCTION}} m³/m³. Its RMSE is {{ECE_AUTO_RMSE}}, close to the global model's {{ECE_GLOBAL_RMSE}} (difference {{ECE_AUTO_GLOBAL_GAP}}). All {{ECE_ROWS}} rows fall in precipitation class 0. Under the mapping chosen with Washington validation, that class points to the Washington-wetter predictor. Its result matches that predictor run alone (RMSE {{ECE_INDEX0_RMSE}}); the other predictor run alone has RMSE {{ECE_INDEX1_RMSE}}. In other words, on this short late-summer ECE window the fallback test exercises only one of the two regional predictors — it checks whether that predictor transfers, not whether the two-predictor system beats the global model.

**Table 9. ECE station-level errors and site context.** Elevation and annual precipitation describe the sites; they do not explain model error by themselves. The final column summarizes the fallback assignment's weight on the fixed comparison predictor (0.000 at every site: every ECE row took the same branch).

{{ECE_STATION_TABLE}}

**Analysis.** Errors vary across the five sites, so the pooled result does not describe every location. This short evaluation helps interpret the regional predictors and their assignments at new stations. Its Washington-selected mapping and late-summer observations do not establish full-season performance or a universal regime definition. ECE targets did not determine the mapping.

![Primary regional model assignment options and ECE station results](figures/ece_policies_and_stations.png)

*Figure 2.* ECE pooled and station-level RMSE. In the carried-forward plot, "Guarded" means the primary regional model; its fixed single-predictor bars are model-specific diagnostic references, not deployable assignments.

![ECE daily observations and primary regional model predictions](figures/ece_guarded_daily_overlay.png)

*Figure 3.* Daily observations and model predictions at the five ECE stations. "Guarded" in the carried-forward plot refers to the primary regional model; fixed single-predictor traces are model-specific diagnostic references.

An earlier comparison of a regional model and the global model was effectively tied: station-mean ΔRMSE (regional model minus global) {{ECE_LEGACY_DIFF}}, with {{ECE_LEGACY_WINS}} station wins and sign-test p={{ECE_LEGACY_SIGN_P}}. A separate hourly analysis found median within-day sensor range {{HOURLY_RANGE}} versus median absolute day-to-day daily-target step {{DAILY_STEP}} in its focus windows. These observations help describe the daily measurement setting; the ECE results here remain a short, site-specific view of the regional models.

## 9. Robustness, negative evidence, and limits

The primary regional model uses the shared feature set described in §4. Earlier feature-set comparisons are kept in Appendix B.

**Out-of-state comparison (one paragraph, details in ledger).** Earlier Washington-trained regional and comparator configurations evaluated on ten out-of-state stations all have negative station-mean R² (table in the claims ledger source below); the two-group variants show no reliable transfer beyond the region. These are not results for the primary regional model, so they appear here as a scope boundary, not a result table.

{{OOS_TABLE}}

**Interpretation.** The results support evaluating regional predictors for within-Washington differences under this development protocol. The out-of-state table shows that the current models do not establish transfer to other regions. Snowpack-dominated sites and missing snow-water-equivalent information are areas for future model development.

**Selection and inference limits.** Seven held-out stations provide a small spatial sample. Temporal seed intervals measure XGBoost fitting variation; they do not include uncertainty from station sampling or feature and model selection. The shared features, XGBoost settings, and some model choices were informed by the 2023–2025 test period, so the scores are development evidence. The station-majority rule applies to known stations; for new stations, predictions use row-level grouping. The ECE evaluation covers five separate stations over a short late-summer window, with a fallback mapping selected using Washington validation. It helps interpret the model behavior at those sites but does not establish broader seasonal or geographic transfer.

## 10. Paper contribution decisions and writing guidance

1. **Introduce the models in plain language.** Explain the two-regime idea, the two XGBoost predictors, the single-regime global comparator, and the station-majority rule before using any implementation names.
2. **Lead with the one paired result.** Report the temporal near-tie and the LOSO differences for primary vs. unguarded together. Present every other model gap as contemporary context, not as a test.
3. **Explain the Washington groups.** Describe the five west/Cascade-side stations and two eastern stations as a sample-specific pattern, with the mountain and lowland context stated accurately.
4. **Use the ECE section to interpret model behavior.** Explain the SMAP shortfall, the Washington-selected precipitation fallback, pooled and site-level errors, and the daily overlay — including the fact that all 150 ECE rows took the same fallback branch. Keep the small sample and short window in view.
5. **State the development-split status in abstract, results, and limitations.** Independent confirmation requires new dates or sites and a selection protocol that keeps final evaluation untouched. Future work can test the number of groups, a global fallback, more station types and snowpack features, full-season ECE data, and a direct paired regional-versus-global station holdout comparison.

**Proposed paper spine.** Introduction: the estimation question after the first paper. Related work: regional soil-moisture models. Methods: data, shared predictors, regional model, and global comparison. Results: temporal performance, held-out station performance, and group interpretation. Discussion: behavior at ECE stations, reliance on satellite and weather inputs, selection limits, and scope. Conclusion: a bounded finding about regional models for Washington stations.

## 11. Reproducibility and evidence trail

`scripts/assemble_report.py` reads the saved evidence and inserts result values into the report and claims ledger. `provenance/source_manifest.json` records exact source paths, run identities, and SHA-256 hashes. `scripts/validate_assembly.py` checks source schemas, seed and station coverage, the station-network mapping, ECE mapping provenance, figures, links, and deterministic regeneration. The figures reuse the existing report notebook and saved tables. No model training, split regeneration, or edits to versioned notebooks are required.

The claims ledger distinguishes **confirmed within the saved protocol**, **contemporary context**, **evaluation-only**, and **unresolved**. A confirmed number does not imply independent test confirmation. The ledger and manifest preserve the exact source artifacts for audit, while this report presents the results in one reader-facing account.

## 12. References

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

The following list is inserted directly from the saved feature record. It is the shared predictor input used by the global model and regional specialists. It is supplied here so the paper author can report or audit the precise feature provenance without searching experiment files.

{{FEATURE_LIST}}

## Appendix B. Earlier V0 regional model

The V0 model is an earlier 50-feature K-means regional variant retained to show how the current comparison developed. It is not part of the main claim. The primary model uses the shared 54-feature input and the station-majority rule described in §4. In the main text, V0 appears only here; the metrics and their comparison limits are collected below.

{{V0_CONTEXT_TABLE}}

**Earlier V0 feature-set comparison.** The table compares the shared V0 feature set with a version that added features selected using Washington validation. These results describe V0 only and are not a comparison of the primary model.

{{DELTA_TABLE}}

**Earlier out-of-state results.** These V0 results are part of the background record; they do not describe the current primary model.

{{V0_OOS_TABLE}}

The temporal and LOSO entries come from separate saved evaluations and are background context; they are not paired comparisons with the primary model. In an earlier comparison without regime-specific feature additions, V0 exceeded the global model by {{V0_GLOBAL_PAIRED}} temporal R². This result belongs to V0 and its saved evaluation only; it does not establish a significance result for the primary model. The main temporal figure retains V0 as a visual reference, while the results tables in §§5–6 focus on the current primary and unguarded two-regime models.
