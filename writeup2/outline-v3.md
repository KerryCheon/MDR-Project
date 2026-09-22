# Second Paper Outline (v3) — Covariate-Defined Regional Experts for Soil-Moisture Estimation

Status: active draft outline for PI · Split: `derived_8.4` only · Scope: Washington (regional) · CS/IEEE venue TBD
Historical predecessor: `outline-v2.md` (kept unchanged) · Status of this file: active draft · v3 reframes the paper around router provenance, controlled hard-gated regional experts, and deployment-available inputs.

This document is the technical-report backbone for writing the full paper without manually digging through experiment notebooks. Every number below is already present in an executed report notebook README (stdout tables), in paper 1, or in repo docs explicitly marked as historical/prior-work sources; the claims ledger (W1) freezes the mapping before drafting.

---

## 0. Locked framing decisions

1. **Claim is multi-regime vs single-regime, NOT "K=2 is universal."**
   - Our data: K=2 is best among tested K ∈ {1,2,3,4} (K=1 = global baseline).
   - Deployment elsewhere may need larger K; K-selection guidance is a contribution, not a fixed answer.
2. **No regime-specific feature selection.** Every configuration (global and all routers) uses the identical shared 54-feature backbone. Delta-feature arms (c0/c1) are OUT of the main claim; at most a one-paragraph robustness note that historical test-selected deltas barely change results (0.8126 vs 0.8118) and val-selected deltas fail.
3. **Venue-agnostic outline.** Sections written so the same material works for:
   (a) ML conference (NeurIPS/ICML workshops, AAAI/ACL-app tracks, KDD application),
   (b) AI-for-science / geoscience ML venue (AGU Fall Meeting abstract → paper, IEEE IGARSS/CIKM application track),
   (c) IEEE AIIoT-style applied venue (same family as paper 1).
   Length knobs marked per section (short = 6–8 pp; full = 10–12 pp).
4. **ECE sensors: conditional inclusion** (see §7 + works-to-do W4). Default = short "deployment stress test" subsection with heavy caveats, not a headline result. Drop entirely if reviewers/PI prefer a pure CS story.
5. **OOS/out-of-state: Limitations only** (one paragraph + numbers), per prior decision.
6. **Sequel positioning to paper 1 (NEW in v2).**
   - Paper 1 = *Enhancing Spatial and Temporal Coverage of Soil Moisture Estimation Using Satellite and Weather-Driven Machine Learning* (submitted IEEE manuscript, `paper/`).
   - Cite paper 1 for foundations: dataset preparation, feature-selection pipeline, preprocessing, global single-regime baseline performance, and (if present in the submitted PDF — verify in W-new) the prior three-regime/oracle analysis and gating limitations.
   - This paper does NOT re-litigate those results; it answers the open question paper 1 left: *what routing signal actually makes regime specialization deployable?*
   - Historical internal experiments (gating v19–v25, threshold recalibrations, adapted-training) are prior-work context only — never mixed into our result tables as comparable rows across dataset versions.
7. **Design principle as a first-class contribution (NEW in v2).**
   - Failed family A: route on the target or a proxy of it (thresholds on true/predicted SM, classifier trained to reproduce regime labels derived from y) → circular + misrouting compounds through hard gates.
   - Failed family B: supervised/heuristic routers on weak signals (learned classifier, seasonal-only, single-index) → unstable across splits (our in-paper foil: `Trained_Gating_k2`).
   - Working family C: **unsupervised routing on the shared covariate backbone** (KMeans) → never sees the target at routing time; yields station-pure, physically interpretable regimes.
   - Headline framing: "route on covariates, not on the target."
8. **Prior-attempt depth in the paper body (NEW in v2, default).**
   - Intro: one tight paragraph on the genealogy (global ceiling → 3-regime MoE idea → deployable routing failed → paper 1 shipped global).
   - Optional appendix/context table ("prior design attempts", dataset-labeled, not ranked against current results) — include only if venue has room; default = omit from main results, keep in outline as available material.
   - All historical numbers attributed to paper 1 or clearly marked as internal prior work with dataset version tags (e.g. `derived_9.0`, `derived_8.1_pos`).

---
### v3 clarifications layered on the retained decisions

- **Controlled router design, not novel architecture:** describe the implemented model as a hard-gated, cluster-wise XGBoost mixture-of-experts formulation. The contribution is the controlled router comparison, not a new MoE architecture.
- **Primary routing provenance:** the main covariate router is in-situ-target-label-free, but it is not proxy-free: the shared 54-feature backbone includes current, lagged, and rolling SMAP-derived soil-moisture features.
- **Empirical comparison:** hold the expert type, shared backbone, split, and evaluation protocol fixed while comparing target-derived, heuristic, and covariate-defined routers.
- **Narrow supervised-router claim:** treat the v2 supervised/heuristic failure language as historical motivation. The current negative result applies specifically to the tested `Trained_Gating_k2` router trained from an in-situ target threshold; do not generalize it to supervised routers as a class.
- **Terminology and interpretation:** use “covariate-defined regional strata” for the current partition. Station purity, east/west composition, and feature separation are descriptive associations; they do not establish causal climate or hydrologic mechanisms without subject-matter validation.
- **Default venue posture:** retain the v2 venue options, while using the CS/IEEE-leaning presentation and minimizing geoscience-process interpretation unless the target venue and reviewers support expanding it.

---

## 1. Title candidates — v3 framing

### Recommended

1. "Covariate-Defined Regional Experts for Soil-Moisture Estimation: A Controlled Study of Hard-Gated Routing"

### Alternatives

2. "When Does Regional Specialization Help? Router Design for Hard-Gated Soil-Moisture Experts"
3. "Hard-Gated Cluster-Wise XGBoost for Regional Soil-Moisture Estimation"
4. "Inference-Available Routing for Regional Soil-Moisture Estimation with Specialized XGBoost Experts"
5. "Beyond a Single Regional Model: A Router Comparison for Soil-Moisture Estimation"

Avoid titles that claim a novel MoE architecture, universal covariate-defined regional strata, or general failure of learned routers.

---

## 2. Abstract skeleton (6 sentences — v3 router-design framing)

1. Problem: regional soil-moisture data combine stations with different environmental conditions, site characteristics, and covariate distributions. A single pooled estimator uses one shared mapping across these groups, while expert specialization requires a routing rule that identifies which mapping to use for each prediction. That rule must operate without the contemporaneous in-situ soil-moisture target—the quantity being estimated—and instead rely on inputs available at deployment.
2. Approach: compare target-derived, heuristic, and inference-available covariate routers under a fixed hard-gated XGBoost expert design; the primary KMeans router is in-situ-target-label-free but uses satellite-soil-moisture-informed features.
3. Result temporal: R² 0.812 vs 0.780 (RMSE 0.044 vs 0.048), 30 seeds, p < 1e-12, sample bootstrap ΔR² +0.035 [0.019, 0.055], p = 0.0005.
4. Result spatial (in-state LOSO): mean R² 0.64 vs 0.58, wins 6/7 stations (descriptive).
5. Analysis: the selected K=2 partition is stable and station-pure on this dataset; the target-derived `Trained_Gating_k2` foil underperforms, while covariate-defined regional strata produce the strongest current result.
6. Scope: this is a Washington regional study, not evidence that K=2 or this router transfers universally; out-of-state transfer fails for all tested models.

---

## 3. Section-by-section outline

Each section lists: goal · claims · numbers to quote · source (experiment path + table) · figures · venue knobs · open writing tasks.

### §1 Introduction [all venues] — router-design problem genealogy

- **Beat 1 — What paper 1 established.** Dataset preparation, feature pipeline, and a strong single-regime global XGBoost model established the baseline; prior error analysis motivated specialization. Cite paper 1 only for claims verified in the submitted manuscript.
- **Beat 2 — Why router provenance matters.** Earlier target-threshold and target-derived routing attempts show why oracle specialization is not the same as deployable specialization: labels derived from the in-situ target are unavailable at inference and hard gates amplify boundary errors.
- **Beat 3 — What this paper studies.** On the clean 7-station `derived_8.4` split, hold the prediction backbone and XGBoost experts fixed while comparing target-derived, seasonal/API, dynamic, and covariate-defined routers.
- **Contributions:**
  - C-A: A hard-gated, cluster-wise XGBoost formulation for regional soil-moisture estimation, treated as an MoE formulation rather than a novel MoE architecture.
  - C-B: A controlled comparison of router provenance under shared features, multi-seed temporal evaluation, in-state LOSO, and block bootstrap.
  - C-C: An empirical result that the selected covariate-defined regional partition improves temporal performance and wins most in-state LOSO station comparisons.
  - C-D: A failure analysis showing that the tested target-derived gate is not deployable and underperforms; this does not generalize to all supervised routers.
  - C-E: A reporting framework covering feature availability, satellite-soil-moisture proxy use, K selection, partition stability, and deployment limits.
- **Scope boundary:** station-group and feature-separation results are descriptive associations, not causal hydrologic explanations.
- Sources: paper 1 PDF; `writeup/sections/three_regime/*`; `docs/gating.md`; `docs/plans/20260703-why-MoE.md`; `derived_8.4-gating-analysis-1.0`.
- Knobs: ML/IEEE venues emphasize router provenance and evaluation; geoscience venues may expand interpretation only after subject-matter review.

### §2 Related work [all venues] — direct-overlap audit and positioning

- **Pillar 0 — Positioning against paper 1.** Distinguish the submitted paper’s global model and prior regime analysis from this paper’s controlled router comparison.
- **Pillar 1 — Direct soil-moisture MoE papers.** Discuss Yang et al. (2025), “An Interpretable framework of Soil moisture estimation based on Mixture-of-Experts (ISMoE)” (*Journal of Hydrology* 661, 133763), which uses CNN-LSTM experts and adaptive routing across Tibetan Plateau climate zones; Zhang et al. (2026), “An Antecedent-Precipitation-Informed Soil Water Balance and Time-Aware Mamba–MoE Framework for Surface Soil Moisture Forecasting” (*Remote Sensing* 18, 3125), which combines a physical prior, time-aware Mamba, and context-based MoE; Singh et al. (2026), “Weak Physics-Guided Multi-Agent Learning for Surface to Subsurface Moisture Estimation Across Diverse Climate and Soil Conditions” (*JGR: Machine Learning and Computation*), with dry/intermediate/wet specialists; and Wu et al. (2026), “An RTE-guided mixture-of-experts framework for AMSR2 passive microwave soil moisture retrieval” (*International Journal of Applied Earth Observation and Geoinformation*).
- **Pillar 2 — Clustered and regional soil-moisture models.** Discuss Chakrabarti et al. (2016), “Disaggregation of Remotely Sensed Soil Moisture in Heterogeneous Landscapes Using Holistic Structure-Based Models” (*IEEE TGRS* 54), which uses soft clustering followed by cluster-specific kernel regressors; Ding et al. (2024), “Addressing spatial gaps in ESA CCI soil moisture product: A hierarchical reconstruction approach using deep learning model” (*International Journal of Applied Earth Observation and Geoinformation* 132), which uses KMeans-defined climate subregions with specialized gap-filling models; and Stahl & McColl (2022), “The Seasonal Cycle of Surface Soil Moisture” (*Journal of Climate* 35), which uses KMeans to identify global surface-soil-moisture seasonal regimes.
- **Pillar 3 — Hydrologic MoE and router selection.** Discuss Moges et al. (2016), “Hierarchical mixture of experts and diagnostic modeling approach to reduce hydrologic model structural uncertainty” (*Water Resources Research*), on indicator-driven hierarchical mixtures, and Rong et al. (2025), “Mixture of experts leveraging Informer and LSTM variants for enhanced daily streamflow forecasting” (*Journal of Hydrology*), on heterogeneous LSTM/GRU/Informer experts with RF/LSTM/Transformer routers.
- **Pillar 4 — Hybrid model terminology.** Distinguish LSTM+XGBoost feature fusion or residual stacking from MoE routing: different model types alone do not constitute a mixture of experts unless a router selects or weights expert outputs.
- **Pillar 5 — Evaluation practice.** Position LOSO, multi-seed inference, block bootstrap, feature provenance, target-label leakage, and deployment-availability checks as the gap addressed here.
- **Positioning sentence:** the reviewed literature establishes that soil-moisture MoE and cluster-specific modeling already exist; this paper studies how router provenance affects hard-gated regional XGBoost performance under a fixed, leakage-audited evaluation protocol.
- **Open task W2:** create verified BibTeX records and preserve DOI/venue metadata for every direct comparator; no citations invented in the outline.

### §3 Problem setup & data [all venues] — expanded with dataset evolution

- `derived_8.4`: 7 WA stations; train 2017–20 (9,803) / val 2021–22 (4,805) / test 2023–25 (6,620); station list with one-line descriptors (east/west, elevation).
- Stations: BeaverPass_WA_990, CayusePass_WA, Darrington, Paradise_WA, Quinault, SourdoughGulch_WA_985, Spokane.
- **Dataset evolution paragraph (NEW):** earlier expansion roughly doubled station count; many added stations removed for data-quality issues and because the current feature set lacks snow/SWE variables needed for snowpack-dominated sites (alpine stations out of scope; cite `docs/plans/20260721-remove-incomplete-stations.md`, `docs/plans/20260726-stations-removal.md`). Result: clean, deployment-relevant 7-station split. One sentence: snowpack regimes are future work (SWE/Snow Depth features), not a silent omission.
- Feature backbone: 54 shared features (identical for every model) — name the source (`derived_8.4-feature-selection-2.0` / eval-1.1 `selected_features.json`); foundations of the selection pipeline cite paper 1; state that the backbone was fixed before this comparison and shared across arms (relative claims).
- Target: `soil_moisture_5cm`; metrics: R², RMSE, MAE, bias (+ Pearson where useful).
- **Cross-dataset caution (R5):** any comparison to paper 1's 0.822 must note different split/dataset version/protocol; never place it in the same ranking table as `derived_8.4` rows without a footnote.
- Sources: `data/splits/derived_8.4/split_meta.json`; `derived_8.4-formal-eval-1.0` protocol section; `derived_8.4-gating-analysis-1.0` feature-list tables; paper 1 §data/features; station-removal plans.

### §4 Method: hard-gated cluster-wise XGBoost (MoE formulation) [all venues]

- Architecture: router → hard assignment → one XGBoost expert per assigned group; inference activates one expert and a small router. The formulation is an MoE, but the paper does not claim a new MoE architecture.
- **Router alternatives and provenance:**
  - A1: Target-derived threshold gate on `soil_moisture_5cm < 0.16` — the in-paper `Trained_Gating_k2` foil; unavailable without the in-situ target and not deployable as an estimator router.
  - A2: Heuristic routers — `Seasonal_Binary_k2` and `Univariate_G_API_k2`; inference-available but limited state/season rules.
  - A3: Covariate routers — `Clustering_Dynamic_k2` and `Clustering_V0_Full_k2`; fitted without the in-situ target and applied to inference-available inputs.
  - A4 (primary): KMeans on the shared 54-feature backbone; label it **in-situ-target-label-free, satellite-soil-moisture-informed covariate routing**, not proxy-free routing.
- **Router provenance table:**

| Family | Configuration | In-situ target used to fit router? | SMAP-derived inputs? | Deployment disclosure |
|---|---|---:|---:|---|
| Target-derived | `Trained_Gating_k2` | Yes | Not required | Diagnostic foil only; unavailable without target labels |
| Seasonal heuristic | `Seasonal_Binary_k2` | No | No | Calendar rule; deployable where date is available |
| API heuristic | `Univariate_G_API_k2` | No | No | Deployable if API inputs are available |
| Dynamic covariate | `Clustering_Dynamic_k2` | No | Yes, lagged SMAP feature | Requires the same satellite product/latency assumptions |
| Full covariate | `Clustering_V0_Full_k2` / `Clustering_Backbone54_k2` | No | Yes, current/lagged/rolling SMAP features | Primary result; explicitly satellite-SM-informed |
| Global baseline | `Global_Single_54` | No router | Yes in prediction backbone | Single shared model |

- Router families compared: retain the existing six configurations and choose one paper-wide name for the equivalent V0/Backbone54 K=2 partition.
- Training protocol: routers fit on trainval only; experts per group; 30 temporal seeds; LOSO 5 seeds × 7 folds with per-fold refitting. State exactly which components are stochastic.
- Statistics: retain seed-level confidence intervals, paired tests, BH-FDR, station-month bootstrap, LOSO win counts, and the explicit limitation that seed variation measures expert fitting stochasticity only.
- **K-selection disclosure:** select K using train/validation or predeclared clustering diagnostics, then reserve test for final evaluation.
- **Feature-availability disclosure:** distinguish in-situ target labels from satellite soil-moisture proxies and state whether each input exists at intended deployment time.
- No claim is made that hard routing is superior to soft routing; the paper compares the implemented hard-gated router families only.
- Sources: `derived_8.4-formal-eval-1.0`; `derived_8.4-gating-analysis-1.0`; `eval11/routers.py`; `derived_8.4-regime-interpretation-1.1`; direct literature listed in §2.

### §5 Partition diagnostics: covariate-defined regional strata

[ML/IEEE venues: 0.5–1 page · geoscience venues: expand only with subject-matter review]

- K=2 purity = 1.000 trainval and test; K=3 mean purity 0.833; K=4 0.695. Report this as partition stability and station-group concentration, not proof of universal covariate-defined regional strata.
- Partition composition: cluster 1 = Spokane + SourdoughGulch; cluster 0 = five western/mountain stations. Use descriptive labels such as “eastern/drier station group” and “western/mountain station group.”
- Top separating features include static/site descriptors, SMAP rolling statistics, and LST/NDMI windows. Treat these as associated partition indicators, not causal drivers.
- Correlation drift, including `SMAP_sm_pm_interp_rollmean30`, is evidence that the pooled mapping may differ across strata; it does not establish a physical mechanism.
- Quality indices: Calinski-Harabasz and Davies-Bouldin favor K=2 while silhouette is marginally higher at K=3. Internal indices alone do not select the meaningful partition.
- Explicit caveat: station purity may reflect site descriptors or station identity proxies. It diagnoses specialization on this dataset but may reduce transferability to unseen stations.
- Contrast with the target-derived gate: target labels create wet/dry separation but do not yield stable station groups and are unavailable at inference.
- Sources: `derived_8.4-gating-analysis-1.0`; `derived_8.4-regime-interpretation-1.0/1.1/1.2`; feature provenance in §4.

### §6 Results [core of every venue]

#### 6.1 Temporal (primary) — no-delta only

- Headline table (rank by RMSE): at minimum
  - `Clustering_V0_Full_k2` (0,0): R² 0.8118 ± 0.0014, RMSE 0.0442
  - `Global_Single_54`: 0.7798 ± 0.0013, RMSE 0.0478
  - `Trained_Gating_k2` (0,0): 0.7354 ± 0.0011
  - plus Dynamic / Seasonal / Univariate no-delta rows for the full router comparison
- Pairwise: clustering vs global ΔR² +0.032 [0.031, 0.033], p < 1e-12, 100% seeds; clustering vs trained gating +0.089. Bootstrap (station, month): ΔR² +0.035 [0.019, 0.055] p = 0.0005; RMSE −0.0040 p = 0.0005.
- **Router-design sentence (NEW):** specialization is not sufficient by itself; under this fixed expert/evaluation setup, the tested covariate-defined regional partition outperforms the global baseline, while the target-derived gate underperforms. This is evidence about router provenance on this benchmark, not a universal ranking of supervised versus unsupervised routers.
- Optional footnote only (not a table row): paper 1 reported 0.822 under a different dataset/split — not directly comparable (R5).
- Source: `derived_8.4-formal-eval-1.0` temporal seed table (filter `delta_source=none`) + focused pairwise + bootstrap tables.
- Figure F1: bar/dot temporal R² with 95% CI across ~8 configs (NEW figure notebook).

#### 6.2 In-state LOSO (secondary, descriptive)

- No-delta: `Clustering_V0_Full` 0.637 / `Backbone54` 0.621 vs `Global_Single_54` 0.580 (and `Baseline_V0_50` 0.591). Wins 6/7 vs global (sign p = 0.125 — state underpowered).
- Station difficulty table (7 stations, median LOSO R² across configs): Darrington 0.70, BeaverPass 0.69, Spokane 0.65, Paradise 0.59, Quinault 0.56, SourdoughGulch 0.43, CayusePass 0.39.
- Explicit sentence: LOSO supports consistency, not statistical significance at n = 7.
- Source: `derived_8.4-formal-eval-1.0` LOSO summary + `loso_pairwise_focused` (filter none); `derived_8.4-eval-1.3` for difficulty narrative if needed.
- Figure F2: per-station paired R² (existing `loso_pair` / `loso_station_bars` PNGs).

#### 6.3 Router comparison (the multi-regime story)

- Ranking message: the full covariate-defined regional partition is strongest in this dataset; heuristics and the target-derived gate are weaker under the same protocol. Do not generalize this single target-derived foil to supervised routing as a class.
- Optional single table merging 6.1 + 6.2 columns (temporal R² | LOSO R²).

#### 6.4 Robustness note (1 paragraph, not a section)

- No-delta ≈ historical test-delta winner (0.8118 vs 0.8126) → headline is not a delta-selection artifact.
- Val-selected deltas historically UNDERPERFORM global (0.735 vs 0.780) → selection instability exists in this benchmark; our simplified no-delta protocol avoids it.
- Optional: one sentence that the shared backbone was itself test-era-selected (shared by all arms; absolute numbers optimistic, relative comparison intact).
- Source: `derived_8.4-formal-eval-1.0` delta-robustness table + caveat bullets.

#### 6.5 (Optional, venue-dependent) Prior-attempts context material

- NOT a ranked results row. If included: small appendix table listing prior design attempts (threshold hard/soft, supervised classifier, adapted training, threshold recalibration, dataset expansion) with **dataset version tags** and qualitative outcome (worked / failed / partially), sourced from paper 1 + internal docs — purpose is narrative honesty, not benchmarking.
- Default: omit from main paper; keep outline entry for PI option (fold into W4-style venue decision).

### §7 Deployment stress test — ECE sensors

[OPTIONAL subsection; default INCLUDE as ~0.5 page under Discussion or a short §7]

- Purpose: not to show superiority — all models are similar/bad — but (i) a real new-deployment transfer attempt on unseen in-situ sensors, (ii) what breaks when you evaluate there.
- What we can honestly say:
  1. Protocol: train on 7 WA trainval; evaluate 5 ECE stations (`derived_8.4_ece_v3`, 150 rows, 2026-07-20..08-19, ~30-day window).
  2. Outcome: two-regime ≈ single-regime on RMSE (ΔRMSE +0.00025, 2/5 wins, ns); all R² deeply negative because σ_y ∈ [0.003, 0.008] — rank by RMSE, not R² (metric-realism lesson).
  3. Predictions collapse toward a common value across sensors with very different target means (0.018–0.198) — line-chart observation.
  4. Sensor diagnosis (`ece-input-diagnose-1.0`): hourly data show strong diurnal cycle (median hourly range 0.035 ≈ 14× median daily step 0.0025); rain falls overnight and dries by midday, so 24-h means look flat; hour-scale corr(rain, Δsensor) ≈ 0 at all lags 0–6h. So: NOT "sensors ignore rain" but "daily aggregation + short dry-down window make this set a weak benchmark for daily models."
  5. Takeaway for the paper: sensor → benchmark validity checks (variance floor, event-response check, aggregation choice) should precede model ranking.
- What we must NOT say: that two-regime wins on ECE; that sensors are broken; any causal claim about sensor hardware.
- Decision rule W4: if PI wants a pure methods/CS paper with no deployment narrative, cut §7 entirely — nothing else depends on it.
- Sources: `derived_8.4-formal-eval-2.1-ece-v3` README (tables 1–3 + time-series figs); `ece-input-diagnose-1.0` README §4–5 + figures; `derived_8.4-ece-error-analysis-1.0` if needed.

### §8 Discussion

- **What the controlled router comparison establishes:**
  1. A router trained from an in-situ target threshold is unavailable at inference and performs poorly in this benchmark; this is a result about the tested target-derived gate.
  2. Inference-available covariate partitions can produce stable station-group specialists and improve temporal performance under the current regional split.
  3. The primary router uses SMAP-derived soil-moisture proxies, so the correct claim is in-situ-target-label-free and satellite-SM-informed, not proxy-free.
  4. Station purity and feature separation are descriptive diagnostics; they do not prove causal climate or hydrologic regimes.
  5. Oracle upper bounds diagnose specialization potential, not deployability.
- **Why the target-derived gate loses in-paper:** target-threshold labels define wet/dry states that are unavailable at prediction time and may assign heterogeneous station groups to the same expert. Avoid saying that all learned or supervised routers fail.
- **Regional strata, not universal regimes:** select K using deployment-region diagnostics, partition stability, group balance, and validation performance. Expect the useful number and meaning of strata to change across regions.
- **Deployment interpretation:** the ECE audit and out-of-state results are essential counterevidence: regional specialization can help in-region while failing under station or distribution shift.
- **Regional scope by design:** the contribution is an empirical router-design result for Washington soil-moisture estimation, not a general hydrologic law.

### §9 Limitations [required]

- n = 7 LOSO, low power; partial 2025 coverage; seed variation does not cover router stochasticity (routers fixed); shared backbone/hyperparameters test-era-selected.
- **OOS paragraph:** train on 7 WA → 10 out-of-state stations: ALL models degrade (station-mean R² negative for every config; pooled global 0.206 vs best two-regime ~0.11–0.14; two-regime worse than global, e.g. V0_Full no-delta wins 2/10). Frame: regime specialization is a within-region prior; outside the region both experts and router are OOD — consistent with regional scope, not a hidden result. Source: `derived_8.4-formal-eval-2.0` summary + Table 1 (no-delta rows only).
- **Snowpack / feature-space boundary (NEW):** stations under alpine snowpack dynamics removed/out-of-scope because the feature set lacks SWE/snow depth; algorithm gating cannot fix missing physics (cite station-removal plan). Future work: SWE integration.
- ECE caveats if §7 kept: 30-day window, tiny target variance, daily aggregation, and possible mismatch between the trained SMAP-informed router and unseen in-situ sensor inputs.
- **Router provenance limitation:** the primary router is in-situ-target-label-free but uses SMAP-derived soil-moisture proxies; deployment requires documenting product availability, latency, missingness, and sensor/domain mismatch.
- **Interpretation limitation:** static/site-driven purity and feature correlations are descriptive; causal geoscience interpretation requires subject-matter validation.
- Historical/attribution limitation (optional one line): some prior routing results live only in an internal tech report / pre-camera-ready analysis — we cite paper 1 where possible and do not over-claim unpublished numbers.

### §10 Conclusion & future work

- Restate design lesson in one line (under a fixed hard-gated XGBoost setup, inference-available covariate-defined regional strata made specialization useful in-region, whereas the tested target-derived gate did not).
- Future work bullets (not claims): automated K selection; OOD/ambiguity fallback to global expert (diagnostics exist in `derived_8.4-formal-eval-2.0` Table 4); station-count / training-set composition ablation (explicitly future); snowpack-aware features (SWE) + expanded station types; multi-year ECE evaluation once sensors accumulate a full season; revisit 3-way partitions only with a routing signal that is not derived from y.

---

## 4. Figure & table plan (all via notebook, `nb execute --uv`)

| ID | Item | Status | Source |
|----|------|--------|--------|
| T0 | (optional) Prior-attempts context table, dataset-labeled | new from paper1+docs | §6.5 / appendix |
| T1 | Config table (routers × features × protocol) | new LaTeX from method text | gating-analysis + eval config |
| T2 | Temporal seed-level results (no-delta) | filter existing table | `formal-eval-1.0` |
| T3 | Temporal pairwise (global, gating) | existing focused rows | `formal-eval-1.0` |
| T4 | LOSO summary + win counts (no-delta) | filter existing | `formal-eval-1.0` |
| T5 | K-sweep partition stability/quality | existing | `gating-analysis-1.0` |
| T6 | Regional-strata station composition + associated features | existing | gating-analysis / regime-interpretation |
| T7 | (opt) ECE RMSE pairwise + station means | existing | `formal-eval-2.1-ece-v3` |
| F1 | Temporal multi-seed dot/bar + CI | NEW | `paper2-figures-1.0` notebook |
| F2 | LOSO per-station pairs | reuse PNG | `formal-eval-1.0` |
| F3 | Regional-strata geographic map (WA) | NEW or reuse | gating geographic PNGs exist — pick/crop |
| F4 | Delta-robustness slope (test/val/none) | NEW, optional | `formal-eval-1.0` |
| F5 | (opt) ECE time-series line charts | reuse ≤5-line figs | `formal-eval-2.1-ece-v3` |
| F6 | (opt) Concept figure: router provenance families vs chosen | NEW, simple diagram | method narrative only |

---

## 5. Claims ledger (excerpt — full ledger = separate file in works)

| # | Claim | Config / source | Number | Artifact |
|---|-------|-----------------|--------|----------|
| C0 | Prior work: global baseline + regime motivation | paper 1 | R² 0.822 (paper-1 protocol) | `paper/` PDF |
| C1 | Prior work: specialization potential under oracle; deployable routing was the gap | paper 1 (if in PDF) / internal prior | oracle ≫ global; e2e routing weak | paper 1 §three_regime (verify) |
| C2 | Two-regime > global temporal (OURS) | V0_Full (0,0) vs Global_54 | 0.8118 vs 0.7798; Δ +0.032 p < 1e-12 | `formal-eval-1.0` |
| C3 | Sample-level significance | same | ΔR² +0.035 [0.019, 0.055] p = 0.0005 | `formal-eval-1.0` bootstrap |
| C4 | Two-regime > global LOSO | V0_Full (0,0) | 0.637 vs 0.580; 6/7 wins | `formal-eval-1.0` LOSO |
| C5 | Target-derived gating inferior temporal (specific in-paper foil) | (0,0) | 0.735 vs 0.812 | `formal-eval-1.0` |
| C6 | K=2 yields a stable station-group partition; K>2 fragments it | Backbone54 | purity 1.000 / 0.833 / 0.695 | `gating-analysis-1.0` |
| C7 | Partition is descriptively eastern/drier vs western/mountain station groups | composition table | Spokane + Sourdough = C1 | `gating-analysis-1.0` |
| C8 | OOS: no model transfers; 2-regime ≤ global | no-delta arms | mean R² < 0 all; Δ −0.17 (2/10) | `formal-eval-2.0` |
| C9 | ECE: models tie; R² invalid at low σ_y | no-delta | ΔRMSE +0.00025 ns | `formal-eval-2.1` |
| C10 | ECE daily means hide diurnal/event response | n/a | hourly range 0.035 vs daily step 0.0025 | `ece-input-diagnose-1.0` |
| C11 | Dataset evolution: expansion → prune → 7 stations (quality + snow fit) | plans | 5→13→7 (approx.; verify exact counts) | removal plans + split_meta |
| C12 | Design lesson: router provenance matters under the fixed benchmark | C5 + method narrative | qualitative; target-derived foil is specific, not universal | §8.1 |
| C13 | Primary router is in-situ-target-label-free but satellite-SM-informed | shared backbone + router table | SMAP features present; deployment availability required | §3/§4 |

Every draft sentence must map to a ledger row; no orphan numbers. C0/C1 must be checked against the actual submitted PDF before citation (W-new).

---

## 6. Venue adaptability matrix

| Element | ML conf (6–8p) | Geo/AGU-style | Applied IEEE (AIIoT-like) |
|---------|----------------|---------------|---------------------------|
| §1 genealogy beats | full (2–3) | short | short, cite paper 1 |
| §4 design-alternatives | full | medium | medium |
| §5 interpretability | short | full | medium |
| §6.5 prior-attempts table | omit or appendix | omit | optional appendix |
| §7 ECE | optional/omit | include (deployment) | include (motivation) |
| Stats detail | protocol emphasis | metrics emphasis | protocol lighter |
| OOS limitation | 3 sentences | 1 short para | 1 short para |
| Related work | MoE-heavy | SM-heavy | balanced |

---

## 7. Works to do (ordered)

- **W0** Freeze v3 claim scope: primary configs = no-delta only; describe the model as hard-gated cluster-wise XGBoost; choose one paper-wide router name; leave `outline-v2.md` unchanged. *(decision, no code)*
- **W-new** Open `paper/*.pdf` and confirm whether three-regime/oracle/limitation content is in the **submitted** manuscript (vs only `writeup/`); record which C0/C1 sentences are safely citable to paper 1. Also confirm exact station counts over time for C11 (5→13→7) from split history before freezing numbers.
- **W1** Write full claims ledger → `writeup2/claims-ledger.md` from `formal-eval-1.0` / `2.0` / `2.1` + `gating-analysis` + `ece-input-diagnose` READMEs (stdout-only numbers for our results; paper-1 citations for C0/C1).
- **W2** Related-work search + BibTeX: include the direct soil-moisture MoE papers, clustered soil-moisture models, seasonal-regime work, and hydrology MoE/router papers listed in §2; verify DOI/venue metadata and tag venue-fit.
- **W3** New figure notebook `notebooks/experiment/paper2-figures-1.0/` producing F1 (+F4/F6 if kept) from existing CSVs; `nb execute --uv`; figures saved under the experiment dir; tables in writeup2 filled only from notebook stdout.
- **W4** ECE inclusion decision (PI): include §7 vs cut. Default plan includes it. Same pass: include T0/§6.5 or not.
- **W5** Create/refresh `writeup2/` skeleton: use `outline-v3.md` as the active outline; retain `outline-v2.md` unchanged for history; add `claims-ledger.md`, `figures/`, and later `main.tex` or `main.md`.
- **W6** Draft §3–§6 (method + results) first — fully supported; then §1 genealogy + §2, §5, §8–10.
- **W7** PI review pass against ledger + title-route pick (A/B/C); then venue-specific trimming via matrix §6.
- **W8** Reproducibility and framing gate before submission: re-execute `formal-eval-1.0` and `paper2-figures`; confirm every table row = stdout; audit router feature availability and K-selection provenance; verify `outline-v2.md` is unchanged; run `make lint` / `make test` if src is touched (expected: no).

Not doing (this paper): station-count ablation, OOS full section, broad architecture redesign, edits to `writeup/` or `paper/`, archived notebooks, re-running historical v19–v25 gates. Any router-provenance audit must remain minimal and must not be presented as a new model campaign.

---

## 8. Risks

- **R1:** LOSO non-significance misrepresented — mitigate: fixed wording in T4 caption.
- **R2:** K=2 or “regimes” read as universal — mitigate: use regional strata terminology, pretest K-selection disclosure, and explicit regional scope.
- **R3:** ECE section distracts or invites reviewer attack — mitigate: W4 cut option; caveats written into the subsection, not buried.
- **R4:** Backbone selected on test — reviewers may challenge absolutes — mitigate: shared-backbone sentence in §3 + §6.4; never compare to literature absolutes without note.
- **R8:** Direct literature overlap makes an architecture-novelty claim untenable — mitigate: frame the paper as a controlled router-provenance study and cite direct soil-moisture MoE work in §2.
- **R9:** SMAP features make “target/proxy-free” inaccurate — mitigate: use in-situ-target-label-free/satellite-SM-informed language and include the provenance table.
- **R10:** Perfect station purity may reflect static station descriptors rather than transferable regimes — mitigate: call it a dataset-specific partition diagnostic and retain ECE/OOS failures as limitations.
- **R11:** Causal geoscience interpretation exceeds team expertise — mitigate: report descriptive associations only and require subject-matter validation before strengthening claims.
- **R5 (NEW):** Cross-dataset number confusion (paper 1's 0.822 vs our 0.780/0.812) — mitigate: never co-rank; footnote-only comparisons; dataset version tags on any historical table (T0).
- **R6 (NEW):** Over-attributing unpublished internal failure numbers as if peer-reviewed — mitigate: cite paper 1 for what it actually contains (W-new); label internal numbers "prior/internal analysis" or omit from camera-ready.
- **R7 (NEW):** Sequel narrative reads as dumping on paper 1 — mitigate: tone = "open question left by prior work," contributions credit paper 1 for pipeline + diagnosis.

---

## 9. Primary source map (where every number lives)

| Source | Role in paper |
|--------|---------------|
| `paper/*.pdf` | Paper 1 — foundations citation; prior regime analysis (verify content) |
| `writeup/sections/three_regime/*` | Detailed prior 3-regime narrative (cross-check vs submitted PDF) |
| `writeup/sections/base_model/*`, `writeup/sections/shared/*` | Paper-1 pipeline/features/data prose to cite, not copy |
| `docs/gating.md` | Internal history of threshold/hard/soft/classifier gates (understanding; cite only if venue accepts tech report) |
| `docs/plans/20260703-why-MoE.md` | Motivation background only (re-verify any number before quoting) |
| `docs/plans/20260721-remove-incomplete-stations.md` | Dataset pruning: incomplete stations |
| `docs/plans/20260726-stations-removal.md` | Dataset pruning: snowpack/out-of-scope stations |
| `notebooks/experiment/derived_8.4-formal-eval-1.0/` | Temporal + LOSO formal stats (core results) |
| `notebooks/experiment/derived_8.4-gating-analysis-1.0/` | K-sweep, purity, regime composition, separating features |
| `notebooks/experiment/derived_8.4-regime-interpretation-1.1/` | Method narrative, physical interpretation of regimes |
| `notebooks/experiment/derived_8.4-eval-1.3/` | LOSO protocol provenance, station difficulty (secondary) |
| `notebooks/experiment/derived_8.4-formal-eval-2.0/` | OOS numbers for Limitations only |
| `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/` | ECE transfer results (§7 if kept) |
| `notebooks/experiment/ece-input-diagnose-1.0/` | ECE sensor/benchmark validity diagnosis (§7 if kept) |
| `data/splits/derived_8.4/split_meta.json` | Split definition for §3 |
| External direct-overlap literature | Related-work records listed in §2; verify DOI/venue metadata in W2 |
