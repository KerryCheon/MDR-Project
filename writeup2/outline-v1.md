# Second Paper Outline (v1) — Regional Multi-Regime MoE for Soil Moisture

Status: draft outline for PI · Split: `derived_8.4` only · Scope: Washington (regional) · CS venue TBD

This document is the technical-report backbone for writing the full paper without manually digging through experiment notebooks. Every number below is already present in an executed report notebook README (stdout tables); the claims ledger (W1) freezes the mapping before drafting.

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

---

## 1. Title candidates (pick at drafting)

- "Two Regimes Are Enough Here: Unsupervised Climate-Regime Routing for Regional Soil Moisture Estimation"
- "Hard-Gated Mixture-of-Experts with Unsupervised Regime Routing for In-Region Soil Moisture Prediction"
- "When Should Soil Moisture Models Be Split by Regime? A Regional Study with Formal Multi-Seed Evaluation"

---

## 2. Abstract skeleton (6 sentences)

1. Problem: a single global tabular model must compromise across wet/dry and east/west soil-moisture behaviors; regional (not continental) deployment target.
2. Method: hard-gated 2-expert MoE; router = KMeans on the same 54 features the single model uses (no extra features, no regime-specific selection); XGBoost experts; multi-seed + block-bootstrap protocol.
3. Result temporal: R² 0.812 vs 0.780 (RMSE 0.044 vs 0.048), 30 seeds, p < 1e-12, sample bootstrap ΔR² +0.035 [0.019, 0.055], p = 0.0005.
4. Result spatial (in-state LOSO): mean R² 0.64 vs 0.58, wins 6/7 stations (descriptive).
5. Analysis: K=2 is the largest K with station-pure regimes on this dataset; regimes = semi-arid east vs maritime west; supervised gating is consistently worse.
6. Scope: regional model; out-of-state transfer fails for all models (honest limitation).

---

## 3. Section-by-section outline

Each section lists: goal · claims · numbers to quote · source (experiment path + table) · figures · venue knobs · open writing tasks.

### §1 Introduction [all venues]

- Goal: motivate regional multi-regime modeling without overselling K=2.
- Claims: (C1) soil-moisture feature–target relationships are regime-dependent (cite correlation-drift); (C2) a single global model compromises; (C3) explicit routing + specialization is a simple, auditable alternative to end-to-end MoE.
- Sources: `docs/plans/20260703-why-MoE.md` (motivation only, re-verify any number); `derived_8.4-gating-analysis-1.0` correlation-drift figures.
- Knobs: science-venue shortens §2 and expands physical motivation; ML venue does the reverse.

### §2 Related work [all venues] — NEW WRITING

- Pillars: (i) regional vs continental/global soil-moisture ML; (ii) MoE / hard vs soft gating / learned vs unsupervised routers; (iii) gradient-boosted tabular baselines for geoscience; (iv) evaluation practice (LOSO, multi-seed, selection leakage).
- Open task W2: literature search + BibTeX (no citations invented in outline).

### §3 Problem setup & data [all venues]

- `derived_8.4`: 7 WA stations; train 2017–20 (9,803) / val 2021–22 (4,805) / test 2023–25 (6,620); station list with one-line descriptors (east/west, elevation).
- Stations: BeaverPass_WA_990, CayusePass_WA, Darrington, Paradise_WA, Quinault, SourdoughGulch_WA_985, Spokane.
- Feature backbone: 54 shared features (identical for every model) — name the source (`derived_8.4-feature-selection-2.0` / eval-1.1 `selected_features.json`) and state plainly that the backbone was fixed before this comparison and shared across arms (relative claims).
- Target: `soil_moisture_5cm`; metrics: R², RMSE, MAE, bias (+ Pearson where useful).
- Sources: `data/splits/derived_8.4/split_meta.json`; `derived_8.4-formal-eval-1.0` protocol section; `derived_8.4-gating-analysis-1.0` feature-list tables.

### §4 Method: multi-regime hard-gated MoE [all venues]

- Architecture: router → hard assignment → per-regime XGBoost expert; inference cost = single model + tiny router.
- Router families compared (same 54 feats, no deltas):
  1. `Global_Single_54` (K=1 baseline)
  2. `Clustering_V0_Full_k2` / `Clustering_Backbone54_k2` (unsupervised KMeans; note V0 and Backbone54 give ARI = 1.0000 partition — pick ONE name for the paper, report the other as equivalent) — PRIMARY
  3. `Clustering_Dynamic_k2` (3-feature light router)
  4. `Seasonal_Binary_k2` (heuristic)
  5. `Univariate_G_API_k2` (heuristic)
  6. `Trained_Gating_k2` (supervised) — the foil
- Training protocol: routers fit on trainval only; experts per regime; 30 seeds temporal (seed scope = expert `random_state` only — state this); LOSO 5 seeds × 7 folds with per-fold router refit (no held-out-station leakage into routing).
- Statistics: seed-level mean ± std + t-CI; paired t / Wilcoxon + BH-FDR; sample-level paired cluster bootstrap over (station, month); LOSO win counts + sign test. Seed-level inference = fitting stochasticity ONLY (must appear in the paper).
- K-selection subsection: sweep K = 2, 3, 4 (+ quality indices K = 2..6); deployment guidance ("choose largest K with coherent, whole-group regimes on YOUR region") — not "always 2."
- Sources: `derived_8.4-formal-eval-1.0` README protocol + stats; `derived_8.4-gating-analysis-1.0` K-sweep tables; `eval11/routers.py`; `derived_8.4-regime-interpretation-1.1` Part A (method narrative).

### §5 Why regimes exist on this dataset (interpretability)

[science venues: full · ML venues: 0.5–1 page]

- K=2 purity = 1.000 trainval AND test; K=3 mean purity 0.833; K=4 0.695 — larger K fragments stations (table).
- Regime composition: cluster 1 = Spokane + SourdoughGulch (semi-arid east); cluster 0 = 5 western/mountain stations. Geographic map figure.
- Top separating features: `J_bio_bio13` (annual precip 391 vs 70), SMAP 30-day roll means, LST/NDMI windows (top-15 table).
- Correlation drift example: `SMAP_sm_pm_interp_rollmean30` flips sign across regimes — the "why specialization can help" evidence.
- Quality indices: Calinski-Harabasz and Davies-Bouldin best at K=2; silhouette marginally higher at K=3 (admit internal indices alone don't decide K — purity/interpretability does).
- Sources: `derived_8.4-gating-analysis-1.0` (all tables/figs); `derived_8.4-regime-interpretation-1.0/1.1/1.2`.

### §6 Results [core of every venue]

#### 6.1 Temporal (primary) — no-delta only

- Headline table (rank by RMSE): at minimum
  - `Clustering_V0_Full_k2` (0,0): R² 0.8118 ± 0.0014, RMSE 0.0442
  - `Global_Single_54`: 0.7798 ± 0.0013, RMSE 0.0478
  - `Trained_Gating_k2` (0,0): 0.7354 ± 0.0011
  - plus Dynamic / Seasonal / Univariate no-delta rows for the full router comparison
- Pairwise: clustering vs global ΔR² +0.032 [0.031, 0.033], p < 1e-12, 100% seeds; clustering vs trained gating +0.089. Bootstrap (station, month): ΔR² +0.035 [0.019, 0.055] p = 0.0005; RMSE −0.0040 p = 0.0005.
- Source: `derived_8.4-formal-eval-1.0` temporal seed table (filter `delta_source=none`) + focused pairwise + bootstrap tables.
- Figure F1: bar/dot temporal R² with 95% CI across ~8 configs (NEW figure notebook).

#### 6.2 In-state LOSO (secondary, descriptive)

- No-delta: `Clustering_V0_Full` 0.637 / `Backbone54` 0.621 vs `Global_Single_54` 0.580 (and `Baseline_V0_50` 0.591). Wins 6/7 vs global (sign p = 0.125 — state underpowered).
- Station difficulty table (7 stations, median LOSO R² across configs): Darrington 0.70, BeaverPass 0.69, Spokane 0.65, Paradise 0.59, Quinault 0.56, SourdoughGulch 0.43, CayusePass 0.39.
- Explicit sentence: LOSO supports consistency, not statistical significance at n = 7.
- Source: `derived_8.4-formal-eval-1.0` LOSO summary + `loso_pairwise_focused` (filter none); `derived_8.4-eval-1.3` for difficulty narrative if needed.
- Figure F2: per-station paired R² (existing `loso_pair` / `loso_station_bars` PNGs).

#### 6.3 Router comparison (the multi-regime story)

- Ranking message: unsupervised climate clustering ≈ or > heuristics > supervised gating on this dataset; global sits in the middle temporally, near bottom on LOSO.
- Optional single table merging 6.1 + 6.2 columns (temporal R² | LOSO R²).

#### 6.4 Robustness note (1 paragraph, not a section)

- No-delta ≈ historical test-delta winner (0.8118 vs 0.8126) → headline is not a delta-selection artifact.
- Val-selected deltas historically UNDERPERFORM global (0.735 vs 0.780) → selection instability exists in this benchmark; our simplified no-delta protocol avoids it.
- Optional: one sentence that the shared backbone was itself test-era-selected (shared by all arms; absolute numbers optimistic, relative comparison intact).
- Source: `derived_8.4-formal-eval-1.0` delta-robustness table + caveat bullets.

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

- Why learned gating loses: routing noise + expert co-adaptation gap (link to adapted-training history only as related experience, or omit); unsupervised router is stable and station-interpretable.
- Multi-regime, not two-regime: decision procedure for choosing K on a new region (purity/coherence, quality indices, physical interpretability); expect K > 2 where climate zones multiply.
- Regional scope by design (PI position: remote sensing / domain-specific models are legitimate; cite paper-1 scope statement if appropriate).

### §9 Limitations [required]

- n = 7 LOSO, low power; partial 2025 coverage; seed variation does not cover router stochasticity (routers fixed); shared backbone/hyperparameters test-era-selected.
- **OOS paragraph:** train on 7 WA → 10 out-of-state stations: ALL models degrade (station-mean R² negative for every config; pooled global 0.206 vs best two-regime ~0.11–0.14; two-regime worse than global, e.g. V0_Full no-delta wins 2/10). Frame: regime specialization is a within-region prior; outside the region both experts and router are OOD — consistent with regional scope, not a hidden result. Source: `derived_8.4-formal-eval-2.0` summary + Table 1 (no-delta rows only).
- ECE caveats if §7 kept: 30-day window, tiny target variance, daily aggregation.

### §10 Conclusion & future work

- Future work bullets (not claims): automated K selection; OOD/ambiguity fallback to global expert (diagnostics exist in `derived_8.4-formal-eval-2.0` Table 4); station-count / training-set composition ablation (explicitly future); multi-year ECE evaluation once sensors accumulate a full season.

---

## 4. Figure & table plan (all via notebook, `nb execute --uv`)

| ID | Item | Status | Source |
|----|------|--------|--------|
| T1 | Config table (routers × features × protocol) | new LaTeX from method text | gating-analysis + eval config |
| T2 | Temporal seed-level results (no-delta) | filter existing table | `formal-eval-1.0` |
| T3 | Temporal pairwise (global, gating) | existing focused rows | `formal-eval-1.0` |
| T4 | LOSO summary + win counts (no-delta) | filter existing | `formal-eval-1.0` |
| T5 | K-sweep purity/quality | existing | `gating-analysis-1.0` |
| T6 | Regime station composition + top features | existing | gating-analysis / regime-interpretation |
| T7 | (opt) ECE RMSE pairwise + station means | existing | `formal-eval-2.1-ece-v3` |
| F1 | Temporal multi-seed dot/bar + CI | NEW | `paper2-figures-1.0` notebook |
| F2 | LOSO per-station pairs | reuse PNG | `formal-eval-1.0` |
| F3 | Regime geographic map (WA) | NEW or reuse | gating geographic PNGs exist — pick/crop |
| F4 | Delta-robustness slope (test/val/none) | NEW, optional | `formal-eval-1.0` |
| F5 | (opt) ECE time-series line charts | reuse ≤5-line figs | `formal-eval-2.1-ece-v3` |

---

## 5. Claims ledger (excerpt — full ledger = separate file in works)

| # | Claim | Config | Number | Artifact |
|---|-------|--------|--------|----------|
| C1 | Two-regime > global temporal | V0_Full (0,0) vs Global_54 | 0.8118 vs 0.7798; Δ +0.032 p < 1e-12 | `formal-eval-1.0` |
| C2 | Sample-level significance | same | ΔR² +0.035 [0.019, 0.055] p = 0.0005 | `formal-eval-1.0` bootstrap |
| C3 | Two-regime > global LOSO | V0_Full (0,0) | 0.637 vs 0.580; 6/7 wins | `formal-eval-1.0` LOSO |
| C4 | Trained gating inferior temporal | (0,0) | 0.735 vs 0.812 | `formal-eval-1.0` |
| C5 | K=2 station-pure; K>2 fragments | Backbone54 | purity 1.000 / 0.833 / 0.695 | `gating-analysis-1.0` |
| C6 | Regimes = east vs west | composition table | Spokane + Sourdough = C1 | `gating-analysis-1.0` |
| C7 | OOS: no model transfers; 2-regime ≤ global | no-delta arms | mean R² < 0 all; Δ −0.17 (2/10) | `formal-eval-2.0` |
| C8 | ECE: models tie; R² invalid at low σ_y | no-delta | ΔRMSE +0.00025 ns | `formal-eval-2.1` |
| C9 | ECE daily means hide diurnal/event response | n/a | hourly range 0.035 vs daily step 0.0025 | `ece-input-diagnose-1.0` |

Every draft sentence must map to a ledger row; no orphan numbers.

---

## 6. Venue adaptability matrix

| Element | ML conf (6–8p) | Geo/AGU-style | Applied IEEE (AIIoT-like) |
|---------|----------------|---------------|---------------------------|
| §5 interpretability | short | full | medium |
| §7 ECE | optional/omit | include (deployment) | include (motivation) |
| Stats detail | protocol emphasis | metrics emphasis | protocol lighter |
| OOS limitation | 3 sentences | 1 short para | 1 short para |
| Related work | MoE-heavy | SM-heavy | balanced |

---

## 7. Works to do (ordered)

- **W0** Freeze claim scope: primary configs = no-delta only; router name paper-wide (recommend `Clustering_V0_Full_k2` or rename to `Clustering_k2` in text; Backbone54 noted ARI=1). *(decision, no code)*
- **W1** Write full claims ledger → `writeup2/claims-ledger.md` from `formal-eval-1.0` / `2.0` / `2.1` + `gating-analysis` + `ece-input-diagnose` READMEs (stdout-only numbers).
- **W2** Related-work search + BibTeX (venue chosen later; collect ~40 candidates across the four pillars, tag venue-fit).
- **W3** New figure notebook `notebooks/experiment/paper2-figures-1.0/` producing F1 (+F4 if kept) from existing CSVs; `nb execute --uv`; figures saved under the experiment dir; tables in writeup2 filled only from notebook stdout.
- **W4** ECE inclusion decision (PI): include §7 vs cut. Default plan includes it.
- **W5** Create `writeup2/` skeleton: this outline + `claims-ledger.md` + `figures/` + (later) `main.tex` or `main.md`; compile script only after venue pick.
- **W6** Draft §3–§6 (method + results) first — fully supported; then §1–2, §5, §8–10.
- **W7** PI review pass against ledger; then venue-specific trimming via matrix §6.
- **W8** Reproducibility gate before any submission: re-execute `formal-eval-1.0` report notebook + `paper2-figures` notebook; confirm every table row = stdout; `make lint` / `make test` if src touched (expected: no).

Not doing (this paper): station-count ablation, OOS full section, new training runs, edits to `writeup/` or `paper/`, archived notebooks.

---

## 8. Risks

- **R1:** LOSO non-significance misrepresented — mitigate: fixed wording in T4 caption.
- **R2:** "two regimes" read as universal — mitigate: §4 K-selection + §8 discussion.
- **R3:** ECE section distracts or invites reviewer attack — mitigate: W4 cut option; caveats written into the subsection, not buried.
- **R4:** Backbone selected on test — reviewers may challenge absolutes; mitigate: shared-backbone sentence in §3 + §6.4; never compare to literature absolutes without note.

---

## 9. Primary source map (where every number lives)

| Experiment dir | Role in paper |
|----------------|---------------|
| `notebooks/experiment/derived_8.4-formal-eval-1.0/` | Temporal + LOSO formal stats (core results) |
| `notebooks/experiment/derived_8.4-gating-analysis-1.0/` | K-sweep, purity, regime composition, separating features |
| `notebooks/experiment/derived_8.4-regime-interpretation-1.1/` | Method narrative, physical interpretation of regimes |
| `notebooks/experiment/derived_8.4-eval-1.3/` | LOSO protocol provenance, station difficulty (secondary) |
| `notebooks/experiment/derived_8.4-formal-eval-2.0/` | OOS numbers for Limitations only |
| `notebooks/experiment/derived_8.4-formal-eval-2.1-ece-v3/` | ECE transfer results (§7 if kept) |
| `notebooks/experiment/ece-input-diagnose-1.0/` | ECE sensor/benchmark validity diagnosis (§7 if kept) |
| `data/splits/derived_8.4/split_meta.json` | Split definition for §3 |
| `docs/plans/20260703-why-MoE.md` | Motivation background only (re-verify numbers before quoting) |
