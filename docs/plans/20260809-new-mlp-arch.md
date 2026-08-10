# Plan: `derived_8.4-eval-mlp-2.0` — optimized MLP architecture to break the ceiling

## 1. Context: what the two source experiments established

### From `derived_8.4-eval-mlp-1.3` (temporal, 1.75 H100-h)
- **The plain-2-regime-MLP ceiling is confirmed.** Val-selected winners are bit-identical to 1.2: 2regime_96 `w512x512x512_d0.3_lr1e-3` test R² **0.761**, 2regime_54 `w512x512x512_d0.3_huber0.1` **0.765**. Test-best reference 0.789 (`w384x384_d0.3_gelu`, 54-family). XGBoost 2-regime winner: **0.815**. Gap ~0.03 is model-class ceiling, not selection.
- **Documented negatives (do not re-spend GPU):** val-fit per-cluster affine calibration does not transfer to test (54-family 0/12 helped); no honest early-stopping rule beats patience-60 (oracle headroom ~0.004 RMSE is unreachable); EMA (decay 0.999 *per optimizer step*, ~14 steps/epoch) is a *trainer bug* — it lags ~70 epochs behind the fast-moving head and is catastrophically bad; trainval retrain poisons cluster-0; aux2020 measures train fit (failed selection signal); residual MLPs are val-overfitters; FT-Transformer fails; center_target / batch 256 hurt.
- **What actually moved the needle (positive):** 54-family good-capacity region — `w448x448_d0.3_gelu` (test R² 0.7809, bias +0.001), `w384x384x256_d0.3_gelu` (0.7695); **54-family val top-10 avg 0.7825**; mixup (54-family `w384x384_d0.3_mixup0.2` 0.7784); huber δ 0.05–0.1 dominates val tops; **96-family bias² ≈ 10–17% of MSE, and it scales with capacity** (<200k params → bias 0.0049, test R² 0.7834 for `w256x256_d0.5`; 1M+ → bias 0.0203) — capacity control is the demonstrably working debias lever.
- **Overfitting kind (from 1.2's `overfitting_analysis.md`):** not classic memorization — the MLPs "spend capacity on period-specific patterns" of the seen years (train-fit ~2× better than any held-out period; test bottoms at ~ep 90 while val stays flat; residual nets top in-sample but worst on test).

### From `derived_8.4-eval-2.0` (LOSO spatial generalization of the same MLPs)
- The **96-family transfers spatially better** (LOSO pooled 0.668 ≈ XGBoost 2-regime 0.689) while the 54-family transfers worse (0.48–0.59) — the 96-pool carries extrapolation power. Temporal ranking ≠ spatial ranking (54-family best temporally, 96-family best spatially).
- Hardest stations are dynamic-regime outliers (Quinault 0.25, SourdoughGulch 0.41); "twin → easy" holds for MLPs. 2025 per-station-year R² is unstable (data artifact).
- Implication for 2.0: keep the 96-pool in the model (mixed family), avoid station-id/embedding features (would leak into LOSO), and don't let the temporal sweep overfit station identity.

### The per-cluster fact that motivates the new family
Per-cluster test R² (mlp-1.3, 2-seed): **c0 on 96-pool = 0.754 vs c0 on 54-backbone = 0.737; c1 on 54+10 = 0.831 vs c1 on 96-pool = 0.776.** Each feature set is better on one cluster. A mixed allocation (c0 ← 96-pool, c1 ← 54+10) is a cheap, evidence-driven architecture change never tried.

## 2. Objective & success criteria

**Objective:** `derived_8.4-eval-mlp-2.0` — an optimized MLP architecture + training, temporal protocol only (no LOSO during the sweep, per request), ~2 H100-hours (run later when GPU is available; this plan + scaffolding now).

Success criteria (honest, val-based):
1. **Val-selected winner (2-seed) test R² > 0.79** (past the 0.789 test-best ceiling *with honest selection*), and/or **val top-5/10 ensemble > 0.80**.
2. **Debias:** for every family, median test **bias²/MSE < 5%** (from 10–17% for the 96-family).
3. **Mixed family per-cluster:** c1 ≥ 0.83 (holds the 54+10 specialist's strength), c0 ≥ 0.76 (beats both current c0 specialists).
4. All documented negatives honored (no calibration, no retrain-on-trainval, no broken per-step EMA, patience-60 kept, aux2020 diagnostic-only).
5. Full reproducibility: `nb execute --uv` from `notebooks/`; README tables from executed-notebook stdout.

## 3. The optimized architecture (what's new in 2.0)

### 3.1 `FeatureGroupedMLP` (`architecture: fg`) — the headline change
The 54/96/64 features are heterogeneous: SMAP soil-moisture series, Sentinel-2 optical bands, NDVI/NDMI vegetation indices, SAR backscatter ratios, LST/thermal, meteorology/API, static geo·BioClim·soil, and temporal harmonics (DOY/year). The plain MLP mixes them in one dense stack, which is exactly the setup that "spends capacity on period-specific interactions".

- **Per-group tower:** each semantic group gets its own small MLP (Linear → Norm → act → Dropout, 2 layers, width ~128–256) producing a group embedding.
- **Fusion:** concat group embeddings → 2–3 layer fusion MLP (same width family as the 1.3 anchors, e.g. 384/448/512) → huber/mse head.
- Groups defined in `mlp20/feature_groups.py` by an explicit, documented prefix→group table (`SMAP_*`, `s2_*`, `F_*`/`C_lag_*` veg, `E_*`/`rough_*` SAR, `LST_*` thermal, `G_*`/`precip_mm` meteo, `J_*`/`latitude`/`elev`/`slope`/`lia_*`/`lc_code` static, `D_*`/`DOY`/`sin_year`/`cos_year`/`SMAP_x_year` temporal), with a validation that every feature lands in exactly one group (throw on overlap/gap). The grouping is generated by code, not ad-hoc — auditable.
- Optional per-group dropout on the concatenation (a cheap "group-dropping" regularizer analogous to mixup at the group level).

### 3.2 PLR encoding (`architecture: plr`) — Gorishniy et al. 2022-style
Piecewise-linear encoding of each input feature (`x·b0+c0 + Σ_k ReLU(α_k(x−β_k))`, k≈8 per feature, ~800 extra params on 96 features) before the first dense layer. Well-established cheap tabular upgrade for MLPs; never tried in this project.

### 3.3 SWA — proper weight averaging (replaces the broken EMA)
EMA was a **trainer bug**, never a valid test of averaging. Add a `swa` knob:
- After `swa_start_frac` (default 0.6) of `max_epochs`, maintain a running average of the live weights **per epoch** (not per step); also average BN buffers, plus an optional one-epoch BN-recalibration forward pass over train when SWA weights are finalized.
- SWA snapshot's val RMSE is tracked; best-SWA-val epoch decides the deployed snapshot; live-model val with patience-60 still governs early stopping (unchanged). Selection stays on val — honest.
- Targets exactly the documented failure mode: "test min at ~ep 90, val flat to 260" — SWA smooths the flat-val region instead of picking a single late epoch.

### 3.4 New family `2regime_mixed` (feature allocation per cluster)
- c0 specialist: 96-pool features (best c0 R² 0.754).
- c1 specialist: 54-backbone + eval-1.1's 10 delta features (best c1 R² 0.831).
- Same router (V0Full KMeans k=2, seed 42, fit on trainval) and same specialist-training protocol as 1.3. Only the feature allocation differs.

### 3.5 Kept (proven) ingredients
huber δ {0.05, 0.1}; gelu/silu; dropout 0.3–0.5; AdamW + warmup 5% + cosine; grad clip 1.0; patience-60; 2-seed {42, 7}; aux2020 diagnostic; median-impute + StandardScaler + clip [−5, 5] fit on train; target in original units; `cudnn.deterministic=True`. Residual-MLP v2 and FT re-run as **reference rows only** (documented failures; do not enter the winner pool).

### 3.6 Explicitly out of scope
- LOSO during the sweep (per request). A post-hoc LOSO validation of the final winner (eval-2.0-style) may be a separate follow-up run if the team wants it — noted, not scheduled.
- New routers (KMeans V0 stays for XGBoost comparability), station embeddings (LOSO leakage), new feature selection (the 96-pool/54-backbone/deltas are pinned from feature-selection-2.0 / eval-1.1).

## 4. Protocol (data_version 6, temporal only)

- Train on official train (2017–2020, n=9,803); early-stop/select on official val (2021–2022, n=4,805); evaluate on untouched test (2023–2025, n=6,620). aux2020 (n=2,519) diagnostic only.
- `data_version: 6` (v5 = mlp-1.3) so stale artifacts are invalidated on resume.
- Phase 1: seed 42 for every (family × config). Phase 2: seed 7 for the top-8/family by **2-seed-able val RMSE** (phase-2 metric `val_rmse`, as in 1.3 — aux2020 stays diagnostic).
- Winner pool restricted to `mlp`/`fg`/`plr` architectures (residual/FT reference-only).
- Champion step (still temporal): 5-seed {42,7,123,2024,999} runs of the top 1–2 winners; offline val top-k config ensembles (k=3/5/10); **cross-family ensembles (54+96+mixed)** — never tried; the families are complementary (54: near-unbiased; 96: extrapolation; mixed: per-cluster-optimal).

## 5. Sweep design

3 families × 10 configs = **30 phase-1 jobs; phase 2 adds 24 second-seed jobs (top-8/family); total ≈ 54 jobs ≈ 1.4–1.5 GPU-h** (1.3: 38 jobs ≈ 0.87 GPU-h at ~83 s/job; the grouped towers add a little per-epoch cost — `fg_tower_width` defaults to 128 to bound fusion cost). Champion step (top-1 winner per family, 3 extra seeds) ≈ +0.3 GPU-h. Total ≈ 1.7–1.8 of the 2.0 h budget, leaving margin.

| family | configs (10 each) | rationale |
|---|---|---|
| `2regime_54` | anchors: `w512x512x512_d0.3_huber0.1`, `w384x384_d0.3_gelu`, `w448x448_d0.3_gelu`; training: `w384x384_d0.3_mixup0.4`, `w512x512x512_d0.3_huber0.1_swa`, `w448x448_d0.3_gelu_swa`; fg: `fg_w384x384_d0.3`, `fg_w384x384_d0.3_swa`, `fg_w512x512_d0.3_huber0.1_swa`; plr: `plr_w384x384_d0.3_gelu_swa` | re-baseline under v6; mixup/SWA on the good-capacity region; grouped/PLR against plain |
| `2regime_96` | anchors: `w512x512x512_d0.3_lr1e-3`, `w256x256_d0.5`, `w512x512_d0.35_gelu`; debias: `w256x256_d0.5_swa`, `w512x512x512_d0.3_lr1e-3_swa`, `w512x512x512_d0.3_huber0.1_swa`; fg: `fg_w256x256_d0.4_swa`, `fg_w384x384_d0.3_huber0.1`, `fg_w384x384_d0.3_huber0.1_swa`; plr: `plr_w256x256_d0.4_swa` | bias-targeting: small nets + higher dropout + SWA (the proven debias direction) |
| `2regime_mixed` (NEW) | `w512x512x512_d0.3_huber0.1`, `w384x384_d0.3_gelu`, `w448x448_d0.3_gelu`, `w512x512x512_d0.3_huber0.1_swa`, `w448x448_d0.3_gelu_swa`, `w384x384_d0.3_mixup0.4_huber0.1`, `fg_w384x384_d0.3_gelu_swa`, `fg_w512x512_d0.3_huber0.1_swa`, `plr_w384x384_d0.3_gelu`, `w256x256_d0.4_swa` | c0=96-pool / c1=54+10; same config space as 54-family for direct comparison |

Defaults (unless overridden): max_epochs 400, patience 60, checkpoint_every 20, grad_clip 1.0, norm bn, activation silu, dropout 0.3, lr 3e-4, wd 1e-4, batch 512, loss mse, warmup_frac 0.05, swa off.

## 6. Deliverables — new experiment directory `notebooks/experiment/derived_8.4-eval-mlp-2.0/`

Scaffolded from `mlp13/` + `run_mlp_sweep/eval` (versioned-dir convention; nothing existing is modified):

- `config.yaml` — protocol v6, 3 families (incl. `2regime_mixed` with its c0/c1 feature allocation), 30 configs, defaults with `swa` knob.
- `mlp20/` package:
  - `model.py` — add `FeatureGroupedMLP` and `PLRRegressor` (extend `build_model`; keep `MLPRegressor`/`ResidualMLP`/`FTTransformer`).
  - `feature_groups.py` — documented prefix→group table + validation (every feature covered exactly once).
  - `trainer.py` — `swa` knob (per-epoch averaging, buffer handling, best-SWA-val selection, patience-60 unchanged); everything else 1.3-identical.
  - `data.py`, `plots.py` — adapted from mlp13.
- `run_mlp_sweep.py` / `run_mlp_worker.py` — mixed-family feature mapping (`family_features()`), data_version 6, phase-2 `val_rmse` top-8/family.
- `run_mlp_eval.py` — leaderboard vs XGBoost (0.815) + mlp-1.3 + eval-2.0 reference rows; selection diagnostics (Spearman val↔test); per-regime; per-year; station-year table; loss curves; sweep summary; timing.
- Analyses (all offline from saved artifacts): `analyze_bias.py` (**headline: bias²/MSE share per config/family**), `analyze_overfitting.py`, `analyze_extrapolation.py`, `analyze_stopping.py` (replay patience-60 vs SWA-val on the new curves), `generate_station_year_table.py`.
- `derived_8.4-eval-mlp-2.0.ipynb` (report notebook) + `README.md` (tables from executed-notebook stdout).
- Optional: `tests/test_mlp20_models.py` (forward/n_params smoke for `fg`/`plr`).

## 7. Execution timeline (GPU later; scaffolding at approval)

1. **Scaffold** (no GPU): copy mlp13 → mlp20, implement `FeatureGroupedMLP`/`PLR`/SWA/mixed family, `--smoke` test (3-epoch cap, data_version −1), config. Validate grouped-feature coverage and that `w512x512x512_d0.3_huber0.1` under v6 reproduces the 1.3 numbers bit-identically (stack check, as eval-2.0 did).
2. **Sweep** (when the H100 is available): `uv run --no-sync python run_mlp_sweep.py --resume` (phase 1 + phase 2, 8 workers) — monitor via logs; trims available (`--only`, `--families`, `--phase2-top-n`).
3. **Champion step:** 5-seed runs of the top 1–2 val winners; offline val top-k + cross-family ensembles.
4. **Offline analyses** (bias, overfitting, extrapolation, stopping replay) — zero GPU.
5. **Report:** `nb execute derived_8.4-eval-mlp-2.0.ipynb --uv` from `notebooks/`; populate README strictly from stdout; commit.
6. **Optional follow-up (not scheduled):** LOSO validation of the final winner (eval-2.0-style), per the note that LOSO is out of the sweep.

## 8. Risks & mitigations

- **SWA underperforms / BN complications** — SWA is only 1/3 of the budget; anchors stay; SWA is deployed only if it wins on val; BN handled via buffer averaging + optional recalibration pass.
- **Grouped towers don't beat plain MLP** (grouping assumption wrong) — grouped configs are a minority; plain-MLP anchors remain; the grouping table is explicit so a wrong group is a fixable config, not a mystery.
- **Mixed family doesn't realize the per-cluster gain** (specialists share the router) — cheap to test (10 configs); per-cluster evidence is strong (0.754 vs 0.737 c0; 0.831 vs 0.776 c1).
- **Budget overrun** — resumable sweep, `--phase2-top-n`/`--only` trims, monitor `timing_log.json`; if tight, reduce champion 5-seed to the single top winner.
- **Selection noise** (val↔test Spearman ≈ 0) — unchanged mitigation: 2-seed val averaging, winner pool restricted to non-reference architectures, top-k ensembles reported separately from single-model claims.

## 9. Reproducibility checklist (AGENTS.md rules)

- All constants generated by code committed in the experiment dir (grouping table, sweep configs, seeds) — no `/scratch` scratch scripts, no inline `python -c` for analysis.
- Deterministic seeds {42, 7} (+ {123, 2024, 999} for champion ensembles); `cudnn.deterministic=True`; anchors must reproduce mlp-1.3 bit-identically under v6 (stack check before the sweep).
- Preprocessing identical to 1.3 (median impute → StandardScaler → clip, fit on train only; target in original units) — no data or target changes.
- Report notebook executed with `nb execute --uv` from `notebooks/`; README tables copied from stdout; figures generated only by notebook/experiment scripts.
- Config paths relative to project root; no hardcoded absolute paths.