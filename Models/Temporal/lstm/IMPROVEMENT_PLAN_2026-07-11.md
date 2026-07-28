# LSTM Temporal Model: Audit Findings + Performance Improvement Plan

## Context

The v23 LSTM experiments (soil-moisture 5cm nowcasting, 5 stations, temporal split 2017–20 train / 2021–22 val / 2023–25 test) sit at 5-seed ensemble test R²=0.792 (RMSE 0.0429, MAE 0.0312) vs the external XGBoost anchor 0.822. The audit found the methodology fundamentally sound (temporal split, train-only scaling, no target leakage, checkpoint-reproduction self-checks) but identified the real bottlenecks:

1. **Temporal distribution shift**, not capacity: persistent ~0.10–0.13 val→test R² gap; val and test rank configs in *opposite* order (window sweep, small-RNN sweep), so val-RMSE selection is actively harmful. The single `SMAP_x_year` drift feature is the most load-bearing feature on test (ablation: 0.79→0.566).
2. **Metric corruption**: `build_datasets_v2` (dataset.py:150-158) detects windows spanning Touchet's ~1437-day data gap (span > 2·seq_len) but still includes them → Touchet test R²=−1.32 pollutes pooled headline metrics. SourdoughGulch (0.336) is a second weak station, undiagnosed.
3. **Untapped free wins**: per-seed prediction CSVs for 4 seq_lens × multiple seeds + small-RNN checkpoints already exist on disk → heterogeneous ensembling needs zero retraining. Rolling per-station z-scoring was already falsified (v22: test 0.614) — do not revisit.

Decisions: implement everything including retraining; test both delta-target variants (persistence and SMAP-offset), report separately; train an in-repo tabular member AND accept external XGBoost prediction CSVs; stay on derived_8.0 this cycle (derived_9.0 upgrade = follow-up).

Goal: beat test R² 0.792 (honest, gap-excluded accounting), approach/exceed 0.822.

## Files

All work in `Models/Temporal/lstm/`. Shared infra to reuse: `train_v9.py` (BiLSTMAttn, `fit/apply_preprocessors`, `compute_metrics`, `set_seed`), `dataset.py` (`build_datasets_v2`), `eval_common.py` (per-station/year breakdowns, checkpoint self-check), `ensemble_eval.py` (seed-averaging), `train_v23_baseline.py` (5-seed protocol, to be de-duplicated).

## Phase 0 — Metric hygiene + free ensemble wins (no retraining)

**0.1 Gap-window exclusion.** Add `max_span_factor: float | None = 2.0` param to `build_datasets_v2`; when set, skip anomalous windows (keep diagnostics, add an `excluded` count). Add a post-hoc filter in `eval_common.py` that computes gap-crossing (station_id, date) targets from the raw split CSVs + seq_len, so *existing* prediction CSVs are re-scorable without re-inference. Headline convention: report pooled-all, pooled gap-excluded (new headline), and Touchet-excluded.

**0.2 Selection-signal fix.** Add a 2022-only-val report view in `eval_common.py`; tabulate 2022-val R² vs test R² across all existing configs (baseline, window sweep, small-RNN) to verify late-val ranks configs closer to test order. If confirmed, all later experiments early-stop/select on 2022-val RMSE.

**0.3 Heterogeneous ensemble.** Generalize `ensemble_eval.py` to average across config dirs, not just seeds: pool `outputs_v23_baseline/top25_seq30/seed*` + `window_sweep/seq5/seed*` (+ seq10), inner-join on (station_id, date). Try simple mean and inverse-val-RMSE weights (weights chosen on 2022-val).

**0.4 Cheap inference passes.** New `infer_checkpoints.py`: run existing `gru_h32`/`rnn_h48` checkpoints to produce standard `{val,test}_preds.csv` (they only have `best_model.pt`); add to the 0.3 pool. Also add a `--refit-scaler-on-test` flag (unsupervised domain-adaptation-lite; cheap to measure, uncertain sign).

**0.5 SourdoughGulch diagnosis.** Per-station×per-year residual breakdown from existing test preds; check whether its 0.336 is level-shift (fixable by Phase 2) or data-quality.

## Phase 1 — Shared runner + cheap retrains (5 seeds each, protocol locked)

**1.0 Runner refactor.** Fold `train_v23_baseline.train_one_seed`'s ~120 duplicated lines back into a parameterized shared runner (`train_v24.py` or extended `train_v20.train_model`): config-driven feature set, seq_len, model factory, target mode, selection metric (2022-val), `build_datasets_v2(max_span_factor=2.0)` default, 5-seed + ensemble reporting via `ensemble_eval`.

**1.1 Short-window sweep.** seq_len ∈ {2, 3, 5} (+10 rerun) — test R² improves monotonically as windows shorten (seq5 0.773 vs seq30 0.758 single-seed); v2 builder makes comparisons coverage-clean. Feed results into the 0.3 ensemble pool.

**1.2 Feature variants.** (a) top25 minus negligible families (SAR, optical_s2, SMAP-rollstd); (b) top25 + in-script drift terms via a small `add_drift_features(df)` helper in `dataset.py` (`LST_x_year`, `precip_x_year` — only `API_x_year`/`SMAP_x_year` exist as columns), computed before scaling.

## Phase 2 — Distribution-shift mitigation (the big modeling lever)

**2.1 Delta-target modeling (both variants, reported separately).** Add `target_mode ∈ {level, delta_persistence, delta_smap}` to the dataset build: emit `y_prev` (previous-day in-situ) and the SMAP reference alongside targets in `SoilMoistureDatasetWithMeta`; skip delta targets across gaps (reuses 0.1 gate). Reconstruction in `eval_common` before `evaluate()`: `ŷ_t = y_{t−1} + Δ̂` (persistence; assumes yesterday's reading available) and `ŷ_t = SMAP_t + Δ̂` (works without in-situ history). Persistence residuals are far more stationary across the 2023–25 shift — strongest single candidate (+0.02–0.05 plausible), directly attacks the Touchet/SourdoughGulch level-shift errors.

**2.2 Fine-tune on val years.** `finetune_on_val` flag in the runner: after selection, continue on train+val at lr=1e-4 for a fixed small budget (~5 epochs, no early stop — nothing to validate on). Moves training 2 years closer to test.

## Phase 3 — Tabular member + hybrid blend

**3.1 In-repo tabular member.** New `train_tabular_member.py`: gradient boosting (XGBoost if installed, else `HistGradientBoostingRegressor`) on the same top25 tabular rows, train-only preprocessing via `train_v9.fit_preprocessors`, standard prediction-CSV output.

**3.2 Blend.** The 0.3 ensemble machinery blends LSTM ensemble + tabular member, weight chosen on 2022-val; accepts external prediction CSVs (drop-in slot for Jakob's XGBoost when exported). Decorrelated error structures make this the most reliable route past 0.822.

## Follow-up (out of scope this cycle)
- Migrate to `derived_9.0` (30 stations, ~29k train rows; schema-compatible) — likely the highest ceiling; headline on the 5-station subset for comparability.

## Verification
- After 0.1: re-score existing baseline CSVs with/without gap exclusion; confirm Touchet's anomalous-window count matches `anomalous_window_counts` diagnostics and pooled R² moves ~0.79→0.80+.
- `eval_common`'s existing checkpoint-reproduction self-check (`_reproduce_known_checkpoint`, test R²=0.7900784 ± 1e-6) must keep passing wherever eval paths are touched with default (no-exclusion) settings.
- Every retraining experiment: 5 seeds, report mean±std AND ensemble, gap-excluded pooled test as headline + per-station/per-year breakdowns; compare against the re-scored 0.79x baseline under identical accounting.
- Delta-target modes: sanity-check reconstruction on train (a persistence-only baseline, Δ̂=0, should itself score high R² — report it as the floor every delta model must beat).
- Final deliverable: results table (baseline → each phase) + updated standup-style writeup.
