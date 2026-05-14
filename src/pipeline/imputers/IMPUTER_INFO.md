# Imputers — Quick README

Casual tour of the imputer code in this folder. This is meant as a quick, usable guide for newcomers and for future-you when you need to tweak behaviour.

Top-level idea: many small imputers try their best and then a voter combines them into one final guess plus a confidence score per timestamp.

Entry point

- `imputers.api.transform_with_ensemble(df, col, return_diag=False, diag_path=None, auto_validate=False)` — the function the rest of the pipeline calls.

How things fit together

- `imputer_utils._run_ensemble` creates the imputer instances, calls `VotingImputer.fit(...)`, then `VotingImputer.impute(...)`.
- `VotingImputer` expects each imputer to implement `fit(dates, values, aux_df)` and `impute(dates, values, aux_df)` and to return `(filled_series, confidence_series)`.

Core voting formula

- Each imputer i produces a value v_i and a confidence c_i (0..1).
- There is a base weight b_i configured in `config.yaml` under `imputer.base_weights`.
- Effective weight: w_i = b_i \* c_i
- Final value (weighted average):
  $$\hat v = \dfrac{\sum_i w_i v_i}{\sum_i w_i}$$
- Confidence is similarly combined: a weighted average of individual confidences.

Outlier suppression (what we do)

- We compute the median and MAD of candidate values at each timestamp. If |v_i - median| > outlier_factor \* MAD we downweight that imputer by a factor (0.1 by default).

Imputer summaries and formulas

- `base.py` — `BaseImputer` interface. Subclass this and implement `fit` / `impute`.

- `interpolation.py` — Linear time interpolation. Confidence decays with gap length:
  $$\text{conf} = e^{-\text{gap_days} / \tau}$$

- `fbfill.py` — Forward/backward fill for very short gaps and edges. Confidence uses nearest known-day distance:
  $$\text{conf} = e^{-\text{distance} / \tau}$$

- `smoothing.py` (RollingMeanImputer) — Centered rolling mean with window `w`. Confidence ≈ count / w where count is non-null count in the window.

- `spline_interpolation.py` — Smooth cubic spline (UnivariateSpline). Confidence is a function of curvature:
  $$\text{conf} \approx e^{- |\kappa| }$$
  where curvature is approximated by the absolute second derivative; extrapolated regions are penalized.

- `knn_temporal.py` — KNN in time (scikit-learn KNeighborsRegressor). Confidence is derived from neighbor distances:
  $$\text{conf} = \dfrac{1}{1 + \text{mean_distance}}$$

- `gaussian_regression.py` — Gaussian Process regression (RBF + white-noise). For each prediction we get mean μ and std σ; confidence is set to:
  $$\text{conf} = e^{-\sigma}$$

- `linear_model.py` — Linear regression on DOY encodings and optional cross-features. Confidence scales with how many clean samples were available (clipped at 1).

- `xgb_model.py` — XGBoost model trained on DOY encodings + cross-features. Produces predictions with a moderate default confidence (0.7) when filling.

- `climatology.py` and `seasonal_naive.py` — Day-of-year historical lookups. Very useful for long gaps; confidence is proportional to how many historical samples exist for that DOY.

- `voting.py` — The ensemble combiner. See the code for diagnostics (`VotingImputer.diagnostics`) — it reports per-imputer contributions and average confidence for missing timestamps.

Configuration

- `config.yaml` has an `imputer` section. Important knobs:
  - `base_weights`: per-imputer base weighting
  - per-imputer params such as `tau_days`, `window`, `min_known`, `length_scale`, etc.

Diagnostics and validation

- Call `transform_with_ensemble(df, col, return_diag=True)` to get a diagnostics dict. The voter also has `diagnostics(...)` to write out per-imputer contribution summaries.

Extending / adding a new imputer

1. Create a new module that subclasses `BaseImputer` and implement `fit` and `impute` (both should accept `dates, values, aux_df`).
2. Return `(filled_series, confidence_series)` from `impute` where both are aligned with the input `dates`.
3. Add the new imputer class to the list in `utils/imputer_utils._run_ensemble` so it gets instantiated and included in the voter.

Testing tips

- Unit-test a single imputer by feeding a small DataFrame with `date` and the target column and asserting reasonable fill and conf values.
- Use `transform_with_ensemble(..., return_diag=True)` to inspect per-imputer activity on a real sample.

Housekeeping

- If an imputer keeps failing or underperforming, just remove it from `_run_ensemble` — the voting system is robust and will adapt.
- Keep the code small and focused: each imputer should be easy to reason about and quick to fit
