# MDR Wet-Regime Improvement Plan

**Goal:** Improve soil moisture performance in high-variance (wet) periods without hurting stable-regime performance.

---

## Core Insight (What We Learned)

- Gating did not meaningfully improve performance.
- Expert B could not extract strong signal in wet regime.
- Wet regime behaves more like heteroskedastic noise than a separate functional system.
- Therefore: weighting is more appropriate than routing.

We pivot from mixture-of-experts to variance-aware training.

---

# Phase 1 — Establish Clean Baseline (No Gating)

1. Use best global feature set.
2. Train single XGB model (no gate).
3. Report:
   - Overall R² (train / val / test)
   - Wet-only R²
   - Dry-only R²

Save these numbers clearly. This is the anchor.

---

# Phase 2 — Weighted Training

## Step 1 — Define Wet Mask (Train-Only Threshold)

- Use `G_rain_sum_7d` or `G_API`
- Define wet using train-only quantile

Example:

```python
WET_Q = 0.90
thr = train_df[GATE_COL].quantile(WET_Q)
```

---

## Step 2 — Create Sample Weights

```python
sample_weight = np.ones(len(train_df))
sample_weight[train_df["is_wet"] == 1] = W
```

Try:

- W = 1.5
- W = 2.0
- W = 3.0
- W = 4.0

---

## Step 3 — Train Weighted XGB

```python
model.fit(X_train, y_train, sample_weight=sample_weight)
```

---

## Step 4 — Evaluate

Report:

- Overall R²
- Wet-only R²
- Dry-only R²
- Worst-decile R² (optional diagnostic)

---

# What Success Looks Like

Target:

- +0.01 to +0.03 overall test R²
- Significant improvement in wet regime
- Minimal degradation in dry regime

If wet improves and dry barely moves, this is a win.

---

# Phase 3 — Optional Refinements (Only If Needed)

If weighting helps but saturates:

1. Try `reg:absoluteerror` objective.
2. Lower `max_depth` to reduce overfitting in wet noise.
3. Try log-transform of target.
4. Try proportional weighting instead of binary wet/dry.

Example:

```python
sample_weight = 1 + alpha * (G_rain_sum_7d / G_rain_sum_7d.max())
```

---

# What NOT To Do

- Do not reintroduce gating.
- Do not expand feature set.
- Do not redesign pipeline.
- Do not chase micro-gains before confirming weighting impact.

Stay focused and controlled.

---

# Final Deliverable

Produce a comparison table:

| Model            | Overall Test R² | Wet R² | Dry R² |
| ---------------- | --------------- | ------ | ------ |
| Baseline         | 0.72            | -1.26  | 0.75   |
| Weighted (W=2.0) | 0.74            | 0.15   | 0.73   |
| Weighted (W=3.0) | 0.75            | 0.28   | 0.70   |

This clearly shows whether the approach improves robustness in high-variance conditions.
