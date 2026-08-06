# Overfitting symptoms in the temporal MLP models (derived_8.4-eval-mlp-1.2)

Status: observed on the `derived_8.4-eval-mlp-1.2` sweep (2-regime families,
test = 2023–2025). All numbers are produced by `analyze_overfitting.py` from the
saved sweep artifacts and rendered in `derived_8.4-eval-mlp-1.2.ipynb`
(Overfitting-Symptom Analysis section). No retraining needed to reproduce.

## Verdict

**Yes — the temporal MLPs are overfitting, but of a specific kind: they overfit
the *seen years* (train + validation periods), not the classic "memorize the
training set, fall back to the mean for unseen inputs" failure.** The models
still reach test R² 0.76–0.79, so they do generalize; the failure is that they
spend capacity on period-specific patterns that do not transfer to 2023–2025.

## The five symptoms

### 1. Train-fit is far better than any held-out fit

`aux2020` is the RMSE on the 2020 slice of *train* (n = 2,519) — data the model
has already seen — so it is a train-fit probe. Median over 2-regime MLP configs:

| family     | aux2020 (train-fit) | val | test | val/train |
|:-----------|--------------------:|----:|-----:|----------:|
| 2regime_96 | 0.030 | 0.051 | 0.051 | 1.7× |
| 2regime_54 | 0.029 | 0.060 | 0.048 | 2.0× |

The models fit the training years ~2× better than any unseen period. A
moderate gap is expected (temporal shift between 2017–20, 2021–22, 2023–25),
but combined with symptoms 2–4 it indicates a real memorization component.

### 2. Test error rises during training while train-fit keeps improving

Per-epoch curves of the 2-regime-96 val winner (`w512x512x512_d0.3_lr1e-3`,
cluster-0 specialist):

| epoch | train-fit (aux) | val | test |
|------:|----------------:|----:|-----:|
| 90    | 0.027 | 0.055 | **0.045 (test minimum)** |
| 260   | 0.018 | 0.053 | 0.050 (worse) |

Train-fit keeps dropping to epoch 260, val plateaus flat, but **test bottoms
out at epoch ~90 and then gets worse**. Early stopping on val cannot catch
this: val is flat because the model is fitting the 2021–22 period's
idiosyncrasies, which do not transfer to 2023–25.

### 3. Capacity buys in-sample fit, not transfer (the residual nets)

The 3 residual-net anchors have the *best* val RMSE **and** the best train-fit
(aux2020) of anything in the sweep, yet the *worst* test R²:

| config_id | n_params | val_rmse | aux_rmse | test_r2 |
|:----------|---------:|---------:|---------:|--------:|
| res_w512x256x128_d0.2 | 1.8M | 0.0472 | 0.0183 | 0.724 |
| res_w512x512x256_d0.2 | 3.3M | 0.0478 | 0.0186 | 0.728 |
| res_w1024x1024_d0.2   | 10.7M | 0.0476 | 0.0146 | 0.729 |

vs mid-size plain MLPs at 0.76–0.79 test R². This is exactly why 1.1's
val-selection kept picking the wrong models: Spearman(val_rmse, test_r2) ≈
**−0.22** for 2-regime-96 — the val ranking is anti-predictive of test for the
high-capacity class.

### 4. Adding the validation years to training actively hurts test

The trainval-retrain experiment (documented negative, see README): training on
train **+ val** (14,608 rows) instead of train only (9,803) makes cluster-0's
test RMSE degrade monotonically after ~epoch 22; even its best epoch (0.052) is
worse than the train-only sweep (0.046–0.049). The model has the capacity to
fit 2021–22 and does so at the expense of 2023–25. This is overfitting to the
training distribution, not distribution shift alone.

### 5. Systematic positive bias on test

MLP median test bias ≈ +0.021 (2-regime-96) vs XGBoost's +0.0065; bias² is
~10–17% of MSE. The MLPs systematically over-predict soil moisture on the test
years — a learned mapping slightly shifted for the 2023–25 distribution.

## What it is NOT

- **Not** the spatial model's "fall back to the mean for unseen inputs" —
  the temporal MLPs still score R² 0.76–0.79 on test.
- **Not** severe classic overfitting (validation loss exploding) — the
  train–val gap is moderate (1.7–2.0×) because part of it is genuine temporal
  distribution shift between the periods.

## What it means for closing the gap to XGBoost (0.815)

The gap is not a weak model class — it is capacity spent on period-specific
patterns. The fixes that already helped were overfitting/variance controls:
2-seed averaging, restricting selection to mid-size plain MLPs, and offline
ensembling (val top-10 → 0.786). Remaining headroom is regularization that
makes features robust across years: stronger weight decay, temporal
cross-validation for early stopping, and ensembling across routers/years.

## Reproduction

```bash
cd notebooks/experiment/derived_8.4-eval-mlp-1.2
python analyze_overfitting.py          # prints tables + writes overfitting_summary.csv
```
The same tables are rendered in the report notebook
(`derived_8.4-eval-mlp-1.2.ipynb`, Overfitting-Symptom Analysis section) and
copied into README.md.
