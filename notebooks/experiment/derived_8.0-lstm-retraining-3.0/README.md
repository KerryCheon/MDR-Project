# Experiment: `derived_8.0-lstm-retraining-3.0` — Full LSTM Retraining from Scratch

## Objective

The previous experiments (`lstm-version-comparison-1.0` and `2.0`) **reused checkpoints** from sweep-3.0, which were trained with a different training pipeline than optimization-2.0. This resulted in v21 scoring **0.7376** instead of the expected **0.8340**.

This experiment **retrains all 15 LSTM versions from scratch** using the identical infrastructure:
- **Training Code**: sweep-3.0's `train_candidate()` with full training (max_epochs=300, not checkpoint cache)
- **Feature Sets**: Each version's native feature set (Jakob 38 + version-specific extras)
- **Evaluation**: 2-regime clustering XGBoost model with frozen config
- **Goal**: Recover the true performance of each LSTM version when trained with proper regularization

## Key Differences from Comparison Experiments

| Aspect | Comparison 1.0/2.0 | Retraining 3.0 |
|--------|-------------------|----------------|
| **Checkpoints** | Reused from sweep-3.0 (cached) | Trained from scratch |
| **max_epochs** | None (checkpoint read-only) | 300 (full training) |
| **Expected v21 R²** | 0.7376 (degraded) | 0.8340 (expected recovery) |
| **Duration** | ~30 min (just representation extraction) | **12-24 hours** (full retraining) |
| **GPU Memory** | Low (inference only) | High (training + backprop) |

## Architecture & Process

### Phase 1: Baseline Tabular-Only Model
- Train Clustering_V0_Full_k2 on 54-feature backbone only (no LSTM features)
- Expected R² ≈ 0.8301 (reference for LSTM contribution)

### Phase 2: Retrain Each LSTM Version (v7-v23, excluding v18/v19)
For each of 15 candidates:
1. **Train**: `sweep_core.train_candidate(..., max_epochs=300)` — full training loop with ReduceLROnPlateau scheduler
2. **Extract**: Generic forward pre-hook on final linear layer → representation arrays (train/val/test)
3. **PCA**: Reduce to 32 components (seed=42, fit on train only)
4. **Evaluate**: Load hybrid data + fit Clustering_V0_Full_k2 with exact config from optimization-2.0

### Phase 3: Compare Performance
- Generate leaderboard sorted by test R² (pooled 2023–2025)
- Highlight v21's recovered performance
- Identify best-performing LSTM version when properly trained

## Files & Outputs

### Input
- `config.yaml`: Copied from optimization-2.0 (54 backbone, router features, XGBoost params, c1 additions)
- `data/splits/derived_8.0/`: Train/val/test CSV splits (unchanged from prior experiments)

### Output
- `artifacts/baseline/ctx_{train,val,test}.npy`: Dummy arrays for baseline evaluation
- `artifacts/<version>/ctx_{train,val,test}.npy`: PCA-32 reduced representations per LSTM version
- `artifacts/summary_records.csv`: Full leaderboard with all metrics
- `artifacts/lstm_metadata.json`: Per-version training summary (LSTM val R², repr dims, variance explained)
- `models/<version>/lstm_seed42.pt`: Trained checkpoints saved during retraining
- This `README.md`: Experiment summary (auto-generated after completion)

## Expected Results

### v21 (Union-Feature BiLSTM)
- **Previous (cache)**: R² = 0.7376 ❌ (9.3% regression)
- **Expected (retrained)**: R² ≈ 0.8340 ✅ (recovered)
- **Why**: Original training used optimization-2.0's exact hyperparameters (LR, dropout, scheduler)

### Other Versions
- v16 (BiLSTM + Engineered Lags) likely remains strong (~0.824+)
- Smaller models (v7, v12) may improve with proper training
- Complex models (v20, v22) may benefit from full training cycle

### Baseline
- Tabular-only (54 features) should remain ~0.8301 (provides LSTM contribution margin)

## Running the Experiment

```bash
cd /Users/kerrycheon/repos/Work/MDR-Project/notebooks/experiment/derived_8.0-lstm-retraining-3.0
uv run python3 run_retrain.py
```

**Estimated Duration**: 12–24 hours (15 versions × ~45–90 min per retraining + evaluation)

**Checkpoints**: Progress printed every epoch; can monitor with:
```bash
tail -f notebooks/experiment/derived_8.0-lstm-retraining-3.0/run_retrain.log
```

## Critical Notes

1. **GPU Memory**: Full training is resource-intensive. Monitor with `nvidia-smi` or `top`.
2. **Early Stopping**: Each version trains for up to 300 epochs with ReduceLROnPlateau + patience-based early stopping.
3. **Determinism**: All seeds set to 42 for reproducibility.
4. **No Modification to Existing Experiments**: This experiment:
   - Uses sweep-3.0 and optimization-2.0 as read-only references
   - Saves outputs to its own `artifacts/` and `models/` directories
   - Does NOT modify any cached checkpoints

## Comparison to Optimization-2.0

| Aspect | Optimization-2.0 | Retraining-3.0 |
|--------|------------------|----------------|
| **Scope** | V21 only | All 15 LSTM versions |
| **Representations** | 3 types (ctx, hh, hp) at 4 PCA levels | 1 type (ctx) at PCA-32 only |
| **Hybrid Models** | 2 strategies × 12 variants = 24 models | 1 strategy × 15 versions = 15 models |
| **Evaluation** | Global + Clustering, separate XGBoost per model | Clustering only, one XGBoost per version |

---

**Generated**: 2026-08-17
**Status**: Pending execution
