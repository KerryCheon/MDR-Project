# Handoff: LSTM v21 Performance Recovery & Multi-Version Retraining

**Date Created**: 2026-08-19  
**Status**: Experiment In Progress (Phase 2: Full Retraining)  
**Primary Goal**: Recover v21 from R²=0.7376 to expected R²=0.8340 by retraining from scratch

---

## Executive Summary

### The Problem
The v21 BiLSTM+Attention model (union-feature variant, 58 input features, seq_len=30) showed **dramatic performance degradation**:
- **Expected**: R² = 0.8340 (from optimization-2.0 report)
- **Actual (cached)**: R² = 0.7376 (from comparison experiments)
- **Regression**: 9.3% loss

### Root Cause Identified
Cached checkpoints in sweep-3.0 were trained with a **different training pipeline** than optimization-2.0:
- Different learning rate schedules
- Different dropout/regularization
- Different convergence criteria
- Same data, different weights → degraded representation quality

### Current Solution
**Retrain all 15 LSTM versions from scratch** using sweep-3.0's training infrastructure (not cached checkpoints), with identical frozen evaluation config from optimization-2.0.

### Expected Outcome
- v21: R² → 0.8340 (recovered)
- v16: R² ≈ 0.824+ (engineered lags variant)
- Baseline: R² ≈ 0.8301 (tabular-only reference)
- **Experiment duration**: 12–24 hours

---

## Experiment Structure

### File Locations

**Primary Experiment** (ACTIVE):
```
notebooks/experiment/derived_8.0-lstm-retraining-3.0/
├── run_retrain.py                    # Main orchestration script
├── config.yaml                       # Frozen hybrid model config (copied from opt-2.0)
├── README.md                         # Detailed experiment design
├── run_retrain.log                   # Active execution log
├── run_retrain.pid                   # Process ID (for monitoring)
├── artifacts/
│   ├── baseline/ctx_{train,val,test}.npy  # Dummy arrays for baseline eval
│   ├── v7/ctx_{train,val,test}.npy        # PCA-32 representations per version
│   ├── v8/...
│   ├── ...v23/
│   ├── summary_records.csv           # Final leaderboard (when complete)
│   └── lstm_metadata.json            # Training metrics per version
├── models/                           # (created during training)
└── training_runs/
    └── <version>/models/<version>/   # Retrained checkpoints (seed42.pt)
```

**Reference Experiments** (for comparison, READ-ONLY):
```
notebooks/experiment/derived_8.0-optimization-2.0/          # Original 0.834 model
notebooks/experiment/derived_8.0-hybrid-lstm-sweep-3.0/     # Cached checkpoints
notebooks/experiment/derived_8.0-lstm-version-comparison-1.0/ # Cache-based (deprecated)
notebooks/experiment/derived_8.0-lstm-version-comparison-2.0/ # Cache-based (deprecated)
```

**Data** (shared across all experiments):
```
data/splits/derived_8.0/
├── train.csv   # Training set
├── val.csv     # Validation set
└── test.csv    # Test set (2023-2025 pooled)
```

---

## How Retraining Works

### Script Logic (`run_retrain.py`)

1. **Load Configuration**
   - Import sweep-3.0's `build_candidates()` → 15 LSTM versions (v7-v23, skip v18/v19)
   - Import optimization-2.0's `compute_c1_gain_additions()` → cluster-1 feature additions
   - Load frozen config: 54-feature backbone, XGBoost hyperparameters, router setup

2. **Baseline Evaluation** (tabular-only)
   - Train 2-regime clustering XGBoost on 54 features only
   - No LSTM representation
   - Expected R² ≈ 0.8301
   - Provides reference point for LSTM contribution

3. **For Each LSTM Version** (v7, v8, ..., v23):
   ```python
   # Key: max_epochs=300 forces retraining (not checkpoint cache)
   run = sweep_core.train_candidate(
       candidate,
       seed=42,
       config=sweep_config,
       splits=splits,
       output_dir=fresh_output_dir,  # No pre-existing checkpoints
       include_test=True,
       max_epochs=300,  # Full training, ReduceLROnPlateau scheduler
   )
   
   # Extract representations via generic forward pre-hook
   train_repr, val_repr, test_repr = run.{train,validation,test}_representations
   
   # PCA-32 reduce (fit on train only)
   pca = PCA(n_components=32, random_state=42, svd_solver="auto")
   train_repr_pca = pca.fit_transform(train_repr)
   val_repr_pca = pca.transform(val_repr)
   test_repr_pca = pca.transform(test_repr)
   
   # Evaluate with 2-regime clustering model
   data_hybrid = load_hybrid_experiment_data(...)
   evaluator = HybridStrategyEvaluator(...)
   result = evaluator.fit_and_evaluate(
       global_features=data_hybrid.hybrid_features,
       cluster_additions={"0": [], "1": c1_additions},
   )
   ```

4. **Output**
   - `summary_records.csv`: Leaderboard with all metrics
   - `lstm_metadata.json`: Training details per version
   - `artifacts/<version>/ctx_{train,val,test}.npy`: Representation arrays
   - `training_runs/<version>/models/<version>/seed42.pt`: Retrained checkpoint

---

## Monitoring Progress

### Check Experiment Status
```bash
# View main log
tail -100 notebooks/experiment/derived_8.0-lstm-retraining-3.0/run_retrain.log

# Check if training has started (should see epoch outputs like "Ep  1", "Ep  2", ...)
grep "Ep  " notebooks/experiment/derived_8.0-lstm-retraining-3.0/run_retrain.log | head -20

# Check process status
ps aux | grep "run_retrain.py" | grep -v grep

# Monitor GPU memory (if GPU available)
nvidia-smi -l 1  # Update every 1 second
```

### What to Expect

**Phase 1: Baseline (5–10 min)**
```
===========================================================================
Tabular-only baseline (no LSTM features)
===========================================================================
[Data] Backbone=54
[Baseline] R²=0.83010
```

**Phase 2: LSTM Retraining (12–24 hours, 15 versions × 45–90 min each)**
```
[1/15] [v7] two-layer LSTM, last state
  [LSTM] validation R²=0.80787, repr_dim=48
  [PCA] 48->32 comps, var=1.0000
  [2-Regime] Test R²=0.78409

[2/15] [v8] feature-attention LSTM
  ...training epochs appear here...
  [LSTM] validation R²=0.70171, repr_dim=64
  [PCA] 64->32 comps, var=0.9994
  [2-Regime] Test R²=0.81964
```

**Phase 3: Final Leaderboard (appears after all versions complete)**
```
===========================================================================
FINAL LEADERBOARD (derived_8.0-lstm-retraining-3.0 — RETRAINED)
===========================================================================
                                                model_name  pooled_r2  ...
Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10) [No LSTM]   0.830097
v16 BiLSTM attention with engineered causal lags (2-Regime, PCA-32, RETRAINED)  0.823938
...
v21 union-feature BiLSTM attention used by the 0.834 hybrid (2-Regime, PCA-32, RETRAINED)  0.XXXXXX  <-- KEY RESULT
```

---

## Key Configuration (Frozen Across All Experiments)

### Hybrid Model Architecture
```yaml
Strategy: Clustering_V0_Full_k2 (2-regime)

Shared Backbone:
  Features: 54 static tabular
  Examples: precip_mm, s2_b4, s2_b8, SMAP_sm_pm_interp, DOY, API, LST, NDVI, aspect, ...

LSTM Representation:
  Input: 58 features (Jakob 38 + V9-unique 20)
  Seq_len: 30 timesteps
  Hidden: 80 dims (BiLSTM+Attn)
  Output: 80-dim → PCA-32
  Extraction: Forward pre-hook on final linear layer (head_hidden)

Router (determines regime):
  Features: V0 OVERALL_SELECTED_FEATURES (from dataset_metadata.py)
  Method: KMeans(n_clusters=2, random_state=42)
  Independent: Uses only tabular features (not LSTM)

Cluster 0 (Low-stress):
  Features: 54 backbone + 0 additions = 54
  Model: XGBRegressor(n_estimators=2500, lr=0.005, max_depth=9, ...)

Cluster 1 (High-stress):
  Features: 54 backbone + 10 additions = 64
  Model: XGBRegressor(n_estimators=2500, lr=0.005, max_depth=9, ...)

Cluster 1 Additions:
  - V_rollmin_F_NDMI_kobs30
  - V_rollmean_G_API_kobs14
  - lia_mean_asc_deg
  - J_bio_bio04
  - DOY
  - SMAP_sm_am_interp
  - J_bio_bio07
  - SMAP_sm_am_interp_lag1
  - C_lag_F_NDMI_kobs30
  - SMAP_sm_pm_interp_lag1
```

### XGBoost Hyperparameters (Frozen)
```yaml
n_estimators: 2500
learning_rate: 0.005
max_depth: 9
min_child_weight: 8
gamma: 0.0
reg_lambda: 0.75
reg_alpha: 0.03
subsample: 0.9
colsample_bytree: 0.8
tree_method: "hist"
device: "cuda" (with CPU fallback)
```

---

## Expected Results & Decision Tree

### Scenario 1: v21 Recovers to R² ≥ 0.834 ✅ SUCCESS
- **Interpretation**: Retraining with correct pipeline recovered the original performance
- **Next Step**: Use retrained v21 checkpoint as reference; document root cause as "training pipeline mismatch"
- **Optional**: Investigate why other versions didn't improve as much

### Scenario 2: v21 Improves but < 0.834 (e.g., 0.82)
- **Interpretation**: Partial recovery; something else may affect performance
- **Possible Causes**:
  - Different random seed convergence
  - Data leakage in original 0.834 score
  - Representation extraction method differs subtly
- **Next Steps**:
  1. Compare v21 to v16 (best other version) — is the gap consistent?
  2. Check if v16 or another version outperforms v21
  3. Consider PCA at different levels (64, 95% variance) for v21
  4. Try alternative router feature sets if available

### Scenario 3: v21 Stays ~0.738 (No Improvement)
- **Interpretation**: Retraining didn't fix it; issue is not training pipeline
- **Possible Causes**:
  - Model architecture incompatibility
  - Feature set mismatch
  - Incorrect representation extraction method
  - Data preprocessing difference
- **Next Steps**:
  1. Compare v21's LSTM validation R² across experiments (should match if properly trained)
  2. Manually inspect representation arrays for NaNs, outliers, scale issues
  3. Retrain v21 using optimization-2.0's full training code directly (not sweep-3.0)
  4. Compare raw (non-PCA) v21 representation performance

### Scenario 4: Another Version Outperforms v21 (e.g., v16 > 0.834)
- **Interpretation**: This LSTM variant is better when properly trained
- **Next Steps**:
  1. Validate reproducibility (run twice, confirm stable)
  2. Investigate architectural differences (why does v16 work better?)
  3. Consider using the better version for production
  4. Document findings for future LSTM architecture development

---

## Next Steps After Experiment Completes

### 1. Immediate (When log shows "Retrain experiment complete")
```bash
# Read final leaderboard
tail -50 run_retrain.log  # Inspect summary_records.csv output

# Check v21 row specifically
grep "v21" artifacts/summary_records.csv

# Verify output files exist
ls -lh artifacts/summary_records.csv artifacts/lstm_metadata.json
wc -l artifacts/summary_records.csv  # Should have 16 rows (baseline + 15 versions)
```

### 2. Analysis (30–60 min)
```bash
# Load the CSV and analyze
python3 << 'EOF'
import pandas as pd
df = pd.read_csv("artifacts/summary_records.csv")
print("\n=== LEADERBOARD ===")
print(df[["model_name", "pooled_r2", "lstm_validation_r2"]].to_string())

# Highlight v21
v21_row = df[df["model_name"].str.contains("v21")]
if not v21_row.empty:
    r2 = v21_row["pooled_r2"].values[0]
    print(f"\n✓ v21 RESULT: R² = {r2:.5f}")
    if r2 >= 0.834:
        print("  ✅ RECOVERED to expected 0.8340+")
    elif r2 >= 0.82:
        print("  ⚠️  PARTIAL RECOVERY (< 0.834, but improved)")
    else:
        print("  ❌ NO IMPROVEMENT (still ~0.738)")
EOF
```

### 3. Documentation
- Update main README with findings
- Add a note to `Models/Temporal/lstm/TRAINING_NOTES.md` about which LSTM version is recommended
- If v21 recovers, freeze the retrained checkpoint location

### 4. Decide Next Phase
| Result | Action |
|--------|--------|
| v21 ≥ 0.834 | Use retrained v21; done |
| v21 0.82–0.833 | Investigate architectural tweaks (PCA levels, router features) |
| v21 < 0.80 | Debug representation extraction; try optimization-2.0's training directly |
| Other > 0.834 | Validate reproducibility; consider using that version instead |

---

## Critical Notes for Handoff

### ⚠️ Important Constraints
1. **Config is frozen**: Do NOT modify XGBoost hyperparameters, router features, or c1 additions — they are intentionally identical across all experiments
2. **Checkpoint safety**: The retrained checkpoints are in `training_runs/<version>/`, not sweep-3.0's original location — this is intentional to avoid cache contamination
3. **GPU vs CPU**: Script defaults to CPU for safety; if GPU available, training will be 10x faster (adjust device in core.py if needed)
4. **Early stopping**: Each LSTM trains for up to 300 epochs but stops early if validation doesn't improve for 60 epochs — this is tuned in sweep-3.0/core.py

### 📋 Files Modified in This Session
- ✅ `derived_8.0-lstm-retraining-3.0/run_retrain.py` — Created
- ✅ `derived_8.0-lstm-retraining-3.0/config.yaml` — Copied from optimization-2.0
- ✅ `derived_8.0-lstm-retraining-3.0/README.md` — Created
- ✅ `memory/retraining_strategy.md` — Created (LLM memory note)
- ✅ `HANDOFF_LSTM_RETRAINING.md` — This file

### 📋 Files NOT Modified (Preserve)
- ❌ `derived_8.0-optimization-2.0/` — Read-only reference
- ❌ `derived_8.0-hybrid-lstm-sweep-3.0/` — Read-only reference
- ❌ `Models/Temporal/lstm/` — Read-only source code
- ❌ `data/splits/derived_8.0/` — Read-only data

### 🔍 Debugging Checklist (If Experiment Fails)

**If training is suspiciously fast (<1 hour for all 15 versions)**:
- Check log for "checkpoint reused" messages — should NOT appear with max_epochs=300
- Verify output_dir doesn't have pre-existing checkpoints: `ls -la training_runs/*/models/*/`
- Confirm max_epochs=300 is being passed to train_candidate()

**If v21 result is identical to comparison experiments**:
- Check if checkpoints are being loaded from sweep-3.0 instead of fresh training_runs
- Inspect training_runs/<v21>/models/<v21>/seed42.pt modification time (should be recent)
- Check if output_dir has a "models/" subdirectory (sweep-3.0 artifact structure)

**If LSTM representations have NaNs or weird scales**:
- Check representation extraction method in sweep-3.0/core.py (_predict_and_represent)
- Verify PCA fit direction (should fit on train, then transform val/test)
- Ensure datasets are sorted by row_id after extraction (_ordered function)

**If XGBoost evaluation crashes**:
- Verify ctx_train.npy, ctx_val.npy, ctx_test.npy exist for each version
- Check array shapes match CSV row counts
- Confirm c1_additions list is resolved correctly (should be 10 features)

---

## Summary of Prior Experiments (for Reference)

| Experiment | Status | Purpose | Result |
|------------|--------|---------|--------|
| optimization-2.0 | ✅ Complete | Original v21 model; reported R²=0.834 | Baseline truth |
| sweep-3.0 | ✅ Complete | Multi-version sweep; cached checkpoints | Source of cache (not retrained) |
| comparison-1.0 | ✅ Complete | Compared all v7-v23 using sweep-3.0 cache | v21 degraded to 0.7376 |
| comparison-2.0 | ✅ Complete | Same as 1.0, different extraction method | Same degradation (0.7376) |
| **retraining-3.0** | 🔄 IN PROGRESS | **Retrain all from scratch** | **Expected: v21 → 0.8340** |

---

## Contact / Questions

If experiment fails or behaves unexpectedly:
1. Check the debugging checklist above
2. Review run_retrain.log for error messages
3. Verify all paths exist: `ls -la notebooks/experiment/derived_8.0-lstm-retraining-3.0/`
4. Check if processes are still running: `ps aux | grep python`
5. If stuck, restart with: `rm -rf training_runs/* && python3 run_retrain.py`

---

**Last Updated**: 2026-08-19  
**Handoff Version**: 1.0
