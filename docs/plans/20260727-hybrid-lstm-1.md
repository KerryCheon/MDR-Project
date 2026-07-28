# Implementation Plan — `derived_8.4-hybrid-lstm-1.0` (Updated)

## Goal Description
The objective of this experiment (`derived_8.4-hybrid-lstm-1.0`) is to build and evaluate a **Hybrid LSTM + XGBoost** modeling architecture on the `derived_8.4` dataset split (7 Washington stations).

Per user directive, **no existing code in `Models/Temporal/lstm/` will be modified**. All LSTM model logic (`BiLSTMAttn`, dataset sequence construction, training loop, and frozen `ctx` vector extraction) will be placed in a self-contained local copy under `notebooks/experiment/derived_8.4-hybrid-lstm-1.0/lstm/`.

The pipeline operates in three sequential phases:
1. **Normal LSTM Training**: Train local `BiLSTMAttn` model on `derived_8.4` sequence dataset until convergence (best validation RMSE checkpoint saved).
2. **Freeze & Extract Context (`ctx`)**: Load the converged, frozen model (`eval()` mode, `torch.no_grad()`) and extract the 160-dimensional attention-pooled hidden state (`ctx`) for all samples (`train`, `val`, `test`).
3. **XGBoost Feature Fusion & Evaluation**: Concatenate the frozen `ctx` vectors with tabular features (54 global backbone + cluster additions) to train XGBoost regressors and compare against the baselines from `derived_8.4-eval-1.1`.

---

## User Review Required

> [!IMPORTANT]
> **No Mutations to `Models/Temporal/lstm/`**: `Models/Temporal/lstm/` remains 100% untouched. All LSTM modeling logic is localized within `notebooks/experiment/derived_8.4-hybrid-lstm-1.0/lstm/`.

> [!NOTE]
> **Sequential 3-Phase Architecture**:
> ```mermaid
> graph TD
>     A["Phase 1: Local BiLSTM Training<br/>(derived_8.4 sequences)"] -->|Early Stop & Save Checkpoint| B["best_model.pt"]
>     B --> C["Phase 2: Freeze Model & Extract CTX<br/>(model.eval(), no_grad)"]
>     C -->|Save ctx_0..159 vectors| D["ctx_train.npy / ctx_val.npy / ctx_test.npy"]
>     D --> E["Phase 3: XGBoost Hybrid Modeling<br/>([54 Backbone + 160 CTX])"]
>     E --> F["Evaluation & Leaderboard"]
> ```

---

## Open Questions

None currently.

---

## Proposed Changes

### `notebooks/experiment/derived_8.4-hybrid-lstm-1.0/`

#### [NEW] [lstm/model.py](../../notebooks/experiment/derived_8.4-hybrid-lstm-1.0/lstm/model.py)
- Self-contained local copy of `BiLSTMAttn` architecture.
- Exposes `forward(self, x, return_ctx: bool = False)`:
  ```python
  def forward(self, x, return_ctx: bool = False):
      b, s, f = x.shape
      x = self.proj(x.reshape(b * s, f)).reshape(b, s, -1)
      out, _ = self.lstm(x)                  # (B, S, 2H)
      scores = self.attn(out).squeeze(-1)    # (B, S)
      weights = torch.softmax(scores, dim=-1)
      ctx = (out * weights.unsqueeze(-1)).sum(dim=1)   # (B, 2H)
      out_pred = self.head(self.dropout(ctx)).squeeze(-1)
      if return_ctx:
          return out_pred, ctx
      return out_pred
  ```

#### [NEW] [lstm/dataset.py](../../notebooks/experiment/derived_8.4-hybrid-lstm-1.0/lstm/dataset.py)
- Self-contained local sequence dataset builder wrapping `_build_sequences` for `derived_8.4` dataset splits.

#### [NEW] [lstm/train.py](../../notebooks/experiment/derived_8.4-hybrid-lstm-1.0/lstm/train.py)
- Performs **Phase 1** (LSTM training until early-stopping convergence on `derived_8.4`) and **Phase 2** (freezing `best_model.pt` and extracting `ctx` vectors into `artifacts/ctx_*.npy`).

#### [NEW] [config.yaml](../../notebooks/experiment/derived_8.4-hybrid-lstm-1.0/config.yaml)
- Configuration specifying `derived_8.4` data splits, 54 shared backbone features, delta addition rules, and exact XGBoost parameters matching `derived_8.4-eval-1.1`.

#### [NEW] [eval_hybrid/data.py](../../notebooks/experiment/derived_8.4-hybrid-lstm-1.0/eval_hybrid/data.py)
- Data loading module that reads `derived_8.4` tabular splits, merges the saved frozen `ctx_*.npy` arrays, and constructs input matrices for XGBoost.

#### [NEW] [eval_hybrid/evaluator.py](../../notebooks/experiment/derived_8.4-hybrid-lstm-1.0/eval_hybrid/evaluator.py)
- Performs **Phase 3**: Fits:
  1. `Global Single Model (54 Backbone + 160 CTX)`
  2. `Clustering_V0_Full_k2 (c0=0, c1=10 + 160 CTX)`
- Evaluates metrics ($R^2$, $\text{RMSE}$, $\text{ubRMSE}$, $\text{Bias}$, $\text{MAE}$, $\text{Pearson}$), per-regime breakdown, and year-by-year performance metrics.

#### [NEW] [run_eval.py](../../notebooks/experiment/derived_8.4-hybrid-lstm-1.0/run_eval.py)
- Main CLI execution script orchestrating Phase 1, Phase 2, and Phase 3.

#### [NEW] [derived_8.4-hybrid-lstm-1.0.ipynb](../../notebooks/experiment/derived_8.4-hybrid-lstm-1.0/derived_8.4-hybrid-lstm-1.0.ipynb)
- Experiment notebook documenting the workflow, displaying metrics tables, and saving outputs.

#### [NEW] [README.md](../../notebooks/experiment/derived_8.4-hybrid-lstm-1.0/README.md)
- Complete summary report with leaderboard tables comparing pure tabular baselines vs. hybrid LSTM+XGBoost models.

---

## Verification Plan

### Automated Tests
1. Run pipeline:
   ```bash
   cd notebooks/experiment/derived_8.4-hybrid-lstm-1.0
   uv run python run_eval.py
   ```
2. Verify experiment notebook execution:
   ```bash
   cd notebooks
   nb execute experiment/derived_8.4-hybrid-lstm-1.0/derived_8.4-hybrid-lstm-1.0.ipynb --uv
   ```

### Manual Verification
- Check `README.md` leaderboard table.
- Confirm `Models/Temporal/lstm/` remains completely unchanged via `git status`.
