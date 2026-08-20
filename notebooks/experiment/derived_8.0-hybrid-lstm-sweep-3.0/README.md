# Experiment: `derived_8.0-hybrid-lstm-sweep-3.0`

This experiment determines which historical LSTM-family recipe contributes the most useful frozen representation to the fixed two-regime model on `derived_8.0`. The router, 54-feature backbone, cluster-1 additions, and XGBoost hyperparameters are held constant.

## Protocol

Every candidate is trained on the repository train split with early stopping on validation RMSE. Its penultimate prediction-head activation is compressed to PCA-32 using training rows only, then appended to the fixed tabular backbone. Candidates are ranked using validation R²; only the validation-selected winner is evaluated on the 2023–2025 test split. The top three screen candidates are repeated with seeds 42, 7, and 123 before final selection.

The comparison uses a coverage-preserving, left-padded sequence builder so each source row has exactly one representation. This corrects the historical builder's loss of the first `seq_len` rows of each station. It also means standalone LSTM scores may differ from the archived version reports.

## Candidate scope

The sweep covers the substantive v7–v17 model recipes plus the later v20, v21, v22, and v23 raw-input recipe. v18 and v19 are excluded because they change the temporal split rather than the encoder. v23's multi-seed work is represented by the confirmation stage instead of being mislabeled as a new architecture.

## Important limitation

The cluster-1 feature additions are frozen from the configuration that produced the historical 0.834 result. Those additions were originally informed by downstream evaluation work, so this experiment answers which LSTM works best with that frozen historical hybrid; it does not remove that older provenance concern.

## Run

From `notebooks/`:

```bash
uv run python experiment/derived_8.0-hybrid-lstm-sweep-3.0/run_experiment.py
uv run python experiment/derived_8.0-hybrid-lstm-sweep-3.0/validate_artifacts.py
nb execute experiment/derived_8.0-hybrid-lstm-sweep-3.0/derived_8.0-hybrid-lstm-sweep-3.0.ipynb --uv
```

The executed leaderboard and conclusion are written to `artifacts/RESULTS.md` and displayed by the notebook.
