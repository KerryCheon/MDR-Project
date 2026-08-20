#!/usr/bin/env python3
"""
Experiment: derived_8.0-lstm-retraining-3.0
Retrain all LSTM versions (v7-v23 excluding v18/v19) from scratch using optimization-2.0's training infrastructure.
Then evaluate each with the 2-regime clustering model.

Key difference from comparison-1.0/2.0:
- Uses sweep-3.0's train_candidate(..., max_epochs=300) to RETRAIN from scratch
- NOT using cached checkpoints
- This should recover the original 0.8340 performance for v21 and potentially improve other versions
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
import torch

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
SWEEP3_DIR = PROJECT_ROOT / "notebooks/experiment/derived_8.0-hybrid-lstm-sweep-3.0"
OPT20_DIR = PROJECT_ROOT / "notebooks/experiment/derived_8.0-optimization-2.0"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(OPT20_DIR))

CONFIG_PATH = EXP_DIR / "config.yaml"
ARTIFACTS_DIR = EXP_DIR / "artifacts"
MODELS_DIR = EXP_DIR / "models"


def main():
    print("=" * 75, flush=True)
    print("Starting derived_8.0-lstm-retraining-3.0: Full Retraining Sweep", flush=True)
    print("Retraining all LSTM versions from scratch (NO checkpoint cache)", flush=True)
    print("=" * 75, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    import importlib.util

    spec_sweep = importlib.util.spec_from_file_location("sweep_core", SWEEP3_DIR / "core.py")
    sweep_core = importlib.util.module_from_spec(spec_sweep)
    sys.modules["sweep_core"] = sweep_core
    spec_sweep.loader.exec_module(sweep_core)

    spec_opt = importlib.util.spec_from_file_location("opt_run_eval", OPT20_DIR / "run_eval.py")
    opt_run_eval = importlib.util.module_from_spec(spec_opt)
    sys.modules["opt_run_eval"] = opt_run_eval
    spec_opt.loader.exec_module(opt_run_eval)

    from eval_hybrid.data import load_hybrid_experiment_data
    from eval_hybrid.evaluator import HybridStrategyEvaluator

    sweep_config = yaml.safe_load((SWEEP3_DIR / "config.yaml").read_text(encoding="utf-8"))
    candidates = sweep_core.build_candidates()

    print(f"\n[Config] Loaded {len(candidates)} candidates from sweep-3.0", flush=True)
    print(f"[Data] {config['data']['splits']}", flush=True)

    splits = sweep_core.load_raw_splits(sweep_config)
    c1_additions = opt_run_eval.compute_c1_gain_additions(config)
    print(f"[Cluster 1 Deltas] {c1_additions}", flush=True)

    summary_records = []

    print("\n" + "=" * 75, flush=True)
    print("Tabular-only baseline (no LSTM features)", flush=True)
    print("=" * 75, flush=True)

    backbone_54 = config["shared_backbone_54"]
    print(f"[Data] Backbone={len(backbone_54)}", flush=True)

    baseline_artifact_dir = ARTIFACTS_DIR / "baseline"
    baseline_artifact_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROJECT_ROOT / config["data"]["splits"]["train"])
    val_df = pd.read_csv(PROJECT_ROOT / config["data"]["splits"]["val"])
    test_df = pd.read_csv(PROJECT_ROOT / config["data"]["splits"]["test"])

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)

    dummy_train = np.zeros((n_train, 1), dtype=np.float32)
    dummy_val = np.zeros((n_val, 1), dtype=np.float32)
    dummy_test = np.zeros((n_test, 1), dtype=np.float32)

    np.save(baseline_artifact_dir / "ctx_train.npy", dummy_train)
    np.save(baseline_artifact_dir / "ctx_val.npy", dummy_val)
    np.save(baseline_artifact_dir / "ctx_test.npy", dummy_test)

    baseline_config = {**config, "artifacts": {"directory": str(baseline_artifact_dir)}}
    data_base = load_hybrid_experiment_data(
        PROJECT_ROOT, EXP_DIR, baseline_config, repr_type="ctx", pca_suffix=""
    )

    eval_baseline = HybridStrategyEvaluator(
        data_base, baseline_config, "Clustering_V0_Full_k2", models_dir=MODELS_DIR
    )
    res_baseline = eval_baseline.fit_and_evaluate(
        model_name="Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10) [No LSTM]",
        candidate_id="baseline_54_only",
        global_features=data_base.hybrid_features[:len(backbone_54)],
        cluster_additions={"0": [], "1": c1_additions},
    )
    rec_baseline = res_baseline.as_record()
    summary_records.append(rec_baseline)
    print(f"[Baseline] R²={rec_baseline['pooled_r2']:.5f}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("Retraining all LSTM versions from scratch", flush=True)
    print("=" * 75, flush=True)

    lstm_metadata = {}
    device = torch.device("cpu")

    for i, candidate in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] [{candidate.version}] {candidate.architecture}", flush=True)

        # RETRAIN from scratch: max_epochs=300 (not None)
        # Use a FRESH output_dir with no existing checkpoints to force training
        fresh_output_dir = EXP_DIR / "training_runs" / candidate.name
        fresh_output_dir.mkdir(parents=True, exist_ok=True)

        run = sweep_core.train_candidate(
            candidate,
            seed=42,
            config=sweep_config,
            splits=splits,
            output_dir=fresh_output_dir,  # No pre-existing checkpoints here
            include_test=True,
            max_epochs=300,  # <-- KEY: Actually train, don't use cache
        )

        print(
            f"  [LSTM] validation R²={run.lstm_validation_r2:.5f}, "
            f"repr_dim={run.representation_dimensions}",
            flush=True,
        )

        train_repr = run.train_representations
        val_repr = run.validation_representations
        test_repr = run.test_representations

        pca_components = 32
        pca = PCA(n_components=pca_components, random_state=42, svd_solver="auto")
        train_repr_pca = pca.fit_transform(train_repr)
        val_repr_pca = pca.transform(val_repr)
        test_repr_pca = pca.transform(test_repr) if test_repr is not None else None
        var_expl = pca.explained_variance_ratio_.sum()

        print(
            f"  [PCA] {train_repr.shape[1]}->{pca_components} comps, var={var_expl:.4f}",
            flush=True,
        )

        version_artifact_dir = ARTIFACTS_DIR / candidate.name
        version_artifact_dir.mkdir(parents=True, exist_ok=True)

        np.save(version_artifact_dir / "ctx_train.npy", train_repr_pca)
        np.save(version_artifact_dir / "ctx_val.npy", val_repr_pca)
        np.save(version_artifact_dir / "ctx_test.npy", test_repr_pca)

        lstm_metadata[candidate.name] = {
            "version": candidate.version,
            "architecture": candidate.architecture,
            "lstm_validation_r2": float(run.lstm_validation_r2),
            "repr_dim_raw": int(run.representation_dimensions),
            "repr_dim_pca": pca_components,
            "variance_explained": float(var_expl),
        }

        version_config = {**config, "artifacts": {"directory": str(version_artifact_dir)}}
        data_hybrid = load_hybrid_experiment_data(
            PROJECT_ROOT, EXP_DIR, version_config, repr_type="ctx", pca_suffix=""
        )

        eval_version = HybridStrategyEvaluator(
            data_hybrid, version_config, "Clustering_V0_Full_k2", models_dir=MODELS_DIR
        )

        res_version = eval_version.fit_and_evaluate(
            model_name=f"{candidate.version} {candidate.architecture} (2-Regime, PCA-32, RETRAINED)",
            candidate_id=f"retrained_{candidate.name}",
            global_features=data_hybrid.hybrid_features,
            cluster_additions={"0": [], "1": c1_additions},
        )

        rec_version = res_version.as_record()
        rec_version["lstm_validation_r2"] = run.lstm_validation_r2
        rec_version["repr_dim_raw"] = run.representation_dimensions
        rec_version["repr_dim_pca"] = pca_components
        rec_version["variance_explained"] = var_expl

        summary_records.append(rec_version)
        print(f"  [2-Regime] Test R²={rec_version['pooled_r2']:.5f}", flush=True)

    df_summary = pd.DataFrame(summary_records).sort_values("pooled_r2", ascending=False)
    df_summary.to_csv(ARTIFACTS_DIR / "summary_records.csv", index=False)

    with open(ARTIFACTS_DIR / "lstm_metadata.json", "w") as f:
        json.dump(lstm_metadata, f, indent=2)

    print("\n" + "=" * 75, flush=True)
    print("FINAL LEADERBOARD (derived_8.0-lstm-retraining-3.0 — RETRAINED)", flush=True)
    print("=" * 75, flush=True)
    display_cols = [
        "model_name",
        "pooled_r2",
        "pooled_rmse",
        "pooled_ubrmse",
        "pooled_bias",
        "pooled_mae",
        "lstm_validation_r2",
        "repr_dim_raw",
        "variance_explained",
    ]
    avail_cols = [c for c in display_cols if c in df_summary.columns]
    print(df_summary[avail_cols].to_string(index=False), flush=True)

    print(f"\n[Output] {ARTIFACTS_DIR / 'summary_records.csv'}", flush=True)
    print(f"[Output] {ARTIFACTS_DIR / 'lstm_metadata.json'}", flush=True)
    print("\nRetrain experiment complete.", flush=True)


if __name__ == "__main__":
    main()
