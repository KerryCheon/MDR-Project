from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cache")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA

from common import (
    ARTIFACT_DIR,
    EXPERIMENT_DIR,
    ID_COLUMNS,
    SEED,
    TABULAR_FEATURES,
    TARGET,
    add_soil_texture_one_hot,
    evaluate_predictions,
    feature_variants,
    fit_xgboost,
    load_splits,
    make_prediction_frame,
    save_json,
    temporal_feature_variants,
    unique,
)
from temporal import (
    FrozenLSTMExtractor,
    SequenceData,
    TemporalPreprocessor,
    build_sequences,
    predict_and_embed,
    save_training_curve,
    sequence_prediction_frame,
    train_model,
)

LEGACY_TEMPORAL_STATIONS = [
    "Darrington",
    "Quinault",
    "SourdoughGulch_WA_985",
    "Spokane",
    "Touchet_WA_824",
]


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


def prepare_source_splits():
    splits, texture_columns = add_soil_texture_one_hot(load_splits())
    splits = {
        name: frame[frame["station_id"].isin(LEGACY_TEMPORAL_STATIONS)].copy()
        for name, frame in splits.items()
    }
    for name, frame in splits.items():
        frame["_split"] = name
    combined = pd.concat(splits.values(), ignore_index=True)
    combined = combined.sort_values(["station_id", "date"]).reset_index(drop=True)
    return splits, combined, texture_columns


def prepare_temporal_data(
    combined: pd.DataFrame,
    features: list[str],
    *,
    fit_mask: pd.Series,
    target_masks: dict[str, pd.Series],
    sequence_length: int,
    max_span_factor: float,
):
    preprocessor = TemporalPreprocessor().fit(combined.loc[fit_mask], features)
    scaled = preprocessor.transform(combined, features)
    datasets = {
        name: build_sequences(
            scaled,
            features,
            target_mask,
            sequence_length=sequence_length,
            max_span_factor=max_span_factor,
        )
        for name, target_mask in target_masks.items()
    }
    return preprocessor, datasets


def temporal_evaluation(data: SequenceData, predictions: np.ndarray) -> dict:
    return evaluate_predictions(sequence_prediction_frame(data, predictions))


def train_encoder_experiment(
    name: str,
    combined: pd.DataFrame,
    features: list[str],
    config: dict,
    device: torch.device,
):
    temporal_config = config["temporal_encoder"]
    output_dir = ARTIFACT_DIR / "hybrid" / "encoders" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_mask = combined["_split"] == "train"
    target_masks = {
        split_name: combined["_split"] == split_name
        for split_name in ("train", "val", "test")
    }
    preprocessor, datasets = prepare_temporal_data(
        combined,
        features,
        fit_mask=fit_mask,
        target_masks=target_masks,
        sequence_length=temporal_config["sequence_length"],
        max_span_factor=temporal_config["max_window_span_factor"],
    )
    print(
        f"[encoder:{name}] features={len(features)} "
        f"train={len(datasets['train'].y)} val={len(datasets['val'].y)} "
        f"test={len(datasets['test'].y)} excluded_test={datasets['test'].excluded_gap_targets}"
    )
    model, history, best_epoch = train_model(
        datasets["train"],
        datasets["val"],
        n_features=len(features),
        projection_size=temporal_config["projection_size"],
        hidden_size=temporal_config["hidden_size"],
        embedding_size=temporal_config["embedding_size"],
        dropout=temporal_config["dropout"],
        batch_size=temporal_config["batch_size"],
        learning_rate=temporal_config["learning_rate"],
        weight_decay=temporal_config["weight_decay"],
        max_epochs=temporal_config["max_epochs"],
        patience=temporal_config["patience"],
        huber_delta=temporal_config["huber_delta"],
        seed=SEED,
        device=device,
    )
    save_training_curve(history, output_dir / "training_curve.png", name)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": features,
            "config": temporal_config,
            "best_epoch": best_epoch,
        },
        output_dir / "best_model.pt",
    )
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    predictions = {}
    embeddings = {}
    metrics = {
        "feature_count": len(features),
        "features": features,
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "excluded_gap_targets": {
            split_name: data.excluded_gap_targets
            for split_name, data in datasets.items()
        },
    }
    for split_name, data in datasets.items():
        split_predictions, split_embeddings = predict_and_embed(
            model,
            data,
            batch_size=temporal_config["batch_size"],
            device=device,
        )
        predictions[split_name] = split_predictions
        embeddings[split_name] = split_embeddings
        metrics[split_name] = temporal_evaluation(data, split_predictions)
        sequence_prediction_frame(data, split_predictions).to_csv(
            output_dir / f"{split_name}_predictions.csv", index=False
        )
    save_json(metrics, output_dir / "metrics.json")
    return {
        "model": model,
        "preprocessor": preprocessor,
        "datasets": datasets,
        "predictions": predictions,
        "embeddings": embeddings,
        "metrics": metrics,
        "features": features,
    }


def attach_embeddings(
    source: pd.DataFrame,
    data: SequenceData,
    predictions: np.ndarray,
    embeddings: np.ndarray,
    *,
    embedding_prefix: str = "lstm_z_",
) -> tuple[pd.DataFrame, list[str]]:
    metadata = data.metadata.copy()
    merged = metadata.merge(source, on=ID_COLUMNS, how="left", validate="one_to_one")
    if merged[TARGET].isna().any():
        raise ValueError("Embedding rows failed to join to source tabular rows")
    if not np.allclose(merged[TARGET].to_numpy(dtype=float), data.y, equal_nan=True):
        raise ValueError("Temporal and tabular targets are not aligned")
    embedding_columns = [
        f"{embedding_prefix}{index:02d}" for index in range(embeddings.shape[1])
    ]
    merged[embedding_columns] = embeddings
    merged["lstm_scalar_prediction"] = predictions
    return merged, embedding_columns


def add_random_control(
    frames: dict[str, pd.DataFrame], embedding_columns: list[str]
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    rng = np.random.default_rng(20260721)
    random_columns = [f"random_z_{index:02d}" for index in range(len(embedding_columns))]
    shuffled_columns = [
        f"shuffled_z_{index:02d}" for index in range(len(embedding_columns))
    ]
    outputs = {}
    for split_name, frame in frames.items():
        random_values = rng.normal(
            size=(len(frame), len(embedding_columns))
        ).astype(np.float32)
        permutation = rng.permutation(len(frame))
        shuffled_values = frame[embedding_columns].to_numpy()[permutation]
        control_frame = pd.DataFrame(
            np.column_stack([random_values, shuffled_values]),
            columns=random_columns + shuffled_columns,
            index=frame.index,
        )
        outputs[split_name] = pd.concat([frame.copy(), control_frame], axis=1)
    return outputs, random_columns, shuffled_columns


def run_tree_suite(
    protocol_name: str,
    frames: dict[str, pd.DataFrame],
    embedding_columns: list[str],
    selected_tabular_features: list[str],
) -> dict:
    output_dir = ARTIFACT_DIR / "hybrid" / protocol_name
    prediction_dir = output_dir / "predictions"
    model_dir = output_dir / "models"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frames, random_columns, shuffled_columns = add_random_control(
        frames, embedding_columns
    )
    model_features = {
        "T0_tabular_38": list(TABULAR_FEATURES),
        "T0_selected_tabular": selected_tabular_features,
        "T2_tabular_plus_scalar": unique(
            list(TABULAR_FEATURES) + ["lstm_scalar_prediction"]
        ),
        "T2_selected_tabular_plus_scalar": unique(
            selected_tabular_features + ["lstm_scalar_prediction"]
        ),
        "T3_tabular_plus_embedding": unique(
            list(TABULAR_FEATURES) + embedding_columns
        ),
        "T3_selected_tabular_plus_embedding": unique(
            selected_tabular_features + embedding_columns
        ),
        "T4_embedding_only": embedding_columns,
        "T5_tabular_plus_random": unique(list(TABULAR_FEATURES) + random_columns),
        "T5_selected_tabular_plus_random": unique(
            selected_tabular_features + random_columns
        ),
        "T6_selected_tabular_plus_shuffled_embedding": unique(
            selected_tabular_features + shuffled_columns
        ),
    }
    results = {
        "T1_temporal_only": {
            split_name: evaluate_predictions(
                make_prediction_frame(frame, frame["lstm_scalar_prediction"])
            )
            for split_name, frame in frames.items()
        }
    }

    summary_rows = []
    for split_name, evaluation in results["T1_temporal_only"].items():
        summary_rows.append(
            {
                "model": "T1_temporal_only",
                "split": split_name,
                **evaluation["overall"],
                "macro_station_r2": evaluation["macro_station_r2"],
            }
        )

    for model_name, features in model_features.items():
        print(f"[hybrid:{protocol_name}] {model_name} features={len(features)}")
        imputer, model = fit_xgboost(frames["train"], frames["val"], features)
        payload = {
            "feature_count": len(features),
            "features": features,
            "best_iteration": int(model.best_iteration),
        }
        for split_name, frame in frames.items():
            predictions = model.predict(imputer.transform(frame[features]))
            prediction_frame = make_prediction_frame(frame, predictions)
            prediction_frame.to_csv(
                prediction_dir / f"{model_name}_{split_name}.csv", index=False
            )
            evaluation = evaluate_predictions(prediction_frame)
            payload[split_name] = evaluation
            summary_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    **evaluation["overall"],
                    "macro_station_r2": evaluation["macro_station_r2"],
                }
            )
        model.save_model(model_dir / f"{model_name}.json")
        joblib.dump(
            {"imputer": imputer, "features": features},
            model_dir / f"{model_name}_preprocessor.joblib",
        )
        results[model_name] = payload

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    save_json(results, output_dir / "metrics.json")

    plot = summary[summary["split"].isin(["val", "test"])].copy()
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    model_order = list(dict.fromkeys(plot["model"]))
    x = np.arange(len(model_order))
    for split_name, offset, color in (
        ("val", -0.18, "tab:blue"),
        ("test", 0.18, "tab:orange"),
    ):
        group = plot[plot["split"] == split_name].set_index("model").loc[model_order]
        axes[0].bar(x + offset, group["r2"], width=0.36, label=split_name, color=color)
        axes[1].bar(x + offset, group["rmse"], width=0.36, label=split_name, color=color)
    axes[0].set_ylabel("R2")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_order, rotation=25, ha="right")
    axes[0].legend()
    axes[1].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(f"Hybrid model comparison: {protocol_name}")
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(figure)
    return results


def load_exported_embedding_frames(
    protocol_name: str,
    source: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames = {}
    embedding_columns = None
    for split_name in ("train", "val", "test"):
        embeddings = pd.read_csv(
            ARTIFACT_DIR / "hybrid" / protocol_name / "embeddings" / f"{split_name}.csv"
        )
        embeddings["date"] = pd.to_datetime(embeddings["date"])
        frame = embeddings.merge(source, on=ID_COLUMNS, how="left", validate="one_to_one")
        if frame[TARGET].isna().any():
            raise ValueError(f"Missing source rows for exported {protocol_name} embeddings")
        frames[split_name] = frame
        embedding_columns = [
            column for column in embeddings.columns if column.startswith("lstm_z_")
        ]
    return frames, embedding_columns


@torch.no_grad()
def trace_representation(
    model: FrozenLSTMExtractor,
    train_data: SequenceData,
    test_data: SequenceData,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    *,
    device: torch.device,
) -> None:
    output_dir = ARTIFACT_DIR / "hybrid" / "hidden_trace"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_mean = train_embeddings.mean(axis=0)
    train_std = train_embeddings.std(axis=0)
    train_std[train_std < 1e-6] = 1.0

    rows = []
    for split_name, data, embeddings in (
        ("train", train_data, train_embeddings),
        ("test", test_data, test_embeddings),
    ):
        for start in range(0, len(data.y), 512):
            end = min(start + 512, len(data.y))
            sequence = torch.from_numpy(data.X[start:end]).to(device)
            trace = model.forward_with_trace(sequence)
            input_abs = trace["input"].abs().mean(dim=(1, 2)).cpu().numpy()
            projected_norm = trace["projected"].norm(dim=2).mean(dim=1).cpu().numpy()
            recurrent_norm = trace["recurrent"].norm(dim=2).mean(dim=1).cpu().numpy()
            hidden_norm = trace["final_hidden"].norm(dim=1).cpu().numpy()
            embedding_norm = trace["embedding"].norm(dim=1).cpu().numpy()
            prediction = trace["prediction"].cpu().numpy()
            ood_score = np.sqrt(
                np.mean(((embeddings[start:end] - train_mean) / train_std) ** 2, axis=1)
            )
            metadata = data.metadata.iloc[start:end].reset_index(drop=True)
            for index in range(end - start):
                rows.append(
                    {
                        "split": split_name,
                        "station_id": metadata.loc[index, "station_id"],
                        "date": metadata.loc[index, "date"],
                        "input_abs_mean": float(input_abs[index]),
                        "projected_norm_mean": float(projected_norm[index]),
                        "recurrent_norm_mean": float(recurrent_norm[index]),
                        "final_hidden_norm": float(hidden_norm[index]),
                        "embedding_norm": float(embedding_norm[index]),
                        "embedding_ood_score": float(ood_score[index]),
                        "y_true": float(data.y[start + index]),
                        "y_pred": float(prediction[index]),
                    }
                )
    trace_frame = pd.DataFrame(rows)
    trace_frame.to_csv(output_dir / "hidden_trace_summary.csv", index=False)
    station_summary = (
        trace_frame[trace_frame["split"] == "test"]
        .groupby("station_id")
        [[
            "input_abs_mean",
            "projected_norm_mean",
            "recurrent_norm_mean",
            "final_hidden_norm",
            "embedding_norm",
            "embedding_ood_score",
        ]]
        .mean()
        .reset_index()
    )
    station_summary.to_csv(output_dir / "hidden_trace_by_station.csv", index=False)

    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(train_embeddings)
    transformed = pca.transform(test_embeddings)
    touchet = test_data.metadata["station_id"].to_numpy() == "Touchet_WA_824"
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        transformed[~touchet, 0],
        transformed[~touchet, 1],
        s=9,
        alpha=0.30,
        label="Other stations",
    )
    axis.scatter(
        transformed[touchet, 0],
        transformed[touchet, 1],
        s=16,
        alpha=0.80,
        color="tab:red",
        label="Touchet",
    )
    axis.set_xlabel("Embedding PC1")
    axis.set_ylabel("Embedding PC2")
    axis.set_title(
        "Frozen LSTM embedding space "
        f"({100 * pca.explained_variance_ratio_.sum():.1f}% variance in PC1-PC2)"
    )
    axis.legend()
    axis.grid(alpha=0.20)
    figure.tight_layout()
    figure.savefig(output_dir / "embedding_pca_touchet.png", dpi=160)
    plt.close(figure)

    selected_indices = []
    for station in ("Touchet_WA_824", "Spokane"):
        candidates = np.flatnonzero(test_data.metadata["station_id"].to_numpy() == station)
        if len(candidates):
            selected_indices.append((station, int(candidates[len(candidates) // 2])))
    figure, axes = plt.subplots(len(selected_indices), 1, figsize=(10, 4 * len(selected_indices)))
    if len(selected_indices) == 1:
        axes = [axes]
    for axis, (station, index) in zip(axes, selected_indices):
        sequence = torch.from_numpy(test_data.X[index : index + 1]).to(device)
        trace = model.forward_with_trace(sequence)
        projected = trace["projected"].norm(dim=2).squeeze(0).cpu().numpy()
        recurrent = trace["recurrent"].norm(dim=2).squeeze(0).cpu().numpy()
        axis.plot(projected, marker="o", label="projection norm")
        axis.plot(recurrent, marker="o", label="LSTM hidden norm")
        axis.set_title(
            f"{station}, target={test_data.metadata.iloc[index]['date'].date()}, "
            f"true={test_data.y[index]:.3f}, pred={trace['prediction'].item():.3f}"
        )
        axis.set_xlabel("Historical timestep")
        axis.set_ylabel("Vector norm")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "hidden_vector_chain_examples.png", dpi=160)
    plt.close(figure)


def main() -> None:
    config = load_config()
    temporal_config = config["temporal_encoder"]
    splits, combined, texture_columns = prepare_source_splits()
    device = torch.device("cpu")
    print(f"[device] {device}")

    encoder_runs = {}
    for variant_name, features in temporal_feature_variants().items():
        encoder_runs[variant_name] = train_encoder_experiment(
            variant_name, combined, features, config, device
        )

    encoder_summary = []
    for variant_name, run in encoder_runs.items():
        for split_name in ("val", "test"):
            encoder_summary.append(
                {
                    "variant": variant_name,
                    "split": split_name,
                    **run["metrics"][split_name]["overall"],
                }
            )
    encoder_summary_frame = pd.DataFrame(encoder_summary)
    encoder_summary_frame.to_csv(
        ARTIFACT_DIR / "hybrid" / "encoder_comparison.csv", index=False
    )
    selected_name = (
        encoder_summary_frame[encoder_summary_frame["split"] == "val"]
        .sort_values("rmse")
        .iloc[0]["variant"]
    )
    selected = encoder_runs[selected_name]
    print(f"[selected encoder] {selected_name}")

    all_source = pd.concat(splits.values(), ignore_index=True)
    conventional_frames = {}
    embedding_columns = None
    for split_name in ("train", "val", "test"):
        frame, columns = attach_embeddings(
            all_source,
            selected["datasets"][split_name],
            selected["predictions"][split_name],
            selected["embeddings"][split_name],
        )
        conventional_frames[split_name] = frame
        embedding_columns = columns
        embedding_export = frame[
            ID_COLUMNS
            + ["window_start", "window_end", "window_span_days", "lstm_scalar_prediction"]
            + columns
        ]
        export_dir = ARTIFACT_DIR / "hybrid" / "conventional_full_train" / "embeddings"
        export_dir.mkdir(parents=True, exist_ok=True)
        embedding_export.to_csv(export_dir / f"{split_name}.csv", index=False)

    feature_summary = pd.read_csv(
        ARTIFACT_DIR / "feature_experiments" / "legacy_five" / "summary.csv"
    )
    selected_tabular_name = (
        feature_summary[feature_summary["split"] == "val"]
        .sort_values("rmse")
        .iloc[0]["variant"]
    )
    selected_tabular_features = feature_variants(texture_columns)[selected_tabular_name]
    print(f"[selected tabular] {selected_tabular_name}")
    conventional_results = run_tree_suite(
        "conventional_full_train",
        conventional_frames,
        embedding_columns,
        selected_tabular_features,
    )

    years = combined["date"].dt.year
    strict_selection_fit = (combined["_split"] == "train") & (years <= 2017)
    strict_selection_masks = {
        "train": (combined["_split"] == "train") & (years <= 2017),
        "val": (combined["_split"] == "train") & (years == 2018),
    }
    strict_selection_preprocessor, strict_selection_data = prepare_temporal_data(
        combined,
        selected["features"],
        fit_mask=strict_selection_fit,
        target_masks=strict_selection_masks,
        sequence_length=temporal_config["sequence_length"],
        max_span_factor=temporal_config["max_window_span_factor"],
    )
    del strict_selection_preprocessor
    print("[strict] selecting epoch on 2017 -> 2018")
    _, strict_history, strict_best_epoch = train_model(
        strict_selection_data["train"],
        strict_selection_data["val"],
        n_features=len(selected["features"]),
        projection_size=temporal_config["projection_size"],
        hidden_size=temporal_config["hidden_size"],
        embedding_size=temporal_config["embedding_size"],
        dropout=temporal_config["dropout"],
        batch_size=temporal_config["batch_size"],
        learning_rate=temporal_config["learning_rate"],
        weight_decay=temporal_config["weight_decay"],
        max_epochs=temporal_config["max_epochs"],
        patience=temporal_config["patience"],
        huber_delta=temporal_config["huber_delta"],
        seed=SEED,
        device=device,
    )
    save_training_curve(
        strict_history,
        ARTIFACT_DIR / "hybrid" / "strict_frozen" / "epoch_selection_curve.png",
        "Strict encoder epoch selection: 2017 to 2018",
    )

    strict_fit_mask = (combined["_split"] == "train") & (years <= 2018)
    strict_masks = {
        "encoder_train": (combined["_split"] == "train") & (years <= 2018),
        "train": (combined["_split"] == "train") & (years >= 2019),
        "val": combined["_split"] == "val",
        "test": combined["_split"] == "test",
    }
    strict_preprocessor, strict_data = prepare_temporal_data(
        combined,
        selected["features"],
        fit_mask=strict_fit_mask,
        target_masks=strict_masks,
        sequence_length=temporal_config["sequence_length"],
        max_span_factor=temporal_config["max_window_span_factor"],
    )
    print(f"[strict] refitting 2017-2018 for {strict_best_epoch} epochs")
    strict_model, strict_refit_history, _ = train_model(
        strict_data["encoder_train"],
        None,
        n_features=len(selected["features"]),
        projection_size=temporal_config["projection_size"],
        hidden_size=temporal_config["hidden_size"],
        embedding_size=temporal_config["embedding_size"],
        dropout=temporal_config["dropout"],
        batch_size=temporal_config["batch_size"],
        learning_rate=temporal_config["learning_rate"],
        weight_decay=temporal_config["weight_decay"],
        max_epochs=strict_best_epoch,
        patience=temporal_config["patience"],
        huber_delta=temporal_config["huber_delta"],
        seed=SEED,
        device=device,
        fixed_epochs=strict_best_epoch,
    )
    strict_output = ARTIFACT_DIR / "hybrid" / "strict_frozen"
    strict_output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": strict_model.state_dict(),
            "features": selected["features"],
            "best_epoch": strict_best_epoch,
            "encoder_train_years": [2017, 2018],
            "stacker_train_years": [2019, 2020],
        },
        strict_output / "encoder.pt",
    )
    joblib.dump(strict_preprocessor, strict_output / "preprocessor.joblib")

    strict_frames = {}
    strict_embeddings = {}
    for split_name in ("train", "val", "test"):
        predictions, embeddings = predict_and_embed(
            strict_model,
            strict_data[split_name],
            batch_size=temporal_config["batch_size"],
            device=device,
        )
        strict_embeddings[split_name] = embeddings
        frame, strict_embedding_columns = attach_embeddings(
            all_source,
            strict_data[split_name],
            predictions,
            embeddings,
        )
        strict_frames[split_name] = frame
        export_dir = strict_output / "embeddings"
        export_dir.mkdir(parents=True, exist_ok=True)
        frame[
            ID_COLUMNS
            + ["window_start", "window_end", "window_span_days", "lstm_scalar_prediction"]
            + strict_embedding_columns
        ].to_csv(export_dir / f"{split_name}.csv", index=False)
    strict_results = run_tree_suite(
        "strict_frozen",
        strict_frames,
        strict_embedding_columns,
        selected_tabular_features,
    )

    trace_representation(
        selected["model"],
        selected["datasets"]["train"],
        selected["datasets"]["test"],
        selected["embeddings"]["train"],
        selected["embeddings"]["test"],
        device=device,
    )

    summary_payload = {
        "selected_encoder": selected_name,
        "selected_tabular_variant": selected_tabular_name,
        "encoder_comparison": encoder_summary,
        "strict_best_epoch": strict_best_epoch,
        "conventional_full_train": conventional_results,
        "strict_frozen": strict_results,
    }
    save_json(summary_payload, ARTIFACT_DIR / "hybrid" / "metrics.json")
    print("\nEncoder comparison")
    print(encoder_summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
