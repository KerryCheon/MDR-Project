from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mdr-mpl")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

TARGET = "soil_moisture_5cm"
ID_COLUMNS = ["station_id", "date"]


@dataclass(frozen=True)
class Candidate:
    name: str
    version: str
    architecture: str
    features: tuple[str, ...]
    seq_len: int
    learning_rate: float
    weight_decay: float
    model_factory: Callable[[int], nn.Module]
    target_mode: str = "direct"
    transform: str = "none"
    aux_weight: float = 0.0


@dataclass
class CandidateRun:
    candidate: str
    seed: int
    best_epoch: int
    epochs_run: int
    parameter_count: int
    representation_dimensions: int
    lstm_validation_r2: float
    lstm_validation_rmse: float
    checkpoint: str
    train_representations: np.ndarray
    validation_representations: np.ndarray
    test_representations: np.ndarray | None = None


def load_config() -> dict:
    return yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text(encoding="utf-8"))


def artifact_dir(smoke: bool) -> Path:
    return EXPERIMENT_DIR / ("artifacts_smoke" if smoke else "artifacts")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def load_raw_splits(config: dict) -> dict[str, pd.DataFrame]:
    root = REPO_ROOT / config["data"]["directory"]
    output = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(root / f"{split}.csv")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.assign(_row_id=np.arange(len(frame), dtype=int))
        if frame.duplicated(ID_COLUMNS).any():
            raise ValueError(f"Duplicate station/date rows in {split}")
        if frame[TARGET].isna().any():
            raise ValueError(f"Missing target rows in {split}")
        output[split] = frame
    return output


def _load_metadata_features(path: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("hybrid_sweep_metadata", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.OVERALL_SELECTED_FEATURES_V0)


def _best_v20_features() -> list[str]:
    path = REPO_ROOT / "Models/Temporal/lstm/outputs_v20/metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["pruned_candidates"]["top25"]["feature_cols"])


def build_candidates() -> list[Candidate]:
    from Models.Temporal.lstm import (
        train_v7,
        train_v8,
        train_v9,
        train_v10,
        train_v11,
        train_v12,
        train_v13,
        train_v14,
        train_v15,
        train_v16,
        train_v17,
        train_v20,
        train_v21,
        train_v23_raw_vs_engineered,
    )

    v20_features = _best_v20_features()

    def bilstm_factory(n_features: int):
        return train_v9.BiLSTMAttn(
            n_features=n_features,
            hidden_size=train_v9.HIDDEN_SIZE,
            num_layers=train_v9.NUM_LAYERS,
            dropout=train_v9.DROPOUT,
            proj_size=train_v9.PROJ_SIZE,
        )

    return [
        Candidate(
            "v7_pruned_lstm", "v7", "two-layer LSTM, last state",
            tuple(train_v7.ALL_FEATURES), train_v7.SEQ_LEN, train_v7.LR,
            train_v7.WEIGHT_DECAY,
            lambda n: train_v7.LSTMv7(n, train_v7.HIDDEN_SIZE, train_v7.NUM_LAYERS, train_v7.DROPOUT, train_v7.PROJ_SIZE),
        ),
        Candidate(
            "v8_feature_attention", "v8", "feature-attention LSTM",
            tuple(train_v8.ALL_FEATURES), train_v8.SEQ_LEN, train_v8.LR,
            train_v8.WEIGHT_DECAY,
            lambda n: train_v8.FALSTM(n, train_v8.HIDDEN_SIZE, train_v8.NUM_LAYERS, train_v8.DROPOUT, train_v8.ATTN_HIDDEN),
        ),
        Candidate(
            "v9_bilstm_attention", "v9", "BiLSTM plus additive attention",
            tuple(train_v9.ALL_FEATURES), train_v9.SEQ_LEN, train_v9.LR,
            train_v9.WEIGHT_DECAY, bilstm_factory,
        ),
        Candidate(
            "v10_regularized_bilstm", "v10", "regularized BiLSTM plus attention",
            tuple(train_v10.ALL_FEATURES), train_v10.SEQ_LEN, train_v10.LR,
            train_v10.WEIGHT_DECAY,
            lambda n: train_v10.BiLSTMAttnReg(n, train_v10.HIDDEN_SIZE, train_v10.NUM_LAYERS, train_v10.DROPOUT, train_v10.PROJ_SIZE, train_v10.INPUT_NOISE),
        ),
        Candidate(
            "v11_feature_multihead", "v11", "feature attention plus multi-head temporal attention",
            tuple(train_v11.ALL_FEATURES), train_v11.SEQ_LEN, train_v11.LR,
            train_v11.WEIGHT_DECAY,
            lambda n: train_v11.BiLSTMMultiHead(n, train_v11.HIDDEN_SIZE, train_v11.NUM_LAYERS, train_v11.DROPOUT, train_v11.PROJ_SIZE, train_v11.N_HEADS, train_v11.ATTN_HIDDEN),
        ),
        Candidate(
            "v12_small_bilstm", "v12", "small one-layer BiLSTM plus attention",
            tuple(train_v12.ALL_FEATURES), train_v12.SEQ_LEN, 1.0e-3,
            train_v12.WEIGHT_DECAY,
            lambda n: train_v12.SmallBiLSTMAttn(n, 40, 0.35, train_v12.PROJ_SIZE, train_v12.NUM_LAYERS),
        ),
        Candidate(
            "v13_residual_smap", "v13", "BiLSTM trained on residual over SMAP",
            tuple(train_v13.ALL_FEATURES), train_v13.SEQ_LEN, train_v13.LR,
            train_v13.WEIGHT_DECAY,
            lambda n: train_v13.BiLSTMAttn(n, train_v13.HIDDEN_SIZE, train_v13.NUM_LAYERS, train_v13.DROPOUT, train_v13.PROJ_SIZE),
            target_mode="residual_smap",
        ),
        Candidate(
            "v14_multiscale_bilstm", "v14", "parallel 5/10/20-day BiLSTMs",
            tuple(train_v14.ALL_FEATURES), train_v14.SEQ_LEN, train_v14.LR,
            train_v14.WEIGHT_DECAY,
            lambda n: train_v14.MultiScaleBiLSTM(n, train_v14.HIDDEN_SIZE, train_v14.PROJ_SIZE, train_v14.DROPOUT, train_v14.WINDOWS, train_v14.NUM_LAYERS, train_v14.HEAD_HIDDEN),
        ),
        Candidate(
            "v15_bigru_multitask", "v15", "BiGRU attention with auxiliary SMAP head",
            tuple(train_v15.ALL_FEATURES), train_v15.SEQ_LEN, train_v15.LR,
            train_v15.WEIGHT_DECAY,
            lambda n: train_v15.BiGRUAttnMultiTask(n, train_v15.HIDDEN_SIZE, train_v15.NUM_LAYERS, train_v15.DROPOUT, train_v15.PROJ_SIZE),
            aux_weight=train_v15.AUX_WEIGHT,
        ),
        Candidate(
            "v16_engineered_lags", "v16", "BiLSTM attention with engineered causal lags",
            tuple(train_v16.ALL_FEATURES), train_v16.SEQ_LEN, train_v16.LR,
            train_v16.WEIGHT_DECAY,
            lambda n: train_v16.BiLSTMAttn(n, train_v16.HIDDEN_SIZE, train_v16.NUM_LAYERS, train_v16.DROPOUT, train_v16.PROJ_SIZE),
            transform="v16_engineering",
        ),
        Candidate(
            "v17_station_demean", "v17", "BiLSTM attention with station-demeaned target",
            tuple(train_v17.ALL_FEATURES), train_v17.SEQ_LEN, train_v17.LR,
            train_v17.WEIGHT_DECAY,
            lambda n: train_v17.BiLSTMAttn(n, train_v17.HIDDEN_SIZE, train_v17.NUM_LAYERS, train_v17.DROPOUT, train_v17.PROJ_SIZE),
            target_mode="station_demean",
        ),
        Candidate(
            "v20_top25_seq30", "v20", "feature-aligned BiLSTM attention",
            tuple(v20_features), 30, train_v20.LR, train_v20.WEIGHT_DECAY,
            bilstm_factory,
        ),
        Candidate(
            "v21_full58_seq30", "v21", "union-feature BiLSTM attention used by the 0.834 hybrid",
            tuple(train_v21.ALL_FEATURES), 30, train_v20.LR,
            train_v20.WEIGHT_DECAY, bilstm_factory,
        ),
        Candidate(
            "v22_top25_smap_norm", "v22", "v20 encoder with causal SMAP normalization",
            tuple(v20_features), 30, train_v20.LR, train_v20.WEIGHT_DECAY,
            bilstm_factory, transform="v22_smap_norm",
        ),
        Candidate(
            "v23_raw_seq10", "v23", "coverage-fixed raw-input BiLSTM attention",
            tuple(train_v23_raw_vs_engineered.FEATURE_COLS), 10,
            train_v20.LR, train_v20.WEIGHT_DECAY, bilstm_factory,
        ),
    ]


def _transform_splits(candidate: Candidate, splits: dict[str, pd.DataFrame]):
    output = {name: frame.copy() for name, frame in splits.items()}
    if candidate.transform == "v16_engineering":
        from Models.Temporal.lstm.train_v16 import engineer_features

        output = {name: engineer_features(frame).assign(_row_id=np.arange(len(frame))) for name, frame in output.items()}
    elif candidate.transform == "v22_smap_norm":
        from Models.Temporal.lstm.train_v22_smap_norm import smap_rolling_normalize_splits

        train, val, test, _ = smap_rolling_normalize_splits(
            output["train"].drop(columns="_row_id"),
            output["val"].drop(columns="_row_id"),
            output["test"].drop(columns="_row_id"),
            list(candidate.features),
            window=30,
            min_periods=5,
        )
        output = {
            "train": train.assign(_row_id=np.arange(len(train))),
            "val": val.assign(_row_id=np.arange(len(val))),
            "test": test.assign(_row_id=np.arange(len(test))),
        }
    return output


def _preprocess_features(candidate: Candidate, splits: dict[str, pd.DataFrame]):
    missing = {
        split: sorted(set(candidate.features) - set(frame.columns))
        for split, frame in splits.items()
    }
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise ValueError(f"{candidate.name} missing features: {missing}")
    train_values = splits["train"][list(candidate.features)].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    scaler.fit(imputer.fit_transform(train_values))
    output = {}
    for name, frame in splits.items():
        clean = frame[list(candidate.features)].replace([np.inf, -np.inf], np.nan)
        values = np.clip(scaler.transform(imputer.transform(clean)), -5.0, 5.0)
        transformed = frame.drop(columns=list(candidate.features)).copy()
        feature_frame = pd.DataFrame(
            values,
            columns=list(candidate.features),
            index=frame.index,
        )
        transformed = pd.concat([transformed, feature_frame], axis=1)
        output[name] = transformed
    return output


def _target_values(candidate: Candidate, splits: dict[str, pd.DataFrame]):
    targets = {}
    station_means = splits["train"].groupby("station_id")[TARGET].mean().to_dict()
    for name, frame in splits.items():
        values = frame[TARGET].to_numpy(dtype=np.float32)
        if candidate.target_mode == "residual_smap":
            values = values - frame["SMAP_sm_pm_interp"].to_numpy(dtype=np.float32)
        elif candidate.target_mode == "station_demean":
            offsets = frame["station_id"].map(station_means).fillna(0.0).to_numpy(dtype=np.float32)
            values = values - offsets
        targets[name] = values
    return targets, station_means


def _build_arrays(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    seq_len: int,
    targets: np.ndarray,
):
    sequences, primary, auxiliary, row_ids = [], [], [], []
    target_series = pd.Series(targets, index=frame.index)
    for _, group in frame.groupby("station_id", sort=False):
        group = group.sort_values("date")
        values = group[list(features)].to_numpy(dtype=np.float32)
        group_targets = target_series.loc[group.index].to_numpy(dtype=np.float32)
        group_aux = group["SMAP_sm_pm_interp"].to_numpy(dtype=np.float32)
        group_rows = group["_row_id"].to_numpy(dtype=int)
        for index in range(len(group)):
            start = max(0, index - seq_len + 1)
            window = values[start : index + 1]
            if len(window) < seq_len:
                window = np.concatenate(
                    [np.repeat(window[:1], seq_len - len(window), axis=0), window],
                    axis=0,
                )
            sequences.append(window)
            primary.append(group_targets[index])
            auxiliary.append(group_aux[index])
            row_ids.append(group_rows[index])
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(primary, dtype=np.float32),
        np.asarray(auxiliary, dtype=np.float32),
        np.asarray(row_ids, dtype=int),
    )


def _primary_prediction(output):
    return output[0] if isinstance(output, tuple) else output


def _final_primary_linear(model: nn.Module) -> nn.Linear:
    if hasattr(model, "head_primary"):
        head = model.head_primary
    else:
        head = model.head
    linear_layers = [layer for layer in head.modules() if isinstance(layer, nn.Linear)]
    if not linear_layers:
        raise ValueError(f"No prediction linear layer in {type(model).__name__}")
    return linear_layers[-1]


def _reconstruct_predictions(
    candidate: Candidate,
    frame: pd.DataFrame,
    transformed_prediction: np.ndarray,
    station_means: dict,
) -> np.ndarray:
    if candidate.target_mode == "residual_smap":
        return transformed_prediction + frame["SMAP_sm_pm_interp"].to_numpy(dtype=float)
    if candidate.target_mode == "station_demean":
        offsets = frame["station_id"].map(station_means).fillna(0.0).to_numpy(dtype=float)
        return transformed_prediction + offsets
    return transformed_prediction


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred) - np.asarray(y_true)
    bias = float(np.mean(error))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "ubrmse": float(math.sqrt(max(0.0, rmse * rmse - bias * bias))),
        "bias": bias,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def _ordered(values: np.ndarray, row_ids: np.ndarray, n_rows: int) -> np.ndarray:
    result = np.empty((n_rows,) + values.shape[1:], dtype=values.dtype)
    result[row_ids] = values
    return result


@torch.no_grad()
def _predict_and_represent(model: nn.Module, loader: DataLoader, device: torch.device):
    predictions, representations, row_ids = [], [], []
    captured = []

    def hook(_module, inputs):
        captured.append(inputs[0].detach().cpu().numpy())

    handle = _final_primary_linear(model).register_forward_pre_hook(hook)
    model.eval()
    try:
        for sequence, _primary, _auxiliary, rows in loader:
            output = model(sequence.to(device))
            predictions.append(_primary_prediction(output).detach().cpu().numpy())
            representations.append(captured.pop(0))
            row_ids.append(rows.numpy())
    finally:
        handle.remove()
    return (
        np.concatenate(predictions),
        np.concatenate(representations),
        np.concatenate(row_ids),
    )


def train_candidate(
    candidate: Candidate,
    seed: int,
    config: dict,
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    include_test: bool,
    max_epochs: int | None = None,
) -> CandidateRun:
    threads = int(config["training"]["num_threads"])
    set_seed(seed, threads)
    device = torch.device("cpu")
    transformed = _preprocess_features(candidate, _transform_splits(candidate, splits))
    targets, station_means = _target_values(candidate, transformed)
    arrays = {
        name: _build_arrays(frame, candidate.features, candidate.seq_len, targets[name])
        for name, frame in transformed.items()
    }
    batch_size = int(config["protocol"]["batch_size"])

    def loader(name: str, shuffle: bool):
        x, y, aux, rows = arrays[name]
        dataset = TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(aux), torch.from_numpy(rows),
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

    model = candidate.model_factory(len(candidate.features)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate.learning_rate, weight_decay=candidate.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"]["plateau_factor"]),
        patience=int(config["training"]["plateau_patience"]),
    )
    loss_fn = nn.HuberLoss(delta=float(config["training"]["huber_delta"]))
    patience = int(config["protocol"]["patience"])
    epochs = int(max_epochs or config["protocol"]["max_epochs"])
    checkpoint_dir = output_dir / "models" / candidate.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"seed{seed}.pt"
    best_rmse, best_epoch, stale = float("inf"), 0, 0

    checkpoint_loaded = False
    if checkpoint.exists() and max_epochs is None:
        try:
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            checkpoint_loaded = True
            best_epoch = _previous_best_epoch(output_dir, candidate.name, seed)
            epoch = 0
            print(f"  [checkpoint] reused {candidate.name} seed={seed}", flush=True)
        except RuntimeError:
            model = candidate.model_factory(len(candidate.features)).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=candidate.learning_rate,
                weight_decay=candidate.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(config["training"]["plateau_factor"]),
                patience=int(config["training"]["plateau_patience"]),
            )

    if not checkpoint_loaded:
        for epoch in range(1, epochs + 1):
            model.train()
            for sequence, primary, auxiliary, _rows in loader("train", True):
                optimizer.zero_grad(set_to_none=True)
                output = model(sequence.to(device))
                primary_prediction = _primary_prediction(output)
                loss = loss_fn(primary_prediction, primary.to(device))
                if isinstance(output, tuple) and candidate.aux_weight:
                    loss = loss + candidate.aux_weight * loss_fn(output[1], auxiliary.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
                optimizer.step()

            val_prediction, _val_repr, val_rows = _predict_and_represent(model, loader("val", False), device)
            val_prediction = _ordered(val_prediction, val_rows, len(transformed["val"]))
            reconstructed = _reconstruct_predictions(candidate, transformed["val"], val_prediction, station_means)
            val_rmse = float(root_mean_squared_error(transformed["val"][TARGET], reconstructed))
            scheduler.step(val_rmse)
            if val_rmse < best_rmse - 1.0e-7:
                best_rmse, best_epoch, stale = val_rmse, epoch, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= patience:
                    break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    train_prediction, train_repr, train_rows = _predict_and_represent(model, loader("train", False), device)
    val_prediction, val_repr, val_rows = _predict_and_represent(model, loader("val", False), device)
    train_repr = _ordered(train_repr, train_rows, len(transformed["train"]))
    val_repr = _ordered(val_repr, val_rows, len(transformed["val"]))
    val_prediction = _ordered(val_prediction, val_rows, len(transformed["val"]))
    reconstructed = _reconstruct_predictions(candidate, transformed["val"], val_prediction, station_means)
    validation_metrics = metrics(transformed["val"][TARGET].to_numpy(), reconstructed)
    test_repr = None
    if include_test:
        _test_prediction, test_repr_raw, test_rows = _predict_and_represent(model, loader("test", False), device)
        test_repr = _ordered(test_repr_raw, test_rows, len(transformed["test"]))

    run = CandidateRun(
        candidate=candidate.name,
        seed=seed,
        best_epoch=best_epoch,
        epochs_run=epoch,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        representation_dimensions=int(train_repr.shape[1]),
        lstm_validation_r2=validation_metrics["r2"],
        lstm_validation_rmse=validation_metrics["rmse"],
        checkpoint=str(checkpoint.relative_to(REPO_ROOT)),
        train_representations=train_repr,
        validation_representations=val_repr,
        test_representations=test_repr,
    )
    return run


def _previous_best_epoch(output_dir: Path, candidate: str, seed: int) -> int:
    confirmation_path = output_dir / "confirmation_runs.csv"
    if confirmation_path.exists():
        confirmation = pd.read_csv(confirmation_path)
        match = confirmation.loc[
            confirmation["candidate"].eq(candidate) & confirmation["seed"].eq(seed)
        ]
        if not match.empty:
            return int(match.iloc[0]["best_epoch"])
    screen_path = output_dir / "screen_leaderboard.csv"
    if seed == 42 and screen_path.exists():
        screen = pd.read_csv(screen_path)
        match = screen.loc[screen["candidate"].eq(candidate)]
        if not match.empty:
            return int(match.iloc[0]["best_epoch"])
    return -1


def _router_labels(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    router_features: list[str],
    seed: int,
):
    means = train[router_features].replace([np.inf, -np.inf], np.nan).mean()
    train_values = train[router_features].replace([np.inf, -np.inf], np.nan).fillna(means)
    eval_values = evaluation[router_features].replace([np.inf, -np.inf], np.nan).fillna(means)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values)
    kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
    train_labels = kmeans.fit_predict(train_scaled)
    eval_labels = kmeans.predict(scaler.transform(eval_values))
    return train_labels, eval_labels


def fit_two_regime(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    train_representation: np.ndarray | None,
    evaluation_representation: np.ndarray | None,
    config: dict,
    router_features: list[str],
    *,
    seed: int,
    smoke: bool,
):
    backbone = list(config["shared_backbone_54"])
    additions = list(config["two_regime"]["cluster_1_additions"])
    train_work = train.copy()
    eval_work = evaluation.copy()
    pca_components = 0
    if train_representation is not None:
        requested = int(config["protocol"]["representation_components"])
        pca_components = min(requested, train_representation.shape[1], len(train_representation))
        pca = PCA(n_components=pca_components, random_state=seed)
        train_reduced = pca.fit_transform(train_representation)
        eval_reduced = pca.transform(evaluation_representation)
        columns = [f"lstm_pc_{index:02d}" for index in range(pca_components)]
        train_work[columns] = train_reduced
        eval_work[columns] = eval_reduced
        backbone += columns

    train_labels, eval_labels = _router_labels(
        train_work, eval_work, router_features, int(config["two_regime"]["router_seed"]),
    )
    y_train = train_work[TARGET].to_numpy(dtype=float)
    predictions = np.empty(len(eval_work), dtype=float)
    params = dict(config["two_regime"]["exact_xgboost_params"])
    if smoke:
        params["n_estimators"] = 20
    params["random_state"] = seed
    for cluster in (0, 1):
        train_mask = train_labels == cluster
        eval_mask = eval_labels == cluster
        features = list(dict.fromkeys(backbone + (additions if cluster == 1 else [])))
        model = XGBRegressor(**params)
        model.fit(train_work.loc[train_mask, features], y_train[train_mask], verbose=False)
        predictions[eval_mask] = model.predict(eval_work.loc[eval_mask, features])
    result = metrics(eval_work[TARGET].to_numpy(dtype=float), predictions)
    result["pca_components"] = pca_components
    result["cluster_0_train_n"] = int(np.sum(train_labels == 0))
    result["cluster_1_train_n"] = int(np.sum(train_labels == 1))
    return result, predictions


def candidate_manifest(candidates: list[Candidate]) -> list[dict]:
    return [
        {
            "name": candidate.name,
            "version": candidate.version,
            "architecture": candidate.architecture,
            "feature_count": len(candidate.features),
            "sequence_length": candidate.seq_len,
            "target_mode": candidate.target_mode,
            "transform": candidate.transform,
        }
        for candidate in candidates
    ]


def run_experiment(*, smoke: bool = False) -> dict:
    config = load_config()
    output_dir = artifact_dir(smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_raw_splits(config)
    candidates = build_candidates()
    if smoke:
        candidates = candidates[:2]
    router_features = _load_metadata_features(REPO_ROOT / config["data"]["metadata_path"])
    screen_seed = int(config["protocol"]["screen_seed"])
    screen_rows, runs = [], {}
    screen_epochs = 2 if smoke else None

    baseline_validation, _ = fit_two_regime(
        splits["train"], splits["val"], None, None, config, router_features,
        seed=screen_seed, smoke=smoke,
    )

    for index, candidate in enumerate(candidates, start=1):
        print(f"[screen {index}/{len(candidates)}] {candidate.name}", flush=True)
        run = train_candidate(
            candidate, screen_seed, config, splits, output_dir,
            include_test=False, max_epochs=screen_epochs,
        )
        runs[(candidate.name, screen_seed)] = run
        hybrid, _ = fit_two_regime(
            splits["train"], splits["val"],
            run.train_representations, run.validation_representations,
            config, router_features, seed=screen_seed, smoke=smoke,
        )
        screen_rows.append({
            "candidate": candidate.name,
            "version": candidate.version,
            "architecture": candidate.architecture,
            "feature_count": len(candidate.features),
            "sequence_length": candidate.seq_len,
            "target_mode": candidate.target_mode,
            "parameter_count": run.parameter_count,
            "representation_dimensions": run.representation_dimensions,
            "best_epoch": run.best_epoch,
            "lstm_validation_r2": run.lstm_validation_r2,
            "lstm_validation_rmse": run.lstm_validation_rmse,
            "hybrid_validation_r2": hybrid["r2"],
            "hybrid_validation_rmse": hybrid["rmse"],
            "hybrid_validation_ubrmse": hybrid["ubrmse"],
            "hybrid_validation_bias": hybrid["bias"],
            "delta_r2_over_two_regime": hybrid["r2"] - baseline_validation["r2"],
        })

    screen = pd.DataFrame(screen_rows).sort_values(
        ["hybrid_validation_r2", "hybrid_validation_rmse"], ascending=[False, True]
    )
    screen.to_csv(output_dir / "screen_leaderboard.csv", index=False)
    top_k = min(int(config["protocol"]["top_k_confirmation"]), len(screen))
    top_names = screen.head(top_k)["candidate"].tolist()
    seeds = [screen_seed] if smoke else list(config["protocol"]["confirmation_seeds"])
    confirmation_rows = []

    by_name = {candidate.name: candidate for candidate in candidates}
    for candidate_name in top_names:
        candidate = by_name[candidate_name]
        for seed in seeds:
            run = runs.get((candidate_name, seed))
            if run is None:
                print(f"[confirm] {candidate_name} seed={seed}", flush=True)
                run = train_candidate(
                    candidate, int(seed), config, splits, output_dir,
                    include_test=False, max_epochs=screen_epochs,
                )
                runs[(candidate_name, seed)] = run
            hybrid, _ = fit_two_regime(
                splits["train"], splits["val"],
                run.train_representations, run.validation_representations,
                config, router_features, seed=int(seed), smoke=smoke,
            )
            confirmation_rows.append({
                "candidate": candidate_name,
                "seed": int(seed),
                "hybrid_validation_r2": hybrid["r2"],
                "hybrid_validation_rmse": hybrid["rmse"],
                "lstm_validation_r2": run.lstm_validation_r2,
                "best_epoch": run.best_epoch,
            })

    confirmation = pd.DataFrame(confirmation_rows)
    confirmation.to_csv(output_dir / "confirmation_runs.csv", index=False)
    summary = (
        confirmation.groupby("candidate", as_index=False)
        .agg(
            mean_validation_r2=("hybrid_validation_r2", "mean"),
            std_validation_r2=("hybrid_validation_r2", "std"),
            mean_validation_rmse=("hybrid_validation_rmse", "mean"),
            seed_count=("seed", "count"),
        )
        .sort_values(["mean_validation_r2", "mean_validation_rmse"], ascending=[False, True])
    )
    summary["std_validation_r2"] = summary["std_validation_r2"].fillna(0.0)
    summary.to_csv(output_dir / "confirmation_summary.csv", index=False)
    winner_name = str(summary.iloc[0]["candidate"])
    winner = by_name[winner_name]

    trainval = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    test_predictions = []
    final_rows = []
    for seed in seeds:
        print(f"[final] {winner_name} seed={seed}", flush=True)
        run = train_candidate(
            winner, int(seed), config, splits, output_dir,
            include_test=True, max_epochs=screen_epochs,
        )
        trainval_representation = np.concatenate(
            [run.train_representations, run.validation_representations], axis=0
        )
        final_metric, prediction = fit_two_regime(
            trainval, splits["test"], trainval_representation,
            run.test_representations, config, router_features,
            seed=int(seed), smoke=smoke,
        )
        test_predictions.append(prediction)
        final_rows.append({"candidate": winner_name, "seed": int(seed), **final_metric})
    final = pd.DataFrame(final_rows)
    final.to_csv(output_dir / "winner_test_runs.csv", index=False)
    ensemble_prediction = np.mean(test_predictions, axis=0)
    ensemble_metrics = metrics(splits["test"][TARGET].to_numpy(dtype=float), ensemble_prediction)
    save_json(ensemble_metrics, output_dir / "winner_ensemble_metrics.json")

    baseline_test, _ = fit_two_regime(
        trainval, splits["test"], None, None, config, router_features,
        seed=screen_seed, smoke=smoke,
    )
    manifest = {
        "experiment": config["experiment"],
        "smoke": smoke,
        "candidate_count": len(candidates),
        "candidates": candidate_manifest(candidates),
        "selection_split": "validation",
        "test_evaluated_candidates": [winner_name, "two_regime_tabular_baseline"],
        "winner": winner_name,
        "baseline_validation": baseline_validation,
        "baseline_test": baseline_test,
        "winner_ensemble_test": ensemble_metrics,
        "state_free": True,
        "test_target_used_for_candidate_selection": False,
    }
    save_json(manifest, output_dir / "run_manifest.json")
    _write_results(output_dir, screen, summary, final, manifest)
    return manifest


def _write_results(
    output_dir: Path,
    screen: pd.DataFrame,
    confirmation: pd.DataFrame,
    final: pd.DataFrame,
    manifest: dict,
) -> None:
    winner = manifest["winner"]
    ensemble = manifest["winner_ensemble_test"]
    baseline = manifest["baseline_test"]
    screen_display = screen[
        ["candidate", "hybrid_validation_r2", "hybrid_validation_rmse", "delta_r2_over_two_regime", "lstm_validation_r2"]
    ].copy()
    confirmation_display = confirmation.copy()
    final_display = final[["candidate", "seed", "r2", "rmse", "ubrmse", "bias", "mae"]].copy()
    content = f"""# Results: derived_8.0 historical LSTM + two-regime sweep

The validation-selected winner is **{winner}**. Its test ensemble reaches **R² {ensemble['r2']:.6f}** and **RMSE {ensemble['rmse']:.6f}**, compared with the fixed tabular two-regime baseline at **R² {baseline['r2']:.6f}** and **RMSE {baseline['rmse']:.6f}**.

## One-seed validation screen

{screen_display.to_markdown(index=False, floatfmt='.6f')}

## Multi-seed validation confirmation

{confirmation_display.to_markdown(index=False, floatfmt='.6f')}

## Winner-only test runs

{final_display.to_markdown(index=False, floatfmt='.6f')}

## Interpretation

Selection used validation performance only. Test targets were read only after the winner was fixed, and no losing candidate received a test score. The LSTM remains state-free: neither the current nor previous in-situ target is an input. The fixed cluster-1 additions retain the provenance limitation documented in the experiment README.
"""
    (output_dir / "RESULTS.md").write_text(content, encoding="utf-8")
