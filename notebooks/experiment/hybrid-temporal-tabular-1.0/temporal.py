from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from common import ID_COLUMNS, TARGET, regression_metrics


@dataclass
class SequenceData:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    excluded_gap_targets: int


class ArrayDataset(Dataset):
    def __init__(self, data: SequenceData):
        self.X = data.X
        self.y = data.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class TemporalPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()

    def fit(self, frame: pd.DataFrame, features: list[str]):
        values = frame[features].replace([np.inf, -np.inf], np.nan)
        values = self.imputer.fit_transform(values)
        self.scaler.fit(values)
        return self

    def transform(self, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        output = frame.copy()
        values = output[features].replace([np.inf, -np.inf], np.nan)
        values = self.imputer.transform(values)
        values = self.scaler.transform(values)
        output[features] = np.clip(values, -5.0, 5.0)
        return output


class FrozenLSTMExtractor(nn.Module):
    """LSTM regressor with an explicit reusable temporal embedding."""

    def __init__(
        self,
        n_features: int,
        projection_size: int = 48,
        hidden_size: int = 64,
        embedding_size: int = 64,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(n_features, projection_size),
            nn.ReLU(),
            nn.LayerNorm(projection_size),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.LayerNorm(embedding_size),
        )
        self.head = nn.Linear(embedding_size, 1)

    def encode(self, sequence: torch.Tensor) -> torch.Tensor:
        projected = self.projection(sequence)
        recurrent, (hidden, _) = self.lstm(projected)
        del recurrent
        final_hidden = hidden[-1]
        return self.embedding_layer(final_hidden)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(sequence)).squeeze(-1)

    def forward_with_trace(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.projection(sequence)
        recurrent, (hidden, cell) = self.lstm(projected)
        final_hidden = hidden[-1]
        embedding = self.embedding_layer(final_hidden)
        prediction = self.head(embedding).squeeze(-1)
        return {
            "input": sequence,
            "projected": projected,
            "recurrent": recurrent,
            "hidden": hidden,
            "cell": cell,
            "final_hidden": final_hidden,
            "embedding": embedding,
            "prediction": prediction,
        }


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_sequences(
    frame: pd.DataFrame,
    features: list[str],
    target_mask: pd.Series,
    *,
    sequence_length: int,
    max_span_factor: float = 2.0,
) -> SequenceData:
    working = frame.copy()
    working["_target_mask"] = target_mask.reindex(working.index).fillna(False).astype(bool)
    sequences = []
    targets = []
    metadata = []
    excluded = 0
    maximum_span = sequence_length * max_span_factor

    for station, group in working.groupby("station_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        values = group[features].to_numpy(dtype=np.float32)
        target_values = group[TARGET].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(group["date"])
        target_flags = group["_target_mask"].to_numpy(dtype=bool)

        for index in range(sequence_length, len(group)):
            if not target_flags[index]:
                continue
            span_days = int((dates.iloc[index] - dates.iloc[index - sequence_length]).days)
            if span_days > maximum_span:
                excluded += 1
                continue
            sequence = values[index - sequence_length : index]
            if not np.isfinite(sequence).all():
                raise ValueError(f"Non-finite sequence for {station} at {dates.iloc[index]}")
            sequences.append(sequence)
            targets.append(target_values[index])
            metadata.append(
                {
                    "station_id": station,
                    "date": dates.iloc[index],
                    "window_start": dates.iloc[index - sequence_length],
                    "window_end": dates.iloc[index - 1],
                    "window_span_days": span_days,
                }
            )

    if not sequences:
        raise ValueError("No temporal sequences were constructed")
    return SequenceData(
        X=np.stack(sequences),
        y=np.asarray(targets, dtype=np.float32),
        metadata=pd.DataFrame(metadata),
        excluded_gap_targets=excluded,
    )


def make_loader(
    data: SequenceData,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        ArrayDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


@torch.no_grad()
def predict_and_embed(
    model: FrozenLSTMExtractor,
    data: SequenceData,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    embeddings = []
    loader = make_loader(data, batch_size=batch_size, shuffle=False)
    for sequences, _ in loader:
        sequences = sequences.to(device)
        embedding = model.encode(sequences)
        prediction = model.head(embedding).squeeze(-1)
        predictions.append(prediction.cpu().numpy())
        embeddings.append(embedding.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(embeddings)


def train_model(
    train_data: SequenceData,
    val_data: SequenceData | None,
    *,
    n_features: int,
    projection_size: int,
    hidden_size: int,
    embedding_size: int,
    dropout: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    huber_delta: float,
    seed: int,
    device: torch.device,
    fixed_epochs: int | None = None,
) -> tuple[FrozenLSTMExtractor, list[dict], int]:
    set_seed(seed)
    model = FrozenLSTMExtractor(
        n_features=n_features,
        projection_size=projection_size,
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    criterion = nn.HuberLoss(delta=huber_delta)
    train_loader = make_loader(train_data, batch_size=batch_size, shuffle=True)

    epochs_to_run = fixed_epochs if fixed_epochs is not None else max_epochs
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_rmse = math.inf
    patience_counter = 0
    history = []

    for epoch in range(1, epochs_to_run + 1):
        model.train()
        total_loss = 0.0
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
        train_loss = total_loss / len(train_data.y)

        row = {"epoch": epoch, "train_loss": train_loss}
        if val_data is not None:
            val_predictions, _ = predict_and_embed(
                model, val_data, batch_size=batch_size, device=device
            )
            val_metrics = regression_metrics(val_data.y, val_predictions)
            row.update(
                {
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                }
            )
            print(
                f"[epoch {epoch:03d}] train_huber={train_loss:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} val_r2={val_metrics['r2']:.4f}"
            )
            if val_metrics["rmse"] < best_val_rmse - 1e-7:
                best_val_rmse = val_metrics["rmse"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if fixed_epochs is None and patience_counter >= patience:
                    break
        else:
            print(f"[epoch {epoch:03d}] train_huber={train_loss:.6f}")
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(row)

    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def save_training_curve(history: list[dict], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train Huber")
    axes[0].set_ylabel("Train loss")
    if history and "val_rmse" in history[0]:
        axes[1].plot(epochs, [row["val_rmse"] for row in history], label="val RMSE")
    axes[1].set_ylabel("Validation RMSE")
    axes[1].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def sequence_prediction_frame(
    data: SequenceData,
    predictions: np.ndarray,
) -> pd.DataFrame:
    output = data.metadata[ID_COLUMNS + ["window_start", "window_end", "window_span_days"]].copy()
    output["y_true"] = data.y
    output["y_pred"] = predictions
    return output

