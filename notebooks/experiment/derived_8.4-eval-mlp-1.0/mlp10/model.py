"""MLP regressor for tabular soil-moisture modeling (derived_8.4-eval-mlp-1.0)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "silu":
        return nn.SiLU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class MLPRegressor(nn.Module):
    """Feed-forward MLP regressor.

    Architecture: input -> [Linear -> BatchNorm1d -> act -> Dropout]^L -> Linear(1).
    Batch norm is applied after the first hidden layer (not on the raw input).

    Parameters mirror the eval-1.1 experiment naming (seed fixed at construction
    time so every config is reproducible given the same seed).
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int] | tuple[int, ...],
        activation: str = "silu",
        dropout: float = 0.1,
        use_bn: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.activation = activation
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)

        torch.manual_seed(int(seed))
        layers: list[nn.Module] = []
        in_dim = self.n_features
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if self.use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_make_activation(activation))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
