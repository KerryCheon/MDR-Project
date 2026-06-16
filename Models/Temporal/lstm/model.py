"""
LSTM regressor for soil moisture prediction.
"""

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        proj_size: int = 64,
    ):
        super().__init__()     
        self.proj = nn.Sequential(
            nn.Linear(n_features, proj_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=proj_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        batch, seq, feat = x.shape
        # apply projection timestep-wise
        x = self.proj(x.view(batch * seq, feat)).view(batch, seq, -1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # take the last timestep
        return self.head(self.dropout(last)).squeeze(-1)   # (batch,)
