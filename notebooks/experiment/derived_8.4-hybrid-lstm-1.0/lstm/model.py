"""
BiLSTM + Temporal Attention Pooling PyTorch Model.
Exposes context vector `ctx` alongside/before the final prediction head when `return_ctx=True`.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class BiLSTMAttn(nn.Module):
    """BiLSTM with additive attention pooling over the time axis."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 80,
        num_layers: int = 2,
        dropout: float = 0.3,
        proj_size: int = 56,
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
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        bi = hidden_size * 2
        self.attn = nn.Sequential(
            nn.Linear(bi, bi),
            nn.Tanh(),
            nn.Linear(bi, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(bi, bi // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bi // 2, 1),
        )

    def forward(self, x: torch.Tensor, return_ctx: bool = False):
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, S, F)
            return_ctx: If True, return tuple (pred, ctx) where ctx shape is (B, 2H=160).
        """
        b, s, f = x.shape
        x_proj = self.proj(x.reshape(b * s, f)).reshape(b, s, -1)
        out, _ = self.lstm(x_proj)              # (B, S, 2H)
        scores = self.attn(out).squeeze(-1)    # (B, S)
        weights = torch.softmax(scores, dim=-1)
        ctx = (out * weights.unsqueeze(-1)).sum(dim=1)   # (B, 2H)
        out_pred = self.head(self.dropout(ctx)).squeeze(-1)
        if return_ctx:
            return out_pred, ctx
        return out_pred
