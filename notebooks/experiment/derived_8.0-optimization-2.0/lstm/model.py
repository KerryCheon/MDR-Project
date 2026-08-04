"""
BiLSTM + Temporal Attention Pooling PyTorch Model.
Exposes context vector `ctx` and/or head intermediate `head_hidden` alongside prediction.
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

    def forward(
        self,
        x: torch.Tensor,
        return_ctx: bool = False,
        return_head_hidden: bool = False,
        return_head_pre_relu: bool = False,
    ):
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, S, F)
            return_ctx: If True, include ctx (B, 2H=160) in output.
            return_head_hidden: If True, include head intermediate (B, H=80)
                after Linear(160→80)→ReLU in output.
            return_head_pre_relu: If True, include head intermediate (B, H=80)
                after Linear(160→80) BEFORE ReLU in output.
        Returns:
            pred only:                               Tensor (B,)
            return_ctx only:                         tuple (pred, ctx)
            return_head_hidden only:                 tuple (pred, head_hidden)
            return_head_pre_relu only:               tuple (pred, head_pre_relu)
            return_ctx and return_head_hidden:       tuple (pred, ctx, head_hidden)
            return_ctx and return_head_pre_relu:     tuple (pred, ctx, head_pre_relu)
        """
        b, s, f = x.shape
        x_proj = self.proj(x.reshape(b * s, f)).reshape(b, s, -1)
        out, _ = self.lstm(x_proj)              # (B, S, 2H)
        scores = self.attn(out).squeeze(-1)    # (B, S)
        weights = torch.softmax(scores, dim=-1)
        ctx = (out * weights.unsqueeze(-1)).sum(dim=1)   # (B, 2H)

        # Run through head manually to optionally capture intermediate
        h = self.dropout(ctx)
        h = self.head[0](h)   # Linear(160 → 80)
        head_pre_relu = h     # capture BEFORE ReLU
        h = self.head[1](h)   # ReLU  ← this is the 80-dim head_hidden
        if return_head_hidden:
            head_hidden = h
        h = self.head[2](h)   # Dropout
        out_pred = self.head[3](h).squeeze(-1)  # Linear(80 → 1)

        if return_head_pre_relu:
            if return_ctx:
                return out_pred, ctx, head_pre_relu
            return out_pred, head_pre_relu
        if return_ctx and return_head_hidden:
            return out_pred, ctx, head_hidden
        if return_ctx:
            return out_pred, ctx
        if return_head_hidden:
            return out_pred, head_hidden
        return out_pred
