"""
Temporal Convolutional Network (TCN) for soil moisture prediction.

Architecture
------------
  x_time   : (B, L, n_time)  — projected per-timestep then fed to a stack of
                               dilated causal 1-D convolutions.
  x_static : (B, n_static)   — projected once, concatenated to the pooled
                               sequence context, then passed to an MLP head.

Causal convolution is implemented with `nn.Conv1d` + left-padding of
`(kernel_size - 1) * dilation`, followed by cropping the output back to the
original length.  This guarantees output at time `t` only depends on inputs
at times `≤ t` (no leakage from the future).

Receptive field with kernel_size=3, dilations [1, 2, 4, 8, 16] and 5 blocks:
    RF = 1 + 2 * Σ dilations = 1 + 2 * 31 = 63 timesteps — covers the full
    60-day window.

References: Bai, Kolter & Koltun (2018) — "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling".
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """1-D convolution with causal left-padding.

    Input  : (B, C_in, L)
    Output : (B, C_out, L)  — same length, no future leakage.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=0, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad only (causal); right side stays untouched.
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """One residual TCN block: CausalConv → GELU → Dropout → CausalConv → GELU → Dropout + residual."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.act(self.conv2(x)))
        return x + res


class TCNRegressor(nn.Module):
    def __init__(
        self,
        n_time:   int,
        n_static: int,
        channels: int = 64,
        kernel_size: int = 3,
        dilations: tuple = (1, 2, 4, 8, 16),
        dropout:  float = 0.2,
        static_proj_size: int = 32,
        head_hidden: int = 64,
        pool: str = "mean_max",   # {"mean_max", "last"}
    ):
        super().__init__()
        self.pool = pool

        # --- Per-timestep time feature projection ---
        self.time_proj = nn.Sequential(
            nn.Linear(n_time, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Stack of dilated causal conv blocks ---
        self.blocks = nn.ModuleList([
            TCNBlock(channels, kernel_size, d, dropout) for d in dilations
        ])

        # --- Static feature branch ---
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, static_proj_size),
            nn.GELU(),
        )

        # --- Pool + head ---
        pooled_dim = 2 * channels if pool == "mean_max" else channels
        self.head = nn.Sequential(
            nn.Linear(pooled_dim + static_proj_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, C, L) → (B, pooled_dim)."""
        if self.pool == "last":
            return h[:, :, -1]
        # mean+max concat
        mean = h.mean(dim=-1)
        max_ = h.max(dim=-1).values
        return torch.cat([mean, max_], dim=-1)

    def forward(self, x_time: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        x_time   : (B, L, n_time)
        x_static : (B, n_static)
        returns  : (B,) predicted soil moisture
        """
        B, L, _ = x_time.shape

        # per-timestep projection  → (B, L, C)
        h = self.time_proj(x_time.reshape(B * L, -1)).reshape(B, L, -1)

        # conv expects (B, C, L)
        h = h.transpose(1, 2)
        for block in self.blocks:
            h = block(h)

        ctx = self._pool(h)                                # (B, pooled_dim)
        static_emb = self.static_proj(x_static)            # (B, static_proj_size)
        combined = torch.cat([ctx, static_emb], dim=-1)
        return self.head(combined).squeeze(-1)             # (B,)
