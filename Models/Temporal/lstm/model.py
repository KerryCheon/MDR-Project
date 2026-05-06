"""
LSTM regressor for soil moisture prediction — raw time-series approach.

Architecture
============
Instead of feeding pre-computed lag/rolling features (which already encode
temporal context), this model receives:

  x_time  : (batch, seq_len, n_time)   — raw daily observations (SAR, optical,
                                          LST, SMAP, precipitation, seasonality)
  x_static: (batch, n_static)           — fixed terrain / soil / location
                                          features (no sequence axis)

The LSTM must learn temporal patterns (lag effects, wetting/drying dynamics,
seasonal cycles) directly from the raw sequence — not from hand-crafted
rolling features.  Static features are injected into the prediction head
after temporal context has been extracted.

Flow
----
x_time  -> time_proj (n_time -> time_proj_size) -> LSTM -> attention -> context
x_static -> static_proj (n_static -> static_proj_size)
cat(context, static_proj) -> dropout -> head -> scalar prediction
"""

import torch
import torch.nn as nn


class LSTMRawSeries(nn.Module):
    def __init__(
        self,
        n_time: int,
        n_static: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        time_proj_size: int = 32,
        static_proj_size: int = 32,
    ):
        super().__init__()

        # --- time feature projection ---
        # small bottleneck so LSTM sees a compact, mixed representation
        self.time_proj = nn.Sequential(
            nn.Linear(n_time, time_proj_size),
            nn.LayerNorm(time_proj_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- LSTM backbone ---
        self.lstm = nn.LSTM(
            input_size=time_proj_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- attention over timesteps ---
        self.attn = nn.Linear(hidden_size, 1, bias=False)

        # --- static feature branch ---
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, static_proj_size),
            nn.ReLU(),
        )

        # --- prediction head ---
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + static_proj_size, 1)

    def forward(
        self,
        x_time: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_time  : (batch, seq_len, n_time)
            x_static: (batch, n_static)
        Returns:
            (batch,) predicted soil moisture
        """
        batch, seq, _ = x_time.shape

        # project each timestep
        x = self.time_proj(x_time.view(batch * seq, -1)).view(batch, seq, -1)

        # LSTM
        out, _ = self.lstm(x)   # (batch, seq, hidden)

        # soft attention over all timesteps
        scores  = self.attn(out)                  # (batch, seq, 1)
        weights = torch.softmax(scores, dim=1)    # (batch, seq, 1)
        context = (out * weights).sum(dim=1)      # (batch, hidden)

        # static branch
        static_emb = self.static_proj(x_static)  # (batch, static_proj)

        # combine and predict
        combined = torch.cat([context, static_emb], dim=-1)   # (batch, hidden + static_proj)
        return self.head(self.dropout(combined)).squeeze(-1)   # (batch,)
