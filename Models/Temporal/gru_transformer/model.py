"""
GRU -> Transformer hybrid regressor for soil moisture prediction.

Inspired by literature study #3 (GRU-Transformer for RZSM, R^2=98.86% at 3-day
forecast). The intuition is a division of labour:

  * GRU (bidirectional, 2 layers) — captures short-term, local dynamics such
    as wetting pulses after rain or daily drydown.  BiGRU hidden states carry
    implicit position information from the recurrence.
  * Transformer encoder (2 layers, 4 heads) — on top of the GRU outputs it
    can attend to any past timestep, so long-range dependencies (e.g. the
    tail of a 30-day drydown, seasonal persistence) are not squashed through
    the recurrence bottleneck.
  * Learned positional encoding — added to GRU outputs as a residual boost;
    the GRU already has some ordering information, so absolute position is
    a complementary signal rather than the only one.
  * Attention pooling (learned query vector) — produces a single context
    vector summarising the whole sequence.
  * Static feature branch — fixed terrain/soil/location features bypass the
    sequence axis entirely and are concatenated to the pooled context.

Flow
----
x_time   -> time_proj (n_time -> d_model) -> BiGRU -> +pos_emb
         -> TransformerEncoder -> attn_pool -> ctx
x_static -> static_proj (n_static -> static_proj_size)
cat(ctx, static_proj) -> dropout -> MLP head -> scalar prediction
"""

import torch
import torch.nn as nn


class GRUTransformerHybrid(nn.Module):
    def __init__(
        self,
        n_time: int,
        n_static: int,
        seq_len: int,
        d_model: int = 96,
        gru_layers: int = 2,
        gru_bidirectional: bool = True,
        transformer_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.25,
        static_proj_size: int = 32,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.d_model  = d_model

        # --- time feature projection ---
        # Linear + LayerNorm + GELU gives the GRU a smooth, compact input
        # representation (GELU tends to work better than ReLU for Transformer
        # stacks downstream; we use it here for consistency across the model).
        self.time_proj = nn.Sequential(
            nn.Linear(n_time, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Bidirectional GRU backbone ---
        # hidden_size is chosen so forward + backward concat equals d_model,
        # keeping the feature dimension consistent through the pipeline.
        assert (not gru_bidirectional) or d_model % 2 == 0, \
            "d_model must be even when gru_bidirectional=True"
        gru_hidden = d_model // 2 if gru_bidirectional else d_model
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=gru_bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # --- learned positional encoding ---
        # Residual boost on top of the GRU's implicit ordering information.
        # One vector per timestep, d_model wide; small init keeps it from
        # overwhelming the GRU signal early in training.
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # --- Transformer encoder on top of the GRU outputs ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,           # pre-LN is more stable for small nets
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(d_model),
        )

        # --- attention pooling over time (learned query vector) ---
        # A single learned query attends over all timesteps to produce one
        # context vector; equivalent to a lightweight additive attention.
        self.pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pool_proj  = nn.Linear(d_model, d_model, bias=False)

        # --- static feature branch ---
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, static_proj_size),
            nn.GELU(),
        )

        # --- MLP prediction head ---
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model + static_proj_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

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

        # project each timestep independently
        x = self.time_proj(x_time.reshape(batch * seq, -1)).view(batch, seq, -1)

        # BiGRU for short-term dynamics; output has d_model channels
        gru_out, _ = self.gru(x)                           # (batch, seq, d_model)

        # residual positional encoding (broadcast over batch)
        z = gru_out + self.pos_emb[:, :seq, :]

        # Transformer encoder for long-range attention
        z = self.transformer(z)                            # (batch, seq, d_model)

        # attention pooling via a learned query
        q       = self.pool_query                          # (d_model,)
        scores  = (self.pool_proj(z) * q).sum(dim=-1)      # (batch, seq)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq, 1)
        context = (z * weights).sum(dim=1)                 # (batch, d_model)

        # static branch
        static_emb = self.static_proj(x_static)            # (batch, static_proj)

        # combine and predict
        combined = torch.cat([context, static_emb], dim=-1)
        return self.head(self.dropout(combined)).squeeze(-1)   # (batch,)
