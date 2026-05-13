"""
Transformer-encoder regressor for soil moisture prediction.

Architecture
============
  x_time   (B, seq_len, n_time)
      -> Linear input projection (n_time -> d_model)
      -> + learned positional embedding (seq_len, d_model)
      -> Transformer encoder stack (num_layers * (MHSA + FFN))
      -> CLS-style pooled context (learned [CLS] token prepended)
  x_static (B, n_static)
      -> MLP projection (n_static -> static_proj_size)
  cat(cls_context, static_emb) -> MLP head -> scalar prediction

Design notes
------------
* Learned positional embeddings: sequences are short (seq_len=90) and the
  training set tiny, so learned embeddings are easier to fit than
  sinusoids and let the model discover its own notion of "recent vs. old".
* CLS pooling: with noisy meteorological sequences a prepended learnable
  [CLS] token that attends freely to the whole window tends to beat
  last-token (too noisy) or mean pooling (dilutes the wetting-front
  signal that sits on specific days).
* pre-LN encoder (norm_first=True) — standard for stable training on
  small Transformers.
"""

import torch
import torch.nn as nn


class TransformerSoilMoisture(nn.Module):
    def __init__(
        self,
        n_time: int,
        n_static: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        seq_len: int = 90,
        static_proj_size: int = 32,
    ):
        super().__init__()

        # --- input projection: per-timestep n_time -> d_model ---
        self.input_proj = nn.Linear(n_time, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # --- learned positional embedding (+1 slot for the CLS token) ---
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- CLS token: learnable pooling query ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --- Transformer encoder stack ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-LN — more stable on small datasets
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # --- static branch ---
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, static_proj_size),
            nn.GELU(),
        )

        # --- prediction head ---
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model + static_proj_size, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x_time: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_time  : (batch, seq_len, n_time)
            x_static: (batch, n_static)
        Returns:
            (batch,) predicted soil moisture
        """
        batch = x_time.size(0)

        # project + add CLS + positional embedding
        x = self.input_norm(self.input_proj(x_time))                # (B, seq_len, d_model)
        cls = self.cls_token.expand(batch, -1, -1)                  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                              # (B, seq_len+1, d_model)
        x = x + self.pos_embed[:, : x.size(1), :]

        # encoder
        x = self.encoder(x)                                         # (B, seq_len+1, d_model)
        x = self.encoder_norm(x)

        cls_out = x[:, 0, :]                                        # (B, d_model) — pooled context

        # static branch + head
        static_emb = self.static_proj(x_static)                     # (B, static_proj_size)
        combined   = torch.cat([cls_out, static_emb], dim=-1)
        return self.head(combined).squeeze(-1)                      # (B,)
