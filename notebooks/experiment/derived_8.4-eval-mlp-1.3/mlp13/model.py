"""Neural tabular regressors for derived_8.4-eval-mlp-1.3.

Three architectures, all seeded reproducibly at construction:
  - MLPRegressor   : Linear -> [Norm -> act -> Dropout]^L -> Linear(1)
                     (mlp-1.0 architecture, plus a LayerNorm option).
  - ResidualMLP    : Linear projection -> stack of ResBlocks -> Linear(1);
                     ResBlock = Linear -> Norm -> act -> Dropout -> Linear -> Norm
                     -> + skip -> act (pre-activation style, inspired by
                     Gorishniy et al. "Revisiting Deep Learning Models for
                     Tabular Data").
  - FTTransformer  : per-feature tokenizer (Linear(1 -> d) per feature) +
                     TransformerEncoder (pre-LN, ReZero-free) + CLS head,
                     following Gorishniy et al. 2021. Attention heads =
                     max(1, d // 16) so small token dims stay valid.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "silu":
        return nn.SiLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


def _make_norm(name: str, dim: int) -> nn.Module:
    if name == "bn":
        return nn.BatchNorm1d(dim)
    elif name == "ln":
        return nn.LayerNorm(dim)
    elif name in (None, "none"):
        return nn.Identity()
    raise ValueError(f"Unknown norm: {name}")


class MLPRegressor(nn.Module):
    """Plain feed-forward MLP regressor (mlp-1.0 compatible)."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int] | tuple[int, ...],
        activation: str = "silu",
        dropout: float = 0.3,
        norm: str = "bn",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.activation = activation
        self.dropout = float(dropout)
        self.norm = norm

        torch.manual_seed(int(seed))
        layers: list[nn.Module] = []
        in_dim = self.n_features
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_make_norm(norm, h))
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


class _ResBlock(nn.Module):
    """Pre-activation residual block: act -> norm -> linear -> act -> norm -> linear, + skip."""

    def __init__(
        self,
        dim: int,
        activation: str,
        dropout: float,
        norm: str,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.net = nn.Sequential(
            _make_activation(activation),
            _make_norm(norm, dim),
            nn.Linear(dim, dim),
            _make_activation(activation),
            _make_norm(norm, dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualMLP(nn.Module):
    """Residual MLP: projection to width, then L residual blocks, then head."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int] | tuple[int, ...],
        activation: str = "silu",
        dropout: float = 0.2,
        norm: str = "bn",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        sizes = [int(h) for h in hidden_sizes]
        self.hidden_sizes = sizes

        torch.manual_seed(int(seed))
        self.proj = nn.Sequential(
            nn.Linear(self.n_features, sizes[0]),
            _make_norm(norm, sizes[0]),
            _make_activation(activation),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        blocks: list[nn.Module] = []
        prev = sizes[0]
        for i, h in enumerate(sizes):
            if i == 0:
                blocks.append(_ResBlock(h, activation, dropout, norm, seed=seed + i))
            else:
                # width-change block: project prev -> h then residual block at h
                blocks.append(
                    nn.Sequential(
                        nn.Linear(prev, h),
                        _ResBlock(h, activation, dropout, norm, seed=seed + i),
                    )
                )
            prev = h
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.blocks(h)
        return self.head(h).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _FTBlock(nn.Module):
    """Pre-LN transformer encoder block (norm -> attn -> +, norm -> mlp -> +)."""

    def __init__(self, d: int, heads: int, dropout: float, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """Feature-tokenizer Transformer (Gorishniy et al. 2021), regression head.

    Every input feature is a separate token: token_i = Linear(1 -> d)(x_i).
    A learnable [CLS] token pools the encoder output; the head is a 2-layer MLP.
    """

    def __init__(
        self,
        n_features: int,
        d: int = 64,
        layers: int = 4,
        dropout: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.d = int(d)
        heads = max(1, int(d) // 16)

        torch.manual_seed(int(seed))
        self.tokenizer = nn.Linear(1, int(d))
        self.cls = nn.Parameter(torch.zeros(1, 1, int(d)))
        nn.init.trunc_normal_(self.cls, std=0.02)
        blocks: list[nn.Module] = []
        for i in range(int(layers)):
            blocks.append(_FTBlock(int(d), heads, float(dropout), seed=seed + 100 + i))
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(int(d)),
            nn.Linear(int(d), int(d)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> tokens: (B, F, d)
        tokens = self.tokenizer(x.unsqueeze(-1))
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.encoder(tokens)
        pooled = out[:, 0]
        return self.head(pooled).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: dict, n_features: int) -> nn.Module:
    """Factory: cfg['architecture'] in {mlp, residual, ft}."""
    arch = cfg.get("architecture", "mlp")
    seed = int(cfg.get("seed", 42))
    activation = cfg.get("activation", "silu")
    dropout = float(cfg.get("dropout", 0.3))
    norm = cfg.get("norm", "bn")
    hidden = [int(h) for h in cfg["hidden_sizes"]] if "hidden_sizes" in cfg else [256, 256]

    if arch == "ft":
        return FTTransformer(
            n_features=n_features,
            d=int(cfg.get("ft_d", 64)),
            layers=int(cfg.get("ft_layers", 4)),
            dropout=float(cfg.get("ft_dropout", 0.1)),
            seed=seed,
        )
    if arch == "residual":
        return ResidualMLP(
            n_features=n_features,
            hidden_sizes=hidden,
            activation=activation,
            dropout=dropout,
            norm=norm,
            seed=seed,
        )
    return MLPRegressor(
        n_features=n_features,
        hidden_sizes=hidden,
        activation=activation,
        dropout=dropout,
        norm=norm,
        seed=seed,
    )


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, max_epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
    """Warmup + cosine LR scheduler.

    warmup_frac (default 0.05) of max_epochs linearly ramps lr from ~0 to the
    base lr, then cosine decays to base_lr/100 at max_epochs (mlp-1.0 behavior
    for the cosine tail; warmup is new in 1.1).
    """
    warmup_frac = float(cfg.get("warmup_frac", 0.05))
    base_lr = float(cfg.get("lr", 3e-4))
    eta_min = base_lr / 100.0
    warmup_epochs = int(round(max_epochs * warmup_frac))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            if warmup_epochs <= 0:
                return 1.0
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(eta_min / base_lr, cos * (1.0 - eta_min / base_lr) + eta_min / base_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
