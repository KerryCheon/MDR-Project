"""Neural tabular regressors for derived_8.4-eval-mlp-2.3.

Extends the mlp13 architecture zoo (plain MLP, residual MLP, FT-Transformer)
with the two new architectures this experiment exists to test:

  - FeatureGroupedMLP  : per-semantic-group towers (see mlp21.feature_groups)
                         -> concat group embeddings -> fusion MLP -> head.
                         Targets 1.2's documented overfitting kind: capacity
                         spent on period-specific *interactions* between
                         heterogeneous sensor families. Group towers force a
                         shared per-sensor representation before mixing.
  - PLRRegressor        : Piecewise-Linear Encoding (Gorishniy et al. 2022,
                         "Revisiting Deep Learning Models for Tabular Data")
                         of each input feature followed by the plain-MLP body.
                         Never tried in this project; a cheap, well-established
                         tabular upgrade for MLPs.

Plain MLP / ResidualMLP / FTTransformer are kept unchanged (ResidualMLP and
FTTransformer remain reference-only rows — documented failures in 1.1/1.2).
All architectures are seeded reproducibly at construction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .feature_groups import FeatureGroups, group_features


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
    """Residual MLP: projection to width, then L residual blocks, then head (reference-only)."""

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
    """Feature-tokenizer Transformer (Gorishniy et al. 2021), regression head (reference-only)."""

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


class FeatureGroupedMLP(nn.Module):
    """Per-semantic-group towers + fusion MLP (architecture: "fg").

    Layout:
      x (B, F) -> for each group g: tower_g(x[:, idx_g])  -> group embedding (B, tower_width)
      concat -> [optional group-dropout] -> fusion MLP (hidden_sizes) -> head(1)

    tower: Linear(len_g -> w) -> Norm -> act -> Dropout -> Linear(w -> w) -> Norm -> act -> Dropout
    fusion: same block pattern as MLPRegressor on the concatenated embeddings.

    The grouping is resolved from the feature names by mlp21.feature_groups
    (single source of truth, validated: every feature in exactly one group).
    """

    def __init__(
        self,
        n_features: int,
        feature_groups: FeatureGroups,
        hidden_sizes: list[int] | tuple[int, ...],
        activation: str = "silu",
        dropout: float = 0.3,
        norm: str = "bn",
        seed: int = 42,
        tower_width: int = 256,
        group_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_groups.n_features != int(n_features):
            raise ValueError(
                f"feature_groups has {feature_groups.n_features} features but n_features={n_features}"
            )
        self.n_features = int(n_features)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.dropout = float(dropout)
        self.tower_width = int(tower_width)
        self.group_dropout = float(group_dropout)
        self._groups = feature_groups  # keep for n_params/describe

        torch.manual_seed(int(seed))
        towers: list[nn.Module] = []
        for gid, idxs in enumerate(feature_groups.groups):
            towers.append(
                nn.Sequential(
                    nn.Linear(len(idxs), self.tower_width),
                    _make_norm(norm, self.tower_width),
                    _make_activation(activation),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(self.tower_width, self.tower_width),
                    _make_norm(norm, self.tower_width),
                    _make_activation(activation),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                )
            )
        self.towers = nn.ModuleList(towers)
        fusion_in = self.tower_width * len(feature_groups.groups)
        layers: list[nn.Module] = []
        in_dim = fusion_in
        for h in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(_make_norm(norm, h))
            layers.append(_make_activation(activation))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.fusion = nn.Sequential(*layers)
        if self.group_dropout > 0.0:
            self.group_dropout_layer = nn.Dropout(self.group_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = [tower(x[:, idxs]) for tower, idxs in zip(self.towers, self._groups.groups)]
        h = torch.cat(embs, dim=1)
        if self.group_dropout > 0.0:
            h = self.group_dropout_layer(h)
        return self.fusion(h).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PLRLayer(nn.Module):
    """Piecewise-linear encoding of every input feature (Gorishniy et al. 2022).

    Each feature x_i maps to n_bins outputs:
        [x_i]  concat  [ReLU(x_i - knot_k)]_k
    followed by a mixing Linear (n_features*n_bins -> n_features*n_bins).
    """

    def __init__(self, n_features: int, n_bins: int = 8, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(int(seed))
        self.n_features = int(n_features)
        self.n_bins = int(n_bins)
        self.linear = nn.Linear(self.n_features * self.n_bins, self.n_features * self.n_bins)
        self.knots = nn.Parameter(torch.randn(self.n_features, self.n_bins - 1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> (B, F, 1); knots broadcast (F, B-1)
        x = x.unsqueeze(-1)
        knots = self.knots.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([x, torch.relu(x - knots)], dim=-1)  # (B, F, n_bins)
        x = x.flatten(1)  # (B, F*n_bins)
        return self.linear(x)


class PLRRegressor(nn.Module):
    """PLR encoding + plain-MLP body (architecture: "plr")."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int] | tuple[int, ...],
        activation: str = "silu",
        dropout: float = 0.3,
        norm: str = "bn",
        seed: int = 42,
        n_bins: int = 8,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_bins = int(n_bins)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.plr = PLRLayer(self.n_features, self.n_bins, seed=seed)
        self.body = MLPRegressor(
            self.n_features * self.n_bins,
            self.hidden_sizes,
            activation=activation,
            dropout=dropout,
            norm=norm,
            seed=seed,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.plr(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(
    cfg: dict,
    n_features: int,
    feature_names: list[str] | None = None,
) -> nn.Module:
    """Factory: cfg['architecture'] in {mlp, residual, ft, fg, plr}.

    `feature_names` is required for architecture "fg" (semantic grouping is
    resolved from the names); all other architectures ignore it.
    """
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
    if arch == "fg":
        if feature_names is None:
            raise ValueError("architecture 'fg' requires feature_names (semantic grouping)")
        groups = group_features(feature_names)
        return FeatureGroupedMLP(
            n_features=n_features,
            feature_groups=groups,
            hidden_sizes=hidden,
            activation=activation,
            dropout=dropout,
            norm=norm,
            seed=seed,
            tower_width=int(cfg.get("fg_tower_width", 256)),
            group_dropout=float(cfg.get("fg_group_dropout", 0.0)),
        )
    if arch == "plr":
        return PLRRegressor(
            n_features=n_features,
            hidden_sizes=hidden,
            activation=activation,
            dropout=dropout,
            norm=norm,
            seed=seed,
            n_bins=int(cfg.get("plr_n_bins", 8)),
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
    """Warmup + cosine LR scheduler (mlp-1.3 behavior, unchanged)."""
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
