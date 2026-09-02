from __future__ import annotations

import torch
from torch import nn


class MultiOrganRegressor(nn.Module):
    """Neural architecture used for the 2 x 2 x 2 structural comparison."""

    def __init__(
        self,
        num_organs: int,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        branch_type: str = "OSB",
        attention: str = "MHA",
        aggregation: str = "AWA",
    ) -> None:
        super().__init__()
        self.num_organs = num_organs
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.branch_type = branch_type
        self.attention_type = attention
        self.aggregation = aggregation

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        def branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        if branch_type == "OSB":
            self.branch_nets = nn.ModuleList([branch() for _ in range(num_organs)])
        elif branch_type == "USB":
            self.shared_branch = branch()
        else:
            raise ValueError(f"Unknown branch type: {branch_type}")

        # Kept for state-dict compatibility even when WMHA skips the operation.
        self.organ_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim // 2)

        self.aggregate_organ = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

        if aggregation == "AWA":
            self.organ_importance_net = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),
            )
        elif aggregation != "APA":
            raise ValueError(f"Unknown aggregation: {aggregation}")
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode_organs(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        if self.branch_type == "OSB":
            return torch.stack(
                [self.branch_nets[index](features[:, index, :]) for index in range(self.num_organs)],
                dim=1,
            )
        flat = features.reshape(-1, features.shape[-1])
        return self.shared_branch(flat).reshape(batch_size, self.num_organs, -1)

    def _organ_representation(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        flat = x.reshape(-1, x.shape[-1])
        features = self.feature_extractor(flat).reshape(batch_size, self.num_organs, -1)
        organ_features = self._encode_organs(features) * mask.unsqueeze(-1)
        if self.attention_type == "MHA":
            attended, _ = self.organ_attention(
                organ_features,
                organ_features,
                organ_features,
                key_padding_mask=~mask.bool(),
            )
            attended = attended * mask.unsqueeze(-1)
            organ_features = self.attention_norm(organ_features + attended)
        elif self.attention_type != "WMHA":
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        return organ_features

    def normalized_organ_weights(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the per-sample normalized AWA weights used for prediction."""
        if self.aggregation != "AWA":
            raise ValueError("Organ weights are defined only for AWA models")
        if mask is None:
            mask = torch.ones(x.shape[0], self.num_organs, device=x.device)
        mask = mask.float()
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("Every sample must retain at least one organ")
        organ_features = self._organ_representation(x, mask)
        weights = self.organ_importance_net(organ_features).squeeze(-1) * mask
        return weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(x.shape[0], self.num_organs, device=x.device)
        mask = mask.float()
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("Every sample must retain at least one organ")

        organ_features = self._organ_representation(x, mask)

        organ_predictions = self.aggregate_organ(organ_features)
        organ_predictions = torch.clamp(organ_predictions * mask.unsqueeze(-1), min=0)
        flat_predictions = organ_predictions.squeeze(-1)

        if self.aggregation == "AWA":
            weights = self.organ_importance_net(organ_features).squeeze(-1) * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            final = (flat_predictions * weights).sum(dim=1)
        else:
            final = flat_predictions.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return torch.clamp(final, min=0), organ_predictions


def build_model(
    branch: str,
    attention: str,
    aggregation: str,
    num_organs: int = 8,
    input_dim: int = 467,
    hidden_dim: int = 256,
    dropout: float = 0.3,
) -> MultiOrganRegressor:
    return MultiOrganRegressor(
        num_organs=num_organs,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        branch_type=branch,
        attention=attention,
        aggregation=aggregation,
    )


def build_from_name(
    name: str,
    num_organs: int = 8,
    input_dim: int = 467,
    hidden_dim: int = 256,
    dropout: float = 0.3,
) -> MultiOrganRegressor:
    parts = name.split("-")
    if len(parts) < 3:
        raise ValueError(f"Expected model name OSB-MHA-AWA[-MORM], got {name!r}")
    return build_model(parts[0], parts[1], parts[2], num_organs, input_dim, hidden_dim, dropout)
