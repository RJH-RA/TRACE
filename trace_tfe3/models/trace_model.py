from __future__ import annotations

import torch
from torch import nn

from .ct_dinov3 import TRACECTEncoder


class TRACEModel(nn.Module):
    """Deployable TRACE-CT network; pathology is consumed only by the training loss."""

    def __init__(self, ct_encoder: TRACECTEncoder, shared_dim: int = 256) -> None:
        super().__init__()
        self.ct_encoder = ct_encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, ct: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.ct_encoder(ct)
        return {
            **encoded,
            "logit": self.classifier(encoded["patient_embedding"]).squeeze(-1),
        }
