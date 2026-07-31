from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from .layers import GatedAttentionAggregator


class PathologyTeacher(nn.Module):
    """Development-only H&E token adapter, aggregator, and auxiliary classifier."""

    def __init__(
        self,
        token_embedding_dim: int = 1536,
        shared_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_adapter = nn.Sequential(
            nn.LayerNorm(token_embedding_dim),
            nn.Linear(token_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, shared_dim),
        )
        self.patient_aggregator = GatedAttentionAggregator(
            input_dim=shared_dim,
            hidden_dim=hidden_dim,
            output_dim=shared_dim,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(shared_dim, 1))

    def forward(
        self,
        token_embeddings: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        adapted_tokens = self.token_adapter(token_embeddings)
        patient_embedding, attention = self.patient_aggregator(adapted_tokens, token_mask)
        return {
            "pathology_tokens": adapted_tokens,
            "patient_embedding": patient_embedding,
            "logit": self.classifier(patient_embedding).squeeze(-1),
            "token_attention": attention,
        }


class GigaPathSlideEmbedder(nn.Module):
    """Thin wrapper around a local Prov-GigaPath tile and slide encoder."""

    def __init__(
        self,
        gigapath_repo: str | Path = "../prov-gigapath",
        tile_encoder_path: str | Path | None = None,
        slide_encoder_path: str | Path | None = None,
        global_pool: bool = False,
    ) -> None:
        super().__init__()
        repository = Path(gigapath_repo)
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        try:
            from gigapath.pipeline import load_tile_slide_encoder
        except Exception as exc:  # pragma: no cover - external dependency
            raise ImportError(f"Could not import Prov-GigaPath from {repository}") from exc
        self.tile_encoder, self.slide_encoder = load_tile_slide_encoder(
            str(tile_encoder_path or ""),
            str(slide_encoder_path or ""),
            global_pool=global_pool,
        )

    @torch.no_grad()
    def encode_slide_from_tile_embeddings(
        self,
        tile_embeddings: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        if tile_embeddings.dim() == 2:
            tile_embeddings = tile_embeddings.unsqueeze(0)
            coords = coords.unsqueeze(0)
        outputs = self.slide_encoder(tile_embeddings, coords, all_layer_embed=False)
        return outputs[-1] if isinstance(outputs, list) else outputs
