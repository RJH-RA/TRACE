from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .layers import ResidualMLPAdapter


class CompactSliceBackbone(nn.Module):
    """Small test backbone with the same tensor contract as a 2D DINOv3 encoder."""

    def __init__(self, in_channels: int = 3, output_dim: int = 768) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)


def build_dinov3_backbone(
    model_name: str,
    checkpoint: str | Path | None,
    in_channels: int = 3,
    backend: str = "torch_hub_local",
    repository: str | Path | None = None,
) -> nn.Module:
    """Build a DINOv3 backbone without silently accepting incompatible weights.

    The production contract uses the official local DINOv3 repository through
    ``torch.hub``. ``timm`` remains available only as an explicit compatibility
    backend and requires a near-complete checkpoint match.
    """

    if in_channels != 3:
        raise ValueError("The locked DINOv3 contract expects three CT-derived channels")
    if backend == "torch_hub_local":
        if not repository:
            raise ValueError("repository is required for backend='torch_hub_local'")
        repository = Path(repository)
        if not repository.exists():
            raise FileNotFoundError(f"DINOv3 repository not found: {repository}")
        if checkpoint and not Path(checkpoint).exists():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")
        kwargs = {"source": "local"}
        if checkpoint:
            kwargs["weights"] = str(checkpoint)
        return torch.hub.load(str(repository), model_name, **kwargs)

    if backend != "timm":
        raise ValueError(f"Unsupported DINOv3 backend: {backend!r}")
    try:
        import timm
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("The explicit timm backend requires timm.") from exc
    model = timm.create_model(model_name, pretrained=False, num_classes=0, in_chans=3)
    if checkpoint:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state.get("state_dict", state.get("model", state))
        normalised = {
            key.removeprefix("module.").removeprefix("backbone."): value
            for key, value in state_dict.items()
        }
        incompatible = model.load_state_dict(normalised, strict=False)
        model_keys = set(model.state_dict())
        matched = model_keys.difference(incompatible.missing_keys)
        match_fraction = len(matched) / max(1, len(model_keys))
        if incompatible.unexpected_keys or match_fraction < 0.95:
            raise RuntimeError(
                "Checkpoint is not compatible with the configured backbone: "
                f"matched={match_fraction:.1%}, "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )
    return model


class TRACECTEncoder(nn.Module):
    """Slice-wise CT encoder and axial token aggregator used by TRACE-c."""

    def __init__(
        self,
        backbone: nn.Module | None = None,
        backbone_output_dim: int = 768,
        shared_dim: int = 256,
        token_count: int = 16,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone or CompactSliceBackbone(3, backbone_output_dim)
        self.token_count = token_count
        self.adapter = ResidualMLPAdapter(
            dim=backbone_output_dim,
            bottleneck_dim=shared_dim,
            dropout=0.1,
        )
        self.project = nn.Linear(backbone_output_dim, shared_dim)
        self.position = nn.Parameter(torch.zeros(1, token_count, shared_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=shared_dim,
            nhead=transformer_heads,
            dim_feedforward=shared_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.aggregator = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(shared_dim)
        nn.init.trunc_normal_(self.position, std=0.02)
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def forward(self, ct: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode CT shaped [B, 3, D, H, W] into axial tokens and a patient vector."""

        if ct.dim() != 5 or ct.shape[1] != 3:
            raise ValueError("ct must have shape [B,3,D,H,W]")
        batch, channels, depth, height, width = ct.shape
        if depth != self.token_count:
            raise ValueError(f"expected {self.token_count} axial positions, got {depth}")
        slices = ct.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
        features = self.backbone(slices)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if isinstance(features, dict):
            features = features.get("x_norm_clstoken", features.get("features"))
        if features.dim() > 2:
            features = features.flatten(2).mean(-1)
        features = features.reshape(batch, depth, -1)
        tokens = self.project(self.adapter(features.reshape(batch * depth, -1)))
        tokens = tokens.reshape(batch, depth, -1) + self.position
        tokens = self.aggregator(tokens)
        patient_embedding = self.norm(tokens.mean(dim=1))
        return {"ct_tokens": tokens, "patient_embedding": patient_embedding}
