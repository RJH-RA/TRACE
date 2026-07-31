from __future__ import annotations

import torch
import torch.nn.functional as F

from trace_tfe3.models.transport import asrot_loss


def trace_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    pathology_tokens: torch.Tensor | None = None,
    pathology_mask: torch.Tensor | None = None,
    classification_weight: float = 1.0,
    lambda_ot: float = 0.20,
    epsilon: float = 0.05,
    tau: float = 0.10,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    labels = labels.to(device=outputs["logit"].device, dtype=torch.float32)
    classification = F.binary_cross_entropy_with_logits(
        outputs["logit"], labels, pos_weight=pos_weight
    )
    total = classification_weight * classification
    parts = {"loss_cls": classification}

    if pathology_tokens is not None and pathology_tokens.numel() and lambda_ot > 0:
        alignment, _ = asrot_loss(
            outputs["ct_tokens"],
            pathology_tokens.to(outputs["ct_tokens"].device),
            target_mask=(
                pathology_mask.to(outputs["ct_tokens"].device)
                if pathology_mask is not None
                else None
            ),
            epsilon=epsilon,
            tau=tau,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        parts["loss_asrot"] = alignment
        total = total + lambda_ot * alignment

    parts["loss_total"] = total
    return parts
