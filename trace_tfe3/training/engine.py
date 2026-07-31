from __future__ import annotations

import torch
from torch import nn

from .losses import trace_loss
from .metrics import binary_auc


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def train_one_epoch(
    model: nn.Module,
    pathology_teacher: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_pathology_transfer: bool,
    asrot_config: dict,
) -> dict[str, float]:
    model.train()
    freeze_module(pathology_teacher)
    totals: list[float] = []

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["ct"].to(device))
        pathology_tokens = None
        pathology_mask = None
        if use_pathology_transfer:
            with torch.no_grad():
                teacher = pathology_teacher(
                    batch["pathology_tokens"].to(device),
                    batch["pathology_mask"].to(device),
                )
            pathology_tokens = teacher["pathology_tokens"].detach()
            pathology_mask = teacher["pathology_mask"]
        losses = trace_loss(
            outputs,
            batch["label"].to(device),
            pathology_tokens=pathology_tokens,
            pathology_mask=pathology_mask,
            lambda_ot=asrot_config.get("lambda_ot", 0.20) if use_pathology_transfer else 0.0,
            epsilon=asrot_config.get("epsilon", 0.05),
            tau=asrot_config.get("tau", 0.10),
            max_iterations=asrot_config.get("max_iterations", 50),
            tolerance=asrot_config.get("tolerance", 1e-6),
        )
        losses["loss_total"].backward()
        optimizer.step()
        totals.append(float(losses["loss_total"].detach().cpu()))
    return {"loss": sum(totals) / max(1, len(totals))}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels, scores = [], []
    for batch in loader:
        outputs = model(batch["ct"].to(device))
        labels.extend(batch["label"].cpu().tolist())
        scores.extend(torch.sigmoid(outputs["logit"]).cpu().tolist())
    return {"auc": binary_auc(labels, scores)}
