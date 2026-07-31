from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_cost(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = F.normalize(source, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - source @ target.transpose(-1, -2)


def generalized_kl(mass: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mass = mass.clamp_min(eps)
    reference = reference.clamp_min(eps)
    return (mass * (mass.log() - reference.log()) - mass + reference).sum(dim=-1)


def asrot_plan(
    cost: torch.Tensor,
    source_mass: torch.Tensor | None = None,
    target_mass: torch.Tensor | None = None,
    epsilon: float = 0.05,
    tau: float = 0.10,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Differentiable ASROT plan with relaxed source and fixed target marginal."""

    if cost.dim() != 3:
        raise ValueError("cost must have shape [B,M,N]")
    batch, source_count, target_count = cost.shape
    dtype, device = cost.dtype, cost.device
    if source_mass is None:
        source_mass = torch.full(
            (batch, source_count), 1.0 / source_count, dtype=dtype, device=device
        )
    if target_mass is None:
        target_mass = torch.full(
            (batch, target_count), 1.0 / target_count, dtype=dtype, device=device
        )
    source_mass = source_mass / source_mass.sum(dim=-1, keepdim=True)
    target_mass = target_mass / target_mass.sum(dim=-1, keepdim=True)

    log_kernel = -cost / epsilon
    log_source = source_mass.clamp_min(1e-12).log()
    log_target = target_mass.clamp_min(1e-12).log()
    log_v = torch.zeros_like(target_mass)
    relaxation = tau / (tau + epsilon)

    for _ in range(max_iterations):
        previous = log_v
        log_u = relaxation * (
            log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        )
        log_v = log_target - torch.logsumexp(
            log_kernel + log_u.unsqueeze(-1), dim=-2
        )
        if torch.max(torch.abs(log_v - previous)).detach().item() < tolerance:
            break
    return torch.exp(log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2))


def asrot_loss(
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    source_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    epsilon: float = 0.05,
    tau: float = 0.10,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean ASROT objective across paired patients."""

    losses, plans = [], []
    for index in range(source_tokens.shape[0]):
        source = source_tokens[index]
        target = target_tokens[index]
        if source_mask is not None:
            source = source[source_mask[index]]
        if target_mask is not None:
            target = target[target_mask[index]]
        if source.numel() == 0 or target.numel() == 0:
            continue
        cost = cosine_cost(source.unsqueeze(0), target.unsqueeze(0))
        source_mass = torch.full(
            (1, source.shape[0]), 1.0 / source.shape[0], device=source.device, dtype=source.dtype
        )
        target_mass = torch.full(
            (1, target.shape[0]), 1.0 / target.shape[0], device=target.device, dtype=target.dtype
        )
        plan = asrot_plan(
            cost,
            source_mass,
            target_mass,
            epsilon=epsilon,
            tau=tau,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        row_mass = plan.sum(dim=-1)
        entropy = (plan.clamp_min(1e-12) * (plan.clamp_min(1e-12).log() - 1.0)).sum()
        objective = (plan * cost).sum() + epsilon * entropy
        objective = objective + tau * generalized_kl(row_mass, source_mass).mean()
        losses.append(objective)
        plans.append(plan.squeeze(0))
    if not losses:
        zero = source_tokens.sum() * 0.0
        return zero, source_tokens.new_empty(0)
    return torch.stack(losses).mean(), torch.stack(plans) if len({p.shape for p in plans}) == 1 else plans
