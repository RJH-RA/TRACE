#!/usr/bin/env python
"""Dependency-light contract smoke test for configured compute environments."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_tfe3.data import TRACECaseDataset, trace_collate
from trace_tfe3.evaluation import (
    apply_trace_clinical,
    fit_trace_clinical,
    select_operating_point,
    threshold_metrics,
)
from trace_tfe3.models import CompactSliceBackbone, TRACECTEncoder, TRACEModel
from trace_tfe3.models.transport import asrot_loss, asrot_plan


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        np.save(root / "nc.npy", np.full((16, 16, 16), 0.25, dtype=np.float32))
        np.save(root / "ap.npy", np.full((16, 16, 16), 0.75, dtype=np.float32))
        torch.save(torch.randn(6, 1536), root / "he.pt")
        manifest = root / "manifest.csv"
        manifest.write_text(
            "patient_id,split,label,noncontrast,arterial,he_token_embeddings\n"
            f"P1,train,1,{root / 'nc.npy'},{root / 'ap.npy'},{root / 'he.pt'}\n",
            encoding="utf-8",
        )
        item = TRACECaseDataset(manifest, "train", require_pathology=True)[0]
        batch = trace_collate([item])
        assert batch["ct"].shape == (1, 3, 16, 16, 16)
        assert torch.allclose(batch["ct"][:, 2], torch.full((1, 16, 16, 16), 0.5))

    encoder = TRACECTEncoder(
        backbone=CompactSliceBackbone(3, 64),
        backbone_output_dim=64,
        shared_dim=32,
        token_count=16,
        transformer_layers=1,
        transformer_heads=4,
        freeze_backbone=False,
    )
    model = TRACEModel(encoder, shared_dim=32)
    output = model(torch.randn(2, 3, 16, 32, 32))
    assert output["logit"].shape == (2,)
    assert output["ct_tokens"].shape == (2, 16, 32)

    cost = torch.rand(2, 5, 7)
    target = torch.full((2, 7), 1.0 / 7)
    plan = asrot_plan(cost, target_mass=target, max_iterations=200, tolerance=1e-8)
    assert torch.allclose(plan.sum(dim=-2), target, atol=1e-5, rtol=1e-5)

    source = torch.randn(2, 6, 32, requires_grad=True)
    pathology = source.detach() + 0.01 * torch.randn(2, 6, 32)
    loss, _ = asrot_loss(source, pathology)
    loss.backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()

    development = pd.DataFrame(
        {
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "trace_ct_score": [0.08, 0.20, 0.31, 0.42, 0.45, 0.64, 0.78, 0.91],
            "age": [66, 59, 63, 57, 42, 37, 45, 31],
            "sex": ["M", "F", "M", "F", "F", "M", "F", "F"],
            "automated_maximum_tumour_diameter_cm": [
                3.1,
                4.8,
                2.7,
                4.0,
                4.5,
                3.8,
                5.2,
                3.4,
            ],
        }
    )
    operating_point = select_operating_point(
        development["label"],
        development["trace_ct_score"],
        minimum_sensitivity=0.75,
    )
    metrics = threshold_metrics(
        development["label"],
        development["trace_ct_score"],
        operating_point["threshold"],
    )
    assert metrics["sensitivity"] >= 0.75
    clinical = fit_trace_clinical(development)
    clinical_scores = apply_trace_clinical(development, clinical)
    assert clinical_scores.shape == (len(development),)
    assert np.isfinite(clinical_scores).all()
    print("TRACE smoke test passed")


if __name__ == "__main__":
    main()
