#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_epoch(model, loader, optimizer, device, train: bool):
    import torch
    import torch.nn.functional as F

    from trace_tfe3.training.metrics import binary_auc

    model.train(train)
    labels, scores, losses = [], [], []
    for batch in loader:
        tokens = batch["pathology_tokens"].to(device)
        mask = batch["pathology_mask"].to(device)
        label = batch["label"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        output = model(tokens, mask)
        loss = F.binary_cross_entropy_with_logits(output["logit"], label)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels.extend(label.detach().cpu().tolist())
        scores.extend(torch.sigmoid(output["logit"]).detach().cpu().tolist())
    return {"loss": sum(losses) / max(1, len(losses)), "auc": binary_auc(labels, scores)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the development-only H&E teacher")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from trace_tfe3.data import TRACECaseDataset, trace_collate
    from trace_tfe3.models import PathologyTeacher
    from trace_tfe3.utils.config import load_config

    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda"))
    manifest = cfg["data"]["manifest_csv"]
    train_set = TRACECaseDataset(manifest, cfg["data"]["train_split"], require_pathology=True)
    validation_set = TRACECaseDataset(
        manifest, cfg["data"]["validation_split"], require_pathology=True
    )
    loader_args = {
        "batch_size": cfg["training"].get("pathology_batch_size", 16),
        "collate_fn": trace_collate,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)

    pathology_cfg = cfg["pathology"]
    model = PathologyTeacher(
        token_embedding_dim=pathology_cfg.get("token_embedding_dim", 1536),
        shared_dim=pathology_cfg.get("shared_dim", 256),
        hidden_dim=pathology_cfg.get("patient_hidden_dim", 256),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"].get("pathology_learning_rate", 1e-5),
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )

    best_auc = float("-inf")
    destination = Path(cfg["output_dir"]) / "pathology_he"
    destination.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg["training"].get("pathology_epochs", 300) + 1):
        training = run_epoch(model, train_loader, optimizer, device, train=True)
        with torch.no_grad():
            validation = run_epoch(model, validation_loader, optimizer, device, train=False)
        print(f"epoch={epoch} train={training} validation={validation}")
        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            torch.save({"model": model.state_dict(), "config": cfg}, destination / "best.pt")


if __name__ == "__main__":
    main()
