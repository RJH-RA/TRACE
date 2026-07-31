#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_model(cfg: dict):
    from trace_tfe3.models import TRACECTEncoder, TRACEModel, build_dinov3_backbone
    from trace_tfe3.utils.config import resolve_config_path

    encoder_cfg = cfg["ct_encoder"]
    checkpoint = resolve_config_path(cfg, cfg["paths"].get("dinov3_checkpoint"))
    backbone = build_dinov3_backbone(
        model_name=encoder_cfg.get("timm_model_name", "vit_base_patch16_224"),
        checkpoint=checkpoint,
        in_channels=encoder_cfg.get("input_channels", 3),
    )
    encoder = TRACECTEncoder(
        backbone=backbone,
        backbone_output_dim=encoder_cfg.get("backbone_output_dim", 768),
        shared_dim=encoder_cfg.get("shared_dim", 256),
        token_count=encoder_cfg.get("token_count", 16),
        transformer_layers=encoder_cfg.get("transformer_layers", 2),
        transformer_heads=encoder_cfg.get("transformer_heads", 8),
        freeze_backbone=encoder_cfg.get("frozen_backbone", True),
    )
    return TRACEModel(encoder, shared_dim=encoder_cfg.get("shared_dim", 256))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRACE-CT with development-only H&E")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pathology-checkpoint", required=True)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from trace_tfe3.data import TRACECaseDataset, trace_collate
    from trace_tfe3.models import PathologyTeacher
    from trace_tfe3.training import evaluate, train_one_epoch
    from trace_tfe3.utils.config import load_config

    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda"))
    manifest = cfg["data"]["manifest_csv"]
    train_set = TRACECaseDataset(
        manifest,
        cfg["data"].get("train_split", "train"),
        require_pathology=True,
    )
    validation_set = TRACECaseDataset(
        manifest,
        cfg["data"].get("validation_split", "validation"),
        require_pathology=False,
    )
    loader_args = {
        "batch_size": cfg["training"].get("batch_size", 8),
        "num_workers": cfg["training"].get("num_workers", 4),
        "collate_fn": trace_collate,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)

    pathology_cfg = cfg["pathology"]
    teacher = PathologyTeacher(
        token_embedding_dim=pathology_cfg.get("token_embedding_dim", 1536),
        shared_dim=pathology_cfg.get("shared_dim", 256),
        hidden_dim=pathology_cfg.get("patient_hidden_dim", 256),
    )
    state = torch.load(args.pathology_checkpoint, map_location="cpu")
    teacher.load_state_dict(state.get("model", state))
    teacher.to(device).eval()

    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=cfg["training"].get("learning_rate", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )
    epochs = cfg["training"].get("epochs", 100)
    warmup = cfg["training"].get("classification_warmup_epochs", 20)
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            teacher,
            train_loader,
            optimizer,
            device,
            use_pathology_transfer=epoch > warmup,
            asrot_config=cfg["asrot"],
        )
        validation_metrics = evaluate(model, validation_loader, device)
        print(f"epoch={epoch} train={train_metrics} validation={validation_metrics}")

    output = Path(cfg["output_dir"]) / "trace_ct"
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg}, output / "final.pt")


if __name__ == "__main__":
    main()
