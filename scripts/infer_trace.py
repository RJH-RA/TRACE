#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pathology-free TRACE-CT inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="external_test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from scripts.train_trace import make_model
    from trace_tfe3.data import TRACECaseDataset, trace_collate
    from trace_tfe3.utils.config import load_config

    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda"))
    dataset = TRACECaseDataset(cfg["data"]["manifest_csv"], args.split, require_pathology=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=trace_collate)
    model = make_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.to(device).eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            score = torch.sigmoid(model(batch["ct"].to(device))["logit"]).cpu().item()
            rows.append({"patient_id": batch["patient_id"][0], "trace_ct_score": score})

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient_id", "trace_ct_score"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
