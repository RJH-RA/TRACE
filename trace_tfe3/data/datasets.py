from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CT_PHASES = ("noncontrast", "arterial")
DEFAULT_PATHOLOGY_DIM = 1536


def _split_paths(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        return [str(item) for item in json.loads(text)]
    return [item.strip() for item in text.split(";") if item.strip()]


def load_tensor(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("tokens", "embeddings", "embedding", "volume", "array", "tensor"):
                if key in obj:
                    obj = obj[key]
                    break
        return torch.as_tensor(obj).float()
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    if path.suffix == ".npz":
        data = np.load(path)
        key = "volume" if "volume" in data else list(data.keys())[0]
        return torch.from_numpy(data[key]).float()
    raise ValueError(f"Unsupported tensor file: {path}")


def _as_volume(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(tensor).float()
    while tensor.dim() > 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 3:
        raise ValueError(f"CT volume must be 3D after singleton removal, got {tuple(tensor.shape)}")
    return tensor


def load_pathology_tokens(paths: list[str]) -> torch.Tensor:
    if not paths:
        return torch.empty(0, DEFAULT_PATHOLOGY_DIM)
    token_sets: list[torch.Tensor] = []
    for path in paths:
        tokens = load_tensor(path)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        elif tokens.dim() > 2:
            tokens = tokens.flatten(0, -2)
        token_sets.append(tokens)
    feature_dims = {tokens.shape[-1] for tokens in token_sets}
    if len(feature_dims) != 1:
        raise ValueError(f"Pathology token dimensions differ: {sorted(feature_dims)}")
    return torch.cat(token_sets, dim=0)


class TRACECaseDataset(Dataset):
    """Patient-level paired-CT dataset with training-only H&E token support."""

    required_columns = {"patient_id", "split", "label", *CT_PHASES}

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        require_pathology: bool = False,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        frame = pd.read_csv(self.manifest_csv)
        missing = self.required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        self.df = frame[frame["split"].astype(str) == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split={split!r} in {manifest_csv}")
        self.require_pathology = require_pathology

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        noncontrast = _as_volume(load_tensor(row["noncontrast"]))
        arterial = _as_volume(load_tensor(row["arterial"]))
        if noncontrast.shape != arterial.shape:
            raise ValueError(
                f"Registered CT shapes differ for {row['patient_id']}: "
                f"{tuple(noncontrast.shape)} vs {tuple(arterial.shape)}"
            )
        enhancement = arterial - noncontrast
        ct = torch.stack([noncontrast, arterial, enhancement], dim=0)

        pathology_paths = _split_paths(row.get("he_token_embeddings", ""))
        if self.require_pathology and not pathology_paths:
            raise ValueError(f"Training row {row['patient_id']} has no H&E token embeddings")
        pathology_tokens = load_pathology_tokens(pathology_paths)

        return {
            "patient_id": str(row["patient_id"]),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "ct": ct,
            "pathology_tokens": pathology_tokens,
        }


def _pad_tokens(token_sets: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    dim = max((tokens.shape[-1] if tokens.numel() else DEFAULT_PATHOLOGY_DIM) for tokens in token_sets)
    max_tokens = max((tokens.shape[0] for tokens in token_sets), default=0)
    padded = torch.zeros(len(token_sets), max_tokens, dim)
    mask = torch.zeros(len(token_sets), max_tokens, dtype=torch.bool)
    for index, tokens in enumerate(token_sets):
        if tokens.numel() == 0:
            continue
        if tokens.shape[-1] != dim:
            raise ValueError("Pathology token dimensions must match within a batch")
        padded[index, : tokens.shape[0]] = tokens
        mask[index, : tokens.shape[0]] = True
    return padded, mask


def trace_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pathology_tokens, pathology_mask = _pad_tokens(
        [item["pathology_tokens"] for item in batch]
    )
    return {
        "patient_id": [item["patient_id"] for item in batch],
        "label": torch.stack([item["label"] for item in batch]),
        "ct": torch.stack([item["ct"] for item in batch]),
        "pathology_tokens": pathology_tokens,
        "pathology_mask": pathology_mask,
    }
