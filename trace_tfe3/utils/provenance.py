from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a recorded experiment."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - PyTorch is a runtime dependency
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def git_commit(repository: str | Path = ".") -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_run_record(
    inputs: Iterable[str | Path],
    repository: str | Path = ".",
    extra: dict | None = None,
) -> dict:
    files = {}
    for value in inputs:
        path = Path(value).resolve()
        files[str(path)] = sha256_file(path)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repository),
        "python": sys.version,
        "platform": platform.platform(),
        "input_sha256": files,
        **(extra or {}),
    }


def write_run_record(record: dict, output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return destination
