from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


_REFERENCE = re.compile(r"^\$\{([A-Za-z0-9_.-]+)\}$")


def _lookup(cfg: dict[str, Any], dotted_key: str) -> Any:
    value: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Unknown configuration reference: ${{{dotted_key}}}")
        value = value[part]
    return value


def _expand_references(value: Any, cfg: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _expand_references(item, cfg) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_references(item, cfg) for item in value]
    if isinstance(value, str):
        match = _REFERENCE.match(value)
        if match:
            return _lookup(cfg, match.group(1))
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _expand_references(cfg, cfg)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    return cfg


def resolve_config_path(cfg: dict[str, Any], value: str | Path | None) -> str | None:
    """Resolve relative paths against the directory containing the YAML config."""
    if value is None or str(value) == "":
        return None
    value = Path(value)
    if value.is_absolute():
        return str(value)
    config_dir = Path(cfg.get("_config_dir", "."))
    return str((config_dir / value).resolve())
