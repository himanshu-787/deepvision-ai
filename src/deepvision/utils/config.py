"""YAML configuration loading with overrides."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

BASE_CONFIG = Path("configs/base.yaml")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load configs/base.yaml, merge the experiment YAML over it, then apply 'a.b=value' overrides."""
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)            # "0.001" -> float, "true" -> bool, "null" -> None
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    """Short, stable fingerprint of a resolved config."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
