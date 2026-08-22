"""Configuration loading with repository-relative path resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def require_environment(name: str) -> str:
    """Return a required environment variable without an insecure fallback."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


ARTIFACT_KEYS = {"dataset", "encoder", "checkpoint", "base_model"}


def _resolve_paths(value: Any, base_dir: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _resolve_paths(v, base_dir, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base_dir, key) for item in value]
    if isinstance(value, str) and (
        key.endswith("_path") or key.endswith("_dir") or key in ARTIFACT_KEYS
    ):
        if key == "base_model" and not value.startswith((".", "/")) and ":\\" not in value:
            return value
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        return str(candidate.resolve())
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML and resolve fields named ``*_path``/``*_dir``."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configuration requires PyYAML") from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError("Configuration must be JSON or YAML")
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    return _resolve_paths(data, config_path.parent)
