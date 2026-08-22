"""Shared helpers for side-effect-free stage checks."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def check_result(stage: str, paths: Iterable[str | Path] = (), **details: Any) -> dict[str, Any]:
    artifacts = []
    for raw in paths:
        path = Path(raw)
        artifacts.append({"path": str(path), "exists": path.exists()})
    return {"stage": stage, "status": "check-only", "artifacts": artifacts, **details}


def require_paths(named_paths: dict[str, str | Path]) -> None:
    missing = [
        f"{name}: {Path(path)}" for name, path in named_paths.items() if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError("Required external artifacts are missing: " + "; ".join(missing))
