"""Repository and external-artifact validation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agew_heabert.stage_utils import check_result


def run(config: dict[str, Any], *, check_only: bool = False) -> dict[str, Any]:
    paths = [
        value for key, value in config.items() if key.endswith("_path") or key.endswith("_dir")
    ]
    if check_only:
        return check_result("validate", paths)
    missing = [str(Path(path)) for path in paths if not Path(path).exists()]
    return {"stage": "validate", "status": "completed", "missing_artifacts": missing}
