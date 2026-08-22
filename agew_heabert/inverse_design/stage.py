"""Inverse-design stage adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agew_heabert.stage_utils import check_result, require_paths


def run(config: dict[str, Any], *, check_only: bool = False) -> dict[str, Any]:
    paths = [
        config["checkpoint"],
        config["encoder"],
        config["process_text_path"],
        config["composition_path"],
        Path(config["output_path"]).parent,
    ]
    if check_only:
        return check_result(
            "inverse-design",
            paths,
            constraints={
                "composition_sum": config.get("composition_sum", 100.0),
                "learning_rate": config.get("learning_rate", 0.01),
                "max_iterations": config.get("max_iterations", 1000),
            },
        )
    require_paths(
        {
            "checkpoint": config["checkpoint"],
            "encoder": config["encoder"],
            "process_text_path": config["process_text_path"],
            "composition_path": config["composition_path"],
        }
    )
    raise RuntimeError(
        "Inverse design requires the non-distributed trained checkpoint; "
        "the exact implementation is preserved at legacy/Inverse/gradient.py."
    )
