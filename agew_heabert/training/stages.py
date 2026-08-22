"""Static-review entry points for manuscript model scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agew_heabert.stage_utils import check_result, require_paths


def _training_result(stage: str, config: dict[str, Any], check_only: bool) -> dict[str, Any]:
    paths = [
        config["dataset"],
        config["encoder"],
        config.get("output_dir", config.get("output_path", ".")),
    ]
    if check_only:
        return check_result(
            stage,
            paths,
            reference_implementation="See the canonical mapping in docs/LEGACY_MAP.csv",
            limitation="Research datasets and checkpoints are intentionally not distributed.",
        )
    require_paths({"dataset": config["dataset"], "encoder": config["encoder"]})
    raise RuntimeError(
        "This code-only release supports static review and configuration validation. "
        "Full training uses the preserved manuscript implementation identified in "
        "docs/LEGACY_MAP.csv."
    )


def run_yield_strength(config: dict[str, Any], *, check_only: bool = False) -> dict[str, Any]:
    return _training_result("train-yield-strength", config, check_only)


def run_stress_strain(config: dict[str, Any], *, check_only: bool = False) -> dict[str, Any]:
    return _training_result("train-stress-strain", config, check_only)


def run_prediction(config: dict[str, Any], *, check_only: bool = False) -> dict[str, Any]:
    paths = [
        config["dataset"],
        config["encoder"],
        config["checkpoint"],
        Path(config["output_path"]).parent,
    ]
    if check_only:
        return check_result("predict", paths)
    require_paths(
        {
            "dataset": config["dataset"],
            "encoder": config["encoder"],
            "checkpoint": config["checkpoint"],
        }
    )
    raise RuntimeError(
        "Prediction requires non-distributed checkpoints; use the mapped manuscript "
        "inference scripts."
    )
