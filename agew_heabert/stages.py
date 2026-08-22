"""Canonical stage registry and review-time validation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    handler: str
    required_config_keys: tuple[str, ...]
    requires_network: bool = False
    requires_gpu_for_full_run: bool = False
    description: str = ""


STAGE_SPECS: dict[str, StageSpec] = {
    "acquire-literature": StageSpec(
        "agew_heabert.acquisition.elsevier:run",
        ("query", "output_dir"),
        requires_network=True,
        description="Acquire licensed article XML and metadata from Elsevier APIs.",
    ),
    "build-corpus": StageSpec(
        "agew_heabert.corpus.stage:run",
        ("input_dir", "output_path"),
        description="Build a normalized text corpus from article XML.",
    ),
    "extract-yield-strength": StageSpec(
        "agew_heabert.extraction.stages:run_yield_strength",
        ("input_dir", "output_dir", "prompt_dir"),
        requires_network=True,
        description="Run AGEW and CoD for yield-strength records.",
    ),
    "extract-stress-strain": StageSpec(
        "agew_heabert.extraction.stages:run_stress_strain",
        ("input_dir", "output_dir", "prompt_dir"),
        requires_network=True,
        description="Run AGEW and CoD for stress-strain curves.",
    ),
    "pretrain-heabert": StageSpec(
        "agew_heabert.models.heabert:run",
        ("train_path", "validation_path", "base_model", "base_model_revision", "output_dir"),
        requires_gpu_for_full_run=True,
        description="Continue masked-language pretraining to produce HeaBERT.",
    ),
    "train-yield-strength": StageSpec(
        "agew_heabert.training.stages:run_yield_strength",
        ("dataset", "encoder", "output_dir"),
        requires_gpu_for_full_run=True,
        description="Train and evaluate the multimodal yield-strength model.",
    ),
    "train-stress-strain": StageSpec(
        "agew_heabert.training.stages:run_stress_strain",
        ("dataset", "encoder", "output_dir"),
        requires_gpu_for_full_run=True,
        description="Train and evaluate the multimodal stress-strain model.",
    ),
    "predict": StageSpec(
        "agew_heabert.training.stages:run_prediction",
        ("dataset", "encoder", "checkpoint", "output_path"),
        requires_gpu_for_full_run=True,
        description="Apply a trained multimodal model.",
    ),
    "inverse-design": StageSpec(
        "agew_heabert.inverse_design.stage:run",
        ("checkpoint", "encoder", "process_text_path", "composition_path", "output_path"),
        requires_gpu_for_full_run=True,
        description="Optimize alloy composition against the trained yield-strength model.",
    ),
    "validate": StageSpec(
        "agew_heabert.validation:run",
        (),
        description="Validate configuration, prompts, paths, and external artifacts.",
    ),
}


def get_stage_spec(name: str) -> StageSpec:
    try:
        return STAGE_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown stage: {name}") from exc


def validate_config_for_stage(name: str, config: dict[str, Any]) -> None:
    spec = get_stage_spec(name)
    missing = [key for key in spec.required_config_keys if key not in config]
    if missing:
        raise ValueError(f"Missing configuration keys for {name}: {', '.join(missing)}")


def execute_stage(name: str, config: dict[str, Any], check_only: bool = False) -> Any:
    spec = get_stage_spec(name)
    validate_config_for_stage(name, config)
    module_name, function_name = spec.handler.split(":", maxsplit=1)
    handler = getattr(importlib.import_module(module_name), function_name)
    return handler(config, check_only=check_only)
