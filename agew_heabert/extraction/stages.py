"""Reviewer-facing AGEW/CoD stage validation and run manifests."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from agew_heabert.config import require_environment
from agew_heabert.prompts import validate_prompt_manifest
from agew_heabert.stage_utils import check_result, require_paths

YIELD_PROMPTS = {
    "yield_strength": [
        "extract_material_text.md",
        "extract_table_llm.md",
        "extract_figure_llm.md",
        "composition_compute.md",
        "data+composition.md",
    ]
}
STRESS_STRAIN_PROMPTS = {
    "stress_strain": [
        "estimate_data.md",
        "figure_split.md",
        "line_label.md",
        "material_composition.md",
        "material_name.md",
        "material_process.md",
        "organize_label.md",
        "split_check.md",
    ]
}


def _run(
    stage: str,
    config: dict[str, Any],
    manifest: dict[str, list[str]],
    *,
    check_only: bool,
) -> dict[str, Any]:
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    prompt_dir = Path(config["prompt_dir"])
    missing_prompts = validate_prompt_manifest(prompt_dir, manifest)
    model_ids = config.get(
        "model_ids",
        ["qwen-max-latest", "qwen-vl-max", "qwen2.5-vl-72b-instruct"],
    )
    if check_only:
        return check_result(
            stage,
            [input_dir, prompt_dir, output_dir],
            missing_prompts=missing_prompts,
            model_ids=model_ids,
            limitation="Moving hosted-model aliases may produce different future outputs.",
        )
    require_paths({"input_dir": input_dir, "prompt_dir": prompt_dir})
    if missing_prompts:
        raise FileNotFoundError("Original prompts are missing: " + ", ".join(missing_prompts))
    require_environment("DASHSCOPE_API_KEY")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "stage": stage,
        "provider": "Alibaba Cloud DashScope",
        "model_ids": model_ids,
        "request_date": date.today().isoformat(),
        "input_dir": str(input_dir),
        "prompt_dir": str(prompt_dir),
        "status": "prepared",
        "implementation_note": "Run the mapped AGEW/CoD algorithms listed in docs/LEGACY_MAP.csv.",
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return {"stage": stage, "status": "prepared", "manifest": str(path)}


def run_yield_strength(config: dict[str, Any], *, check_only: bool = False) -> dict[str, Any]:
    return _run("extract-yield-strength", config, YIELD_PROMPTS, check_only=check_only)


def run_stress_strain(config: dict[str, Any], *, check_only: bool = False) -> dict[str, Any]:
    return _run("extract-stress-strain", config, STRESS_STRAIN_PROMPTS, check_only=check_only)
