"""Prompt inventory validation; prompt text is never synthesized."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path


def validate_prompt_manifest(
    prompt_root: str | Path,
    manifest: Mapping[str, Sequence[str]],
) -> list[str]:
    root = Path(prompt_root)
    missing: list[str] = []
    for stage, names in manifest.items():
        for name in names:
            relative = Path(stage) / name
            if not (root / relative).is_file():
                missing.append(relative.as_posix())
    return sorted(missing)
