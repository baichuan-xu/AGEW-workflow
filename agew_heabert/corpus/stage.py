"""Corpus-building stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agew_heabert.corpus.processing import extract_xml_text, normalize_corpus_text
from agew_heabert.stage_utils import check_result, require_paths


def run(config: dict[str, Any], *, check_only: bool = False) -> dict[str, Any]:
    input_dir = Path(config["input_dir"])
    output_path = Path(config["output_path"])
    if check_only:
        return check_result("build-corpus", [input_dir, output_path.parent])
    require_paths({"input_dir": input_dir})
    paragraphs: list[str] = []
    for xml_path in sorted(input_dir.rglob("*.xml")):
        paragraphs.extend(extract_xml_text(xml_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        normalize_corpus_text("\n".join(paragraphs)) + "\n",
        encoding="utf-8",
    )
    return {"stage": "build-corpus", "status": "completed", "paragraphs": len(paragraphs)}
