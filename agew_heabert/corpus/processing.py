"""Deterministic XML-to-corpus transformations."""

from __future__ import annotations

import re
from pathlib import Path

from defusedxml import ElementTree as ET

WHITESPACE = re.compile(r"[ \t]+")
TARGET_TAGS = {"description", "simple-para", "caption", "para"}


def normalize_corpus_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("–", "-")
    lines = [WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def extract_xml_text(path: str | Path) -> list[str]:
    root = ET.parse(Path(path)).getroot()
    seen: set[str] = set()
    paragraphs: list[str] = []
    for element in root.iter():
        local_name = element.tag.rsplit("}", maxsplit=1)[-1]
        if local_name not in TARGET_TAGS:
            continue
        text = normalize_corpus_text(" ".join(element.itertext()))
        if text and text not in seen:
            seen.add(text)
            paragraphs.append(text)
    return paragraphs
