"""Small, dependency-free records shared across manuscript stages."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

ELEMENT = re.compile(r"^[A-Z][a-z]?$")


@dataclass(frozen=True)
class Composition:
    """An alloy composition normalized to atomic percent."""

    components: dict[str, float]

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> Composition:
        if not values:
            raise ValueError("composition cannot be empty")
        normalized: dict[str, float] = {}
        for element, raw in values.items():
            if not ELEMENT.fullmatch(element):
                raise ValueError(f"invalid element symbol: {element}")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("composition values must be finite")
            if value < 0:
                raise ValueError("composition values must be non-negative")
            normalized[element] = value
        total = sum(normalized.values())
        if math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-8):
            normalized = {key: value * 100.0 for key, value in normalized.items()}
        elif not math.isclose(total, 100.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("composition values must sum to 1 or 100")
        return cls(normalized)

    @property
    def total(self) -> float:
        return sum(self.components.values())

    def to_dict(self) -> dict[str, object]:
        return {"components_at_percent": dict(self.components), "total": self.total}


@dataclass(frozen=True)
class StressStrainPoint:
    strain: float
    stress_mpa: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.strain) or self.strain < 0:
            raise ValueError("strain must be finite and non-negative")
        if not math.isfinite(self.stress_mpa):
            raise ValueError("stress must be finite")


@dataclass(frozen=True)
class ArticleRecord:
    doi: str
    title: str
    source: str = "Elsevier"
    xml_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedPropertyRecord:
    material_name: str
    composition: dict[str, float]
    processing_text: str
    property_name: str
    value: float
    unit: str
    source_doi: str
    source_kind: str
    source_identifier: str
    extraction_method: str
    model_id: str | None = None
    request_date: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
