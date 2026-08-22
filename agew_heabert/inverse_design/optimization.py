"""Composition constraints shared by gradient-based inverse design."""

from __future__ import annotations

from collections.abc import Mapping


def project_composition(values: Mapping[str, float]) -> dict[str, float]:
    """Project arbitrary values onto a non-negative 100 at.% composition."""
    if not values:
        raise ValueError("composition cannot be empty")
    clipped = {element: max(0.0, float(value)) for element, value in values.items()}
    total = sum(clipped.values())
    if total <= 0:
        raise ValueError("composition must contain at least one positive value")
    return {element: value * 100.0 / total for element, value in clipped.items()}
