"""Deterministic graph-axis calibration used by AGEW."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearAxis:
    pixel_min: float
    pixel_max: float
    value_min: float
    value_max: float

    def __post_init__(self) -> None:
        if self.pixel_min == self.pixel_max:
            raise ValueError("pixel range cannot be zero")

    def transform(self, pixel: float) -> float:
        fraction = (pixel - self.pixel_min) / (self.pixel_max - self.pixel_min)
        return self.value_min + fraction * (self.value_max - self.value_min)


def pixel_to_data(
    x_pixel: float,
    y_pixel: float,
    *,
    x_axis: LinearAxis,
    y_axis: LinearAxis,
) -> tuple[float, float]:
    return x_axis.transform(x_pixel), y_axis.transform(y_pixel)
