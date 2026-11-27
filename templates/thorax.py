"""Thorax and indirect flight-muscle template."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ThoraxTemplate:
    length: float = 0.95
    height: float = 0.54
    width: float = 0.72
    colour: tuple = (0.33, 0.23, 0.16)
    flight_muscle_radius: float = 0.035
    flight_muscle_colour: tuple = (0.91, 0.34, 0.17)

    # Each fibre is (local start, local direction delta); edit independently.
    flight_muscle_paths: tuple = (
        ((0.24, 0.18, -0.18), (-0.18, -0.13, 0.18)),
        ((0.24, 0.18, 0.18), (-0.18, -0.13, -0.18)),
        ((-0.18, 0.17, -0.18), (0.22, -0.13, 0.18)),
        ((-0.18, 0.17, 0.18), (0.22, -0.13, -0.18)),
    )

DEFAULT_THORAX = ThoraxTemplate()
