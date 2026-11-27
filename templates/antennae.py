"""Antenna and arista/club template."""
from dataclasses import dataclass

@dataclass(frozen=True)
class AntennaTemplate:
    base: tuple = (0.77, 0.34, 0.10)
    end_x: float = 0.93
    end_y: float = 0.52
    lateral_end: float = 0.19
    shaft_radius: float = 0.018
    club_radius: float = 0.045
    shaft_colour: tuple = (0.56, 0.42, 0.26)
    club_colour: tuple = (0.58, 0.43, 0.25)

DEFAULT_ANTENNA = AntennaTemplate()
