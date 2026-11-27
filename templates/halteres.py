"""Halteres: stalk and knob visual templates."""
from dataclasses import dataclass

@dataclass(frozen=True)
class HaltereTemplate:
    base: tuple = (-0.48, 0.16, 0.24)
    tip: tuple = (-0.70, 0.22, 0.48)
    stalk_radius: float = 0.018
    knob_radius: float = 0.075
    stalk_colour: tuple = (0.65, 0.52, 0.31)
    knob_colour: tuple = (0.86, 0.75, 0.45)

DEFAULT_HALTERE = HaltereTemplate()
