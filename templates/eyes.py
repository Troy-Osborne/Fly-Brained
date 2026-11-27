"""Compound-eye and ommatidia patch template."""
from dataclasses import dataclass

@dataclass(frozen=True)
class EyeTemplate:
    radius: float = 0.19
    colour: tuple = (0.75, 0.12, 0.10)
    centre: tuple = (0.70, 0.20, 0.19)
    ommatidia_radius: float = 0.018
    ommatidia_colour: tuple = (1.0, 0.48, 0.25)
    columns: int = 5
    rows: int = 3
    column_spacing: float = 0.06
    row_spacing: float = 0.055
    lateral_offset: float = 0.045

DEFAULT_EYES = EyeTemplate()
