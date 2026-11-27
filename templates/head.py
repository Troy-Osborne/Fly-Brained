"""Head geometry template. Edit this file to change head proportions."""
from dataclasses import dataclass

@dataclass(frozen=True)
class HeadTemplate:
    length: float = 0.52
    height: float = 0.48
    width: float = 0.50
    local_position: tuple = (0.60, 0.08, 0.0)
    colour: tuple = (0.36, 0.25, 0.18)

DEFAULT_HEAD = HeadTemplate()
