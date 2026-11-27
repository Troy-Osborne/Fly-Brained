"""Pure wing shape, placement and appearance template."""
from dataclasses import dataclass

@dataclass(frozen=True)
class WingTemplate:
    length: float = 1.05
    thickness: float = 0.035
    chord: float = 0.36
    shoulder_x: float = 0.02
    shoulder_y: float = 0.19
    shoulder_z: float = 0.25
    span: float = 1.00
    sweep_x: float = -0.16
    colour: tuple = (0.70, 0.82, 0.88)
    opacity: float = 0.38

DEFAULT_WING_TEMPLATE = WingTemplate()
