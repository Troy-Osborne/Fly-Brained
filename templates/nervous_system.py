"""Stylised neural anatomy display template (not a connectome rendering)."""
from dataclasses import dataclass

@dataclass(frozen=True)
class NervousSystemTemplate:
    brain_radius: float = 0.105
    brain_colour: tuple = (0.66, 0.78, 0.42)
    nerve_colour: tuple = (0.48, 0.88, 0.72)
    sensory_nerve_colour: tuple = (0.50, 0.77, 0.92)
    cord_front_x: float = 0.30
    cord_back_x: float = -1.12
    cord_y: float = -0.19
    cord_radius: float = 0.035
    ganglion_x: tuple = (0.08, -0.28, -0.64, -0.96)
    ganglion_radius: float = 0.055
    motor_nerve_radius: float = 0.012
    sensory_nerve_radius: float = 0.010

DEFAULT_NERVOUS_SYSTEM = NervousSystemTemplate()
