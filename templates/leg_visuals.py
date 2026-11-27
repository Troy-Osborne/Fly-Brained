"""Pure visual dimensions and colours for the leg assembly."""
from dataclasses import dataclass

@dataclass(frozen=True)
class LegVisualTemplate:
    femur_radius: float = 0.075
    tibia_radius: float = 0.045
    tarsus_radius: float = 0.025
    hip_radius: float = 0.09
    knee_radius: float = 0.055
    ankle_radius: float = 0.035
    foot_radius: float = 0.035
    flexor_radius: float = 0.025
    extensor_radius: float = 0.022
    tendon_radius: float = 0.012
    ligament_radius: float = 0.009
    bristle_radius: float = 0.006
    flexor_colour: tuple = (0.92, 0.30, 0.16)
    extensor_colour: tuple = (0.95, 0.62, 0.20)
    exoskeleton_colour: tuple = (0.48, 0.30, 0.16)
    tibia_colour: tuple = (0.65, 0.46, 0.24)
    tarsus_colour: tuple = (0.75, 0.62, 0.39)
    tendon_colour: tuple = (0.95, 0.86, 0.55)
    ligament_colour: tuple = (0.30, 0.78, 0.65)

DEFAULT_LEG_VISUAL = LegVisualTemplate()
