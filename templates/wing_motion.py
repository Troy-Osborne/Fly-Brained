"""Independent wing oscillator: replace this module to experiment with stroke models.

The default oscillator is a visual/kinematic placeholder. Rates are radians
per second, not measured Drosophila wingbeat frequency.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class WingMotionTemplate:
    base_angular_rate: float = 24.0
    activation_angular_rate: float = 170.0
    passive_stroke_amplitude_rad: float = 0.06
    active_stroke_amplitude_rad: float = 0.36


DEFAULT_WING_MOTION = WingMotionTemplate()


class WingMotionModel:
    """Phase oscillator whose output is mirrored across the fly's sagittal plane."""

    def __init__(self, template: WingMotionTemplate = DEFAULT_WING_MOTION):
        self.template = template
        self.phase = 0.0

    def step(self, dt: float, activation: float) -> float:
        level = max(0.0, min(1.0, float(activation)))
        rate = self.template.base_angular_rate + self.template.activation_angular_rate * level
        self.phase = (self.phase + max(0.0, dt) * rate) % (2.0 * math.pi)
        return self.phase

    def wing_frame(self, side: int, activation: float, span: float = 1.0,
                   sweep_x: float = -0.16) -> tuple[Vec3, Vec3]:
        """Return local wing long-axis and up vectors for one side.

        The left/right frames are mirror images: their Z components have
        opposite signs while their vertical flap component is identical.
        """
        if side not in (-1, 1):
            raise ValueError("wing side must be -1 or +1")
        level = max(0.0, min(1.0, float(activation)))
        amplitude = (self.template.passive_stroke_amplitude_rad
                     + self.template.active_stroke_amplitude_rad * level)
        stroke = math.sin(self.phase) * amplitude
        c, s = math.cos(stroke), math.sin(stroke)
        axis = (sweep_x, -s * span, side * c * span)
        up = (0.0, c, side * s)
        return axis, up
