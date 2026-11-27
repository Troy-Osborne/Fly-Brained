"""Lumped antagonistic actuators in model length and force units."""
import math
import numpy as np


class TendonPair:
    """One forward and one reverse tensile branch per leg."""
    stiffness = 12.0
    moment_arm = 0.035
    relaxation_s = 0.060
    peak_force = 1.5
    slack_length = 0.28

    def __init__(self, angles):
        self.extension = np.zeros((len(angles), 2))
        self.angles = np.array(angles, dtype=float)

    def step(self, angles, rates, available_activation, dt):
        # Signed extension is relative to slack. Negative extension stores no energy.
        # A moving joint lengthens one branch and shortens the other.
        angles = np.asarray(angles)
        excursion = self.moment_arm * (angles - self.angles)
        self.angles = angles.copy()
        drive = np.column_stack((np.maximum(rates, 0), np.maximum(-rates, 0)))
        target = drive * available_activation[:, None] * self.peak_force / self.stiffness
        # Exact update for constant drive and linearly changing joint angle:
        # dx/dt = (target-x)/tau + signed moment_arm * angular_velocity.
        gain = -math.expm1(-dt / self.relaxation_s)
        self.extension += gain * (target - self.extension)
        self.extension += (gain * self.relaxation_s / dt
                           * np.column_stack((excursion, -excursion)))
        return self.stiffness * np.maximum(self.extension, 0)
