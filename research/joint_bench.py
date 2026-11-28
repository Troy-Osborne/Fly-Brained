"""Standalone torque-driven hinge; parameters use illustrative model units."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class JointState:
    angle: float = 0.
    velocity: float = 0.
    activation: float = 0.


DT = 0.001
ARM = 0.10
STIFFNESS = 50.
INERTIA = 0.009
DAMPING = 0.045
PRELOAD = 0.012
LIMIT = 1.25


def tensions(state):
    # Positive activation shortens the forward muscle's rest length.
    # The two tendon paths wind in opposite directions around the hinge.
    forward = PRELOAD + 0.16 * max(state.activation, 0.) - ARM * state.angle
    reverse = PRELOAD + 0.16 * max(-state.activation, 0.) + ARM * state.angle
    return STIFFNESS * max(0., forward), STIFFNESS * max(0., reverse)


def advance(state, command, stop, contact, dt=DT):
    """Return post-step state and mean obstacle reaction torque over this step."""
    if contact not in ('free', 'compliant', 'rigid'):
        raise ValueError('unknown contact model')
    if not (-1 <= command <= 1 and 0 < dt <= DT):
        raise ValueError('command or integration step outside bench range')
    activation = state.activation + (command-state.activation) * -math.expm1(-dt/.035)
    forward, reverse = tensions(replace(state, activation=activation))
    reaction = 0.
    if contact == 'compliant' and state.angle > stop:
        reaction = max(0., 6. * (state.angle-stop) + .08 * state.velocity)
    torque = ARM * (forward-reverse) - DAMPING * state.velocity - reaction
    velocity = state.velocity + dt * torque / INERTIA
    angle = state.angle + dt * velocity
    if contact == 'rigid' and angle >= stop and velocity > 0:
        reaction = INERTIA * velocity / dt
        angle, velocity = stop, 0.
    if abs(angle) >= LIMIT and angle * velocity > 0:
        angle, velocity = math.copysign(LIMIT, angle), 0.
    return JointState(angle, velocity, activation), reaction
