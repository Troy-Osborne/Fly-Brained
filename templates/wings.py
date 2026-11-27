"""Replaceable VPython wing assembly and appearance/shape template."""
from __future__ import annotations

from vpython import ellipsoid, vector, rotate

from .wing_geometry import DEFAULT_WING_TEMPLATE, WingTemplate


class WingAssembly:
    """One wing, separated from the body and from its motion model."""

    def __init__(self, side: int, template: WingTemplate = DEFAULT_WING_TEMPLATE):
        if side not in (-1, 1):
            raise ValueError("wing side must be -1 or +1")
        self.side = side
        self.template = template
        self.visual = ellipsoid(
            pos=vector(0, 0, 0), length=template.length, height=template.thickness,
            width=template.chord, color=vector(*template.colour), opacity=template.opacity,
        )

    def update(self, root, yaw: float, wing_motion, activation: float, local_to_world):
        t = self.template
        shoulder = local_to_world(vector(t.shoulder_x, t.shoulder_y, self.side * t.shoulder_z))
        axis_local, up_local = wing_motion.wing_frame(
            self.side, activation, span=t.span, sweep_x=t.sweep_x
        )
        self.visual.pos = shoulder
        self.visual.axis = rotate(vector(*axis_local), angle=yaw, axis=vector(0, 1, 0))
        self.visual.up = rotate(vector(*up_local), angle=yaw, axis=vector(0, 1, 0))
