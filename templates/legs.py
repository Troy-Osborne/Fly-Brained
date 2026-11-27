"""VPython renderer for one replaceable three-segment leg template."""
from __future__ import annotations

import math
from typing import Optional

from vpython import color, curve, cylinder, sphere, vector, mag, rotate

from .leg_geometry import DEFAULT_LEG_TEMPLATE, LegGeometryTemplate, local_leg_points
from .leg_visuals import DEFAULT_LEG_VISUAL, LegVisualTemplate


class LegAssembly:
    """One leg with replaceable geometry and appearance templates."""

    def __init__(
        self,
        name: str,
        index: int,
        side: int,
        geometry: LegGeometryTemplate = DEFAULT_LEG_TEMPLATE,
        visual: LegVisualTemplate = DEFAULT_LEG_VISUAL,
    ):
        self.name = name
        self.index = index
        self.side = side
        self.geometry_template = geometry
        self.visual_template = visual
        attach = local_leg_points(index, side, 0.0, template=geometry)["attach"]
        self.attach = vector(*attach)

        self.femur = cylinder(pos=vector(0, 0, 0), axis=vector(0.1, 0, 0),
                              radius=visual.femur_radius, color=vector(*visual.exoskeleton_colour))
        self.tibia = cylinder(pos=vector(0, 0, 0), axis=vector(0.1, 0, 0),
                              radius=visual.tibia_radius, color=vector(*visual.tibia_colour))
        self.tarsus = cylinder(pos=vector(0, 0, 0), axis=vector(0.1, 0, 0),
                               radius=visual.tarsus_radius, color=vector(*visual.tarsus_colour))
        self.hip = sphere(pos=vector(0, 0, 0), radius=visual.hip_radius,
                          color=vector(0.52, 0.34, 0.20))
        self.knee = sphere(pos=vector(0, 0, 0), radius=visual.knee_radius,
                           color=vector(0.68, 0.48, 0.27))
        self.ankle = sphere(pos=vector(0, 0, 0), radius=visual.ankle_radius,
                            color=vector(0.68, 0.48, 0.27))
        self.foot = sphere(pos=vector(0, 0, 0), radius=visual.foot_radius,
                           color=vector(0.88, 0.72, 0.45))
        self.flexor = cylinder(pos=vector(0, 0, 0), axis=vector(0.1, 0, 0),
                               radius=visual.flexor_radius, color=vector(*visual.flexor_colour))
        self.extensor = cylinder(pos=vector(0, 0, 0), axis=vector(0.1, 0, 0),
                                 radius=visual.extensor_radius, color=vector(*visual.extensor_colour))
        self.tendon = cylinder(pos=vector(0, 0, 0), axis=vector(0.1, 0, 0),
                               radius=visual.tendon_radius, color=vector(*visual.tendon_colour))
        self.ligament = curve(pos=[vector(0, 0, 0), vector(0.1, 0, 0), vector(0.2, 0, 0)],
                              radius=visual.ligament_radius, color=vector(*visual.ligament_colour))
        self.bristle = cylinder(pos=vector(0, 0, 0), axis=vector(0, 0.12, 0),
                                radius=visual.bristle_radius, color=vector(0.75, 0.85, 0.75))
        self.last_foot = vector(0, 0, 0)
        self.contact = False
        self.contact_force = 0.0
        self.joint_angle = 0.0
        self.muscle_force = 0.0

    def update(self, root, yaw: float, phase_time: float, activation: float,
               stride: Optional[float] = None, lift: Optional[float] = None, points=None):
        points = points if points is not None else local_leg_points(
            self.index, self.side, phase_time, stride, lift, self.geometry_template
        )

        def world(point):
            return root + rotate(vector(*point), angle=yaw, axis=vector(0, 1, 0))

        a, k, an, f = (world(points[key]) for key in ("attach", "knee", "ankle", "foot"))
        self.hip.pos, self.knee.pos, self.ankle.pos, self.foot.pos = a, k, an, f
        self.femur.pos, self.femur.axis = a, k - a
        self.tibia.pos, self.tibia.axis = k, an - k
        self.tarsus.pos, self.tarsus.axis = an, f - an

        level = max(0.0, min(1.0, float(activation)))
        self.flexor.pos = a + (k - a) * 0.22 + vector(0, 0.045, 0)
        self.flexor.axis = (k - a) * 0.48
        self.flexor.radius = 0.012 + 0.026 * level
        self.extensor.pos = a + (k - a) * 0.25 - vector(0, 0.045, 0)
        self.extensor.axis = (k - a) * 0.45
        self.extensor.radius = 0.012 + 0.022 * (1.0 - level)
        self.tendon.pos, self.tendon.axis = an, f - an
        self.ligament.modify(0, pos=k + vector(0, 0.035, 0))
        self.ligament.modify(1, pos=k + vector(0.07, 0.0, self.side * 0.035))
        self.ligament.modify(2, pos=k + vector(0.03, -0.045, self.side * 0.025))
        self.bristle.pos = k + vector(0.0, 0.02, self.side * 0.025)
        self.bristle.axis = vector(0.02, 0.10, self.side * 0.04)

        self.contact = f.y <= 0.055
        penetration = max(0.0, 0.055 - f.y)
        self.contact_force = penetration * 18.0 + (0.12 if self.contact else 0.0)
        self.joint_angle = math.atan2((k - a).y, max(1e-6, mag(k - a)))
        self.muscle_force = level * 0.75
        self.last_foot = f
        return f
