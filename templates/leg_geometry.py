"""Pure, testable local-coordinate geometry for the six insect legs.

Coordinate convention:
    +X = forward (toward head)
    +Y = up
    -Z = fly's anatomical left
    +Z = fly's anatomical right

The camera may make the near-side leg appear on the opposite side of the
screen.  The anatomical names refer to the fly's own coordinate frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

Vec3 = Tuple[float, float, float]

SIDE_LEFT = -1
SIDE_RIGHT = 1


@dataclass(frozen=True)
class LegGeometryTemplate:
    """Editable geometry and gait parameters for one family of legs."""
    front_attach_x: float = 0.34
    middle_attach_x: float = 0.0
    hind_attach_x: float = -0.34
    front_attach_z: float = 0.23
    middle_attach_z: float = 0.25
    hind_attach_z: float = 0.23
    attach_drop: float = 0.15
    foot_base_z: float = 0.78
    foot_spacing_z: float = 0.04
    foot_y: float = -1.18
    knee_mix_attach: float = 0.48
    knee_mix_foot: float = 0.52
    knee_x_front_hind: float = 0.20
    knee_x_middle: float = -0.16
    knee_y_offset: float = 0.20
    knee_z_offset: float = 0.04
    ankle_mix_knee: float = 0.48
    ankle_mix_foot: float = 0.52
    ankle_x_offset: float = -0.04
    ankle_y_offset: float = 0.02
    ankle_z_offset: float = 0.015
    gait_angular_rate: float = 8.0
    stride: float = 0.26
    lift: float = 0.17


DEFAULT_LEG_TEMPLATE = LegGeometryTemplate()


def _lerp3(a: Vec3, b: Vec3, amount_a: float, amount_b: float) -> Vec3:
    return (
        a[0] * amount_a + b[0] * amount_b,
        a[1] * amount_a + b[1] * amount_b,
        a[2] * amount_a + b[2] * amount_b,
    )


def local_leg_points(
    index: int,
    side: int,
    phase_time: float,
    stride: float | None = None,
    lift: float | None = None,
    template: LegGeometryTemplate = DEFAULT_LEG_TEMPLATE,
) -> Dict[str, Vec3]:
    """Return attach, knee, ankle and foot coordinates in body-local space.

    Left and right points are exact mirror images across the sagittal (X-Y)
    plane when their gait phase is the same.  The hip anchor and distal foot
    are deliberately assigned the *same* lateral sign; this prevents the
    limbs from crossing under the body.
    """
    if index not in (0, 1, 2):
        raise ValueError("leg index must be 0 (front), 1 (middle), or 2 (hind)")
    if side not in (SIDE_LEFT, SIDE_RIGHT):
        raise ValueError("side must be SIDE_LEFT (-1) or SIDE_RIGHT (+1)")

    attach_x = (template.front_attach_x, template.middle_attach_x, template.hind_attach_x)[index]
    attach_z_mag = (template.front_attach_z, template.middle_attach_z, template.hind_attach_z)[index]

    # CRITICAL: lateral offsets are positive magnitudes multiplied by side.
    # The previous implementation started the hip on -side and ended the
    # foot on +side, making every leg cross the midline.
    attach = (attach_x, -template.attach_drop, side * attach_z_mag)
    stride_value = template.stride if stride is None else float(stride)
    lift_value = template.lift if lift is None else float(lift)

    side_index = 0 if side == SIDE_LEFT else 1
    tripod_phase = 0.0 if (index + side_index) % 2 == 0 else math.pi
    phase = phase_time * template.gait_angular_rate + tripod_phase

    foot = (
        attach_x + stride_value * math.cos(phase),
        template.foot_y + lift_value * max(0.0, math.sin(phase)),
        side * (template.foot_base_z + template.foot_spacing_z * index),
    )
    knee_x_bend = template.knee_x_middle if index == 1 else template.knee_x_front_hind
    knee_base = _lerp3(attach, foot, template.knee_mix_attach, template.knee_mix_foot)
    knee = (
        knee_base[0] + knee_x_bend,
        knee_base[1] + template.knee_y_offset,
        knee_base[2] + side * template.knee_z_offset,
    )
    ankle_base = _lerp3(knee, foot, template.ankle_mix_knee, template.ankle_mix_foot)
    ankle = (
        ankle_base[0] + template.ankle_x_offset,
        ankle_base[1] + template.ankle_y_offset,
        ankle_base[2] + side * template.ankle_z_offset,
    )
    return {"attach": attach, "knee": knee, "ankle": ankle, "foot": foot}
