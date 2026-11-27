"""Fast, VPython-free checks for the anatomical side/geometry templates."""
import math
import unittest

from templates.leg_geometry import (
    DEFAULT_LEG_TEMPLATE,
    SIDE_LEFT,
    SIDE_RIGHT,
    local_leg_points,
)
from templates.wing_motion import WingMotionModel


class LegLateralityTests(unittest.TestCase):
    def test_each_leg_stays_on_its_own_side(self):
        for index in range(3):
            for side in (SIDE_LEFT, SIDE_RIGHT):
                points = local_leg_points(index, side, phase_time=0.37)
                for joint_name, point in points.items():
                    with self.subTest(index=index, side=side, joint=joint_name):
                        self.assertGreater(side * point[2], 0.0)

    def test_attach_offsets_are_mirrored(self):
        for index in range(3):
            left = local_leg_points(index, SIDE_LEFT, phase_time=0.0)["attach"]
            right = local_leg_points(index, SIDE_RIGHT, phase_time=0.0)["attach"]
            self.assertAlmostEqual(left[0], right[0])
            self.assertAlmostEqual(left[1], right[1])
            self.assertAlmostEqual(left[2], -right[2])

    def test_left_and_right_leg_geometry_reflects_when_gait_phase_matches(self):
        # Tripod coordination offsets opposite sides by pi, so compare the
        # opposite-side samples half a cycle apart to isolate mirror geometry.
        half_cycle = math.pi / DEFAULT_LEG_TEMPLATE.gait_angular_rate
        for index in range(3):
            left = local_leg_points(index, SIDE_LEFT, phase_time=0.41)
            right = local_leg_points(index, SIDE_RIGHT, phase_time=0.41 + half_cycle)
            for name in ("attach", "knee", "ankle", "foot"):
                self.assertAlmostEqual(left[name][0], right[name][0], places=8)
                self.assertAlmostEqual(left[name][1], right[name][1], places=8)
                self.assertAlmostEqual(left[name][2], -right[name][2], places=8)


class WingMirrorTests(unittest.TestCase):
    def test_wing_frames_are_mirrored_and_do_not_cross(self):
        model = WingMotionModel()
        model.phase = 1.17
        left_axis, left_up = model.wing_frame(SIDE_LEFT, 0.72)
        right_axis, right_up = model.wing_frame(SIDE_RIGHT, 0.72)
        self.assertAlmostEqual(left_axis[0], right_axis[0])
        self.assertAlmostEqual(left_axis[1], right_axis[1])
        self.assertAlmostEqual(left_axis[2], -right_axis[2])
        self.assertAlmostEqual(left_up[0], right_up[0])
        self.assertAlmostEqual(left_up[1], right_up[1])
        self.assertAlmostEqual(left_up[2], -right_up[2])
        self.assertLess(left_axis[2], 0.0)
        self.assertGreater(right_axis[2], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
