import unittest
import numpy as np

from experiments.touch_reflex import trial
from nervous_system.adapters.mcf import TouchController
from research.muscle_tendon import TendonPair


class MuscleTendonTests(unittest.TestCase):
    def test_load_release_and_reverse(self):
        unit = TendonPair([0.])
        def advance(rate):
            return unit.step([0.], np.array([rate]), np.ones(1), .02)[0]
        forward, reverse = advance(1.)
        self.assertTrue(0 < forward < unit.peak_force)
        self.assertEqual(reverse, 0)
        residual, _ = advance(0.)
        self.assertTrue(0 < residual < forward)
        for _ in range(30):
            forward, reverse = advance(-1.)
        self.assertGreater(reverse, forward)

    def test_passive_motion_and_timestep(self):
        def run(count):
            pair = TendonPair([0.])
            for i in range(count):
                force = pair.step([(i+1)/count], np.zeros(1), np.zeros(1), .04/count)
            return force[0]
        coarse = run(1)
        self.assertGreater(coarse[0], 0)
        self.assertEqual(coarse[1], 0)
        np.testing.assert_allclose(coarse, run(8))

    def test_contact_direction_and_sham(self):
        for side, sign in (("left", -1), ("right", 1)):
            result = trial(side, TouchController, "series_elastic")
            self.assertGreater(sign * result["final_yaw_rad"], 0)
            self.assertGreater(result["peak_leg_tension"], 0)
        sham = trial("sham", TouchController, "series_elastic")
        self.assertEqual(sham["final_yaw_rad"], 0)
        self.assertEqual(sham["peak_leg_tension"], 0)
