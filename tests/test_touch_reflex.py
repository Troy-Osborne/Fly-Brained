from importlib.util import find_spec
import unittest

from experiments.touch_reflex import trial
from nervous_system.adapters.mcf import MCFTouchController, TouchController


class TouchReflexTests(unittest.TestCase):
    def test_contact_causes_turn_away(self):
        for side, sign in (("left", -1), ("right", 1)):
            with self.subTest(side=side):
                result = trial(side, TouchController)
                self.assertEqual(result["first_contact_tick"], 1)
                self.assertEqual(result["first_command_tick"], 1)
                self.assertGreater(sign * result["final_yaw_rad"], 0)

    def test_sham_stays_still(self):
        result = trial("sham", TouchController)
        self.assertIsNone(result["first_contact_tick"])
        self.assertIsNone(result["first_command_tick"])
        self.assertEqual(result["final_yaw_rad"], 0)

    @unittest.skipUnless(find_spec("mcf_parts"), "optional external MCF checkout")
    def test_mcf_reproduces_reference(self):
        for side in ("left", "right", "sham"):
            with self.subTest(side=side):
                self.assertEqual(trial(side, MCFTouchController), trial(side, TouchController))
