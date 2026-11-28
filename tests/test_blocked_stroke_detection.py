import unittest
from experiments.blocked_stroke_detection import DWELL, PROTOCOLS, evaluate


class DetectionTests(unittest.TestCase):
    def test_free_negative_and_ablation(self):
        for p in PROTOCOLS:
            full = evaluate(p, 'free', 'command')
            self.assertIsNone(full['first_alarm_tick'])
            self.assertEqual(full['velocity_rmse'], 0.)
            self.assertGreater(evaluate(p, 'free', 'no_command')['velocity_rmse'], 0.)

    def test_rigid_contact_is_detected_causally(self):
        for p in PROTOCOLS:
            for condition in ('rigid', 'release'):
                row = evaluate(p, condition, 'command')
                self.assertIsNotNone(row['first_loaded_alarm_tick'])
                self.assertGreaterEqual(row['first_alarm_tick'] - row['first_contact_tick'], DWELL-1)
                self.assertEqual(row['false_alarm_ticks'], 0)
