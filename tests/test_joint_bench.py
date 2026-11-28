import unittest

from experiments.blocked_joint import RELEASE, simulate


class JointBenchTests(unittest.TestCase):
    def test_matched_load_and_release(self):
        for angle, effort in ((0., .8), (-.08, .65), (.05, .9)):
            trace = simulate(angle, effort)
            rows = trace['trials']
            free, soft, held = (rows[n][RELEASE] for n in ('free','compliant','rigid-held'))
            with self.subTest(angle=angle, effort=effort):
                self.assertLess(held['angle'], soft['angle'])
                self.assertLess(soft['angle'], free['angle'])
                self.assertLess(free['forward'], soft['forward'])
                self.assertLess(soft['forward'], held['forward'])
                self.assertEqual(rows['rigid-held'][:RELEASE+1], rows['rigid-release'][:RELEASE+1])
                self.assertGreater(rows['rigid-release'][RELEASE+20]['velocity'], .1)
                self.assertEqual(rows['rigid-held'][RELEASE+20]['velocity'], 0.)
                self.assertLessEqual(max(r['angle'] for r in rows['rigid-held']), trace['stop_rad']+1e-12)

    def test_step_refinement_and_replay(self):
        coarse = simulate()
        self.assertEqual(coarse, simulate())
        fine = simulate(step=.0005)
        for name, rows in coarse['trials'].items():
            error = max(abs(a['angle']-b['angle']) for a,b in zip(rows, fine['trials'][name]))
            self.assertLess(error, .012, name)
