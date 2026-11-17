"""Check holdout methodology on synthetic trajectories, without match assets."""
import unittest

import numpy as np

from scripts.check_ball_quality import evaluate, sample_blocks


def linear_frames(n=1000):
    return dict(ball_x=np.arange(n, dtype=float) * .1,
                ball_y=np.arange(n, dtype=float) * .05,
                ball_observed=np.ones(n, dtype=bool), half=np.ones(n, dtype=int))


class QualityCheckTests(unittest.TestCase):
    def test_blocks_keep_visible_endpoints_and_do_not_overlap(self):
        frames = linear_frames()
        frames['ball_observed'][400:410] = False
        frames['half'][500:] = 2
        starts = sample_blocks(frames, 25, limit=20)
        reserved = set()
        self.assertTrue(len(starts))
        for start in starts:
            indices = set(range(start - 1, start + 26))
            self.assertFalse(reserved & indices)
            reserved.update(indices)
            self.assertTrue(frames['ball_observed'][start - 1:start + 26].all())
            self.assertEqual(len(set(frames['half'][start - 1:start + 26])), 1)
        np.testing.assert_array_equal(starts, sample_blocks(frames, 25, limit=20))

    def test_linear_motion_is_reconstructed_without_mutating_reference(self):
        frames = linear_frames()
        before = {key: value.copy() for key, value in frames.items()}
        result = evaluate(frames, 25, limit=10)
        self.assertEqual(result['blocks'], 10)
        self.assertEqual(result['hidden_frames'], 250)
        self.assertEqual(result['p3']['coverage'], 1)
        self.assertLess(result['p3']['mean_m'], 1e-10)
        self.assertLess(result['scaffold']['mean_m'], 1e-10)
        for key in frames:
            np.testing.assert_array_equal(frames[key], before[key])

    def test_curved_motion_has_measurable_holdout_error(self):
        frames = linear_frames()
        frames['ball_y'] = np.sin(np.arange(1000) / 10)
        result = evaluate(frames, 25, limit=10)
        self.assertGreater(result['p3']['mean_m'], .1)
        self.assertAlmostEqual(result['p3']['mean_m'], result['scaffold']['mean_m'])

    def test_no_eligible_blocks(self):
        frames = linear_frames(10)
        self.assertEqual(evaluate(frames, 25)['blocks'], 0)
        frames['half'][:] = 0
        self.assertEqual(evaluate(frames, 2)['blocks'], 0)


if __name__ == '__main__':
    unittest.main()
