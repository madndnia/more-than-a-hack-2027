"""Synthetic P3 checks: python -m unittest discover -s tests -v."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mtah import ball, contract


def frames(x, y=None, half=None):
    x = np.asarray(x, dtype=float)
    return dict(ball_x=x, ball_y=np.zeros_like(x) if y is None else np.asarray(y, dtype=float),
                half=np.ones(len(x), dtype=int) if half is None else np.asarray(half),
                ball_observed=np.isfinite(x))


class ReconstructionTests(unittest.TestCase):
    def test_short_gap_recovers_linear_motion_in_both_axes(self):
        f = frames([0, np.nan, np.nan, 0.6], [1, np.nan, np.nan, 1.3])
        r = ball.reconstruct(f)
        np.testing.assert_allclose(r['x'], [0, .2, .4, .6])
        np.testing.assert_allclose(r['y'], [1, 1.1, 1.2, 1.3])
        self.assertTrue(np.all((r['conf'][1:3] > 0) & (r['conf'][1:3] < 1)))
        np.testing.assert_array_equal(r['observed'], f['ball_observed'])

    def test_long_gap_has_lower_confidence(self):
        short = ball.reconstruct(frames([0, np.nan, .2]))
        long = ball.reconstruct(frames([0] + [np.nan] * 100 + [20]))
        self.assertTrue(np.isfinite(long['x']).all())
        self.assertLess(long['conf'][50], .1)
        self.assertLess(long['conf'][50], short['conf'][1])

    def test_no_bridge_across_half_or_ungraded_frames(self):
        r = ball.reconstruct(frames([1, np.nan, np.nan, 90], half=[1, 0, 0, 2]))
        self.assertTrue(np.isnan(r['x'][1:3]).all())
        np.testing.assert_array_equal(r['conf'][1:3], 0)

    def test_edges_hold_only_for_configured_duration(self):
        r = ball.reconstruct(frames([np.nan] * 20 + [5] + [np.nan] * 20))
        self.assertTrue(np.isnan(r['x'][:8]).all())
        self.assertTrue(np.isnan(r['x'][33:]).all())
        np.testing.assert_array_equal(r['x'][8:33], 5)
        self.assertLess(r['conf'][8], r['conf'][19])

    def test_isolated_spike_is_replaced_but_raw_flag_preserved(self):
        r = ball.reconstruct(frames([0, .2, 20, .6, .8]))
        self.assertTrue(r['rejected'][2])
        self.assertTrue(r['observed'][2])
        self.assertAlmostEqual(r['x'][2], .4)
        self.assertLess(r['conf'][2], 1)

    def test_fast_pass_and_out_of_pitch_positions_survive(self):
        f = frames([-2, 1, 4, 7, 10], [69] * 5)
        r = ball.reconstruct(f)
        self.assertFalse(r['rejected'].any())
        np.testing.assert_array_equal(r['x'], f['ball_x'])
        np.testing.assert_array_equal(r['y'], f['ball_y'])

    def test_realistic_direction_change_survives(self):
        r = ball.reconstruct(frames([0, 1, 0, -1]))
        self.assertFalse(r['rejected'].any())
        np.testing.assert_array_equal(r['x'], [0, 1, 0, -1])

    def test_invalid_y_is_not_an_anchor(self):
        r = ball.reconstruct(frames([0, 5, .4], [0, np.inf, 0]))
        self.assertTrue(r['rejected'][1])
        self.assertAlmostEqual(r['x'][1], .2)
        self.assertTrue(np.isfinite(r['y']).all())

    def test_empty_or_unobserved_input(self):
        for f in (frames([]), frames([np.nan] * 10)):
            r = ball.reconstruct(f)
            self.assertTrue(np.isnan(r['x']).all())
            self.assertTrue((r['conf'] == 0).all())

    def test_input_not_mutated_and_output_is_deterministic(self):
        f = frames([0, np.nan, .4])
        before = {k: v.copy() for k, v in f.items()}
        first, second = ball.reconstruct(f), ball.reconstruct(f)
        for k in f:
            np.testing.assert_array_equal(f[k], before[k])
        for k in first:
            np.testing.assert_array_equal(first[k], second[k])

    def test_invalid_parameters_and_shapes(self):
        for kwargs in (dict(fps=0), dict(short_gap_seconds=-1),
                       dict(edge_hold_seconds=-1), dict(spike_speed=np.inf)):
            with self.assertRaises(ValueError):
                ball.reconstruct(frames([1]), **kwargs)
        f = frames([1, 2])
        f['ball_y'] = np.zeros(3)
        with self.assertRaises(ValueError):
            ball.reconstruct(f)

    def test_full_size_build_and_contract_roundtrip(self):
        f = frames(np.full(contract.N_FRAMES, np.nan))
        f['half'][:100] = 0
        f['ball_x'][100:200] = np.arange(100) * .2
        f['ball_observed'] = np.isfinite(f['ball_x'])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(contract, 'DERIVED_DIR', directory), patch.object(ball, 'load_frames', return_value=f):
                ball.build()
                r = contract.load('ball_clean')
        self.assertEqual(len(r['x']), contract.N_FRAMES)
        self.assertTrue(np.all(np.isfinite(r['x'][r['conf'] > 0])))
        self.assertTrue(np.all(np.isfinite(r['y'][r['conf'] > 0])))
        np.testing.assert_array_equal(r['observed'], f['ball_observed'])


if __name__ == '__main__':
    unittest.main()
