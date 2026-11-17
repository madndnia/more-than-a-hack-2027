"""Independent P3 check against held-out tracking observations, not possession GT.

    python scripts/check_ball_quality.py

Prints aggregate metrics only. Does not overwrite ball_clean or submission files.
Each duration is a separate deterministic experiment with nonoverlapping blocks
whose interiors and endpoints were all observed. Tracker observations are noisy
references, not manually corrected ball ground truth. Natural occlusions may be
harder than these artificially hidden visible stretches. Do not tune possession
parameters from this check or interpret it as an official challenge score.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mtah.ball import reconstruct  # noqa: E402
from mtah.contract import FPS  # noqa: E402
from mtah.loader import load_frames  # noqa: E402


def sample_blocks(frames, gap_frames, limit=200, seed=0):
    """Select disjoint fully observed blocks, reserving both visible endpoints."""
    if gap_frames < 1 or limit < 1:
        raise ValueError("gap_frames and limit must be positive")
    half = np.asarray(frames['half'])
    valid = (np.asarray(frames['ball_observed'], dtype=bool)
             & np.isfinite(frames['ball_x']) & np.isfinite(frames['ball_y'])
             & np.isin(half, (1, 2)))
    # Runs split at either a missing coordinate or a change of half.
    breaks = np.r_[0, np.flatnonzero((valid[1:] != valid[:-1])
                                    | (half[1:] != half[:-1])) + 1, len(valid)]
    candidates = []
    for start, stop in zip(breaks[:-1], breaks[1:]):
        if start < stop and valid[start] and stop - start >= gap_frames + 2:
            candidates.append(np.arange(start + 1, stop - gap_frames))
    if not candidates:
        return np.empty(0, dtype=int)
    candidates = np.concatenate(candidates)
    rng = np.random.RandomState(seed)
    reserved = np.zeros(len(valid), dtype=bool)
    selected = []
    for start in rng.permutation(candidates):
        right = start + gap_frames
        if not reserved[start - 1:right + 1].any():
            selected.append(start)
            reserved[start - 1:right + 1] = True
            if len(selected) == limit:
                break
    return np.asarray(sorted(selected), dtype=int)


def scaffold_positions(frames):
    """The original P3 interpolation, including its unlimited edge holds."""
    result = {}
    for coordinate in ('x', 'y'):
        values = np.asarray(frames['ball_' + coordinate], dtype=float)
        out = values.copy()
        for half in (1, 2):
            idx = np.flatnonzero(frames['half'] == half)
            seen = idx[~np.isnan(values[idx])]
            if len(seen) >= 2:
                out[idx] = np.interp(idx, seen, values[seen])
        result[coordinate] = out
    return result


def position_metrics(prediction, frames, indices):
    distance = np.hypot(prediction['x'][indices] - frames['ball_x'][indices],
                        prediction['y'][indices] - frames['ball_y'][indices])
    finite = np.isfinite(distance)
    error = distance[finite]
    return dict(coverage=float(finite.mean()),
                mean_m=float(error.mean()) if len(error) else None,
                median_m=float(np.median(error)) if len(error) else None,
                rmse_m=float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
                p95_m=float(np.percentile(error, 95)) if len(error) else None)


def evaluate(frames, gap_frames, limit=200, seed=0):
    starts = sample_blocks(frames, gap_frames, limit, seed)
    if not len(starts):
        return dict(gap_frames=gap_frames, blocks=0)
    indices = (starts[:, None] + np.arange(gap_frames)).ravel()
    # Hide both coordinates AND the observed flag: no hidden observations can
    # enter the reconstruction or the spike filter through an alternative field.
    masked = {key: np.array(frames[key], copy=True)
              for key in ('ball_x', 'ball_y', 'ball_observed', 'half')}
    masked['ball_x'][indices] = np.nan
    masked['ball_y'][indices] = np.nan
    masked['ball_observed'][indices] = False
    ours = reconstruct(masked)
    original = scaffold_positions(masked)
    return dict(gap_frames=gap_frames, gap_seconds=gap_frames / float(FPS),
                blocks=len(starts), hidden_frames=len(indices),
                p3=position_metrics(ours, frames, indices),
                scaffold=position_metrics(original, frames, indices),
                mean_conf=float(ours['conf'][indices].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json', action='store_true', help='print aggregate report as JSON')
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error('--blocks must be positive')
    frames = load_frames()
    reports = [evaluate(frames, gap, args.blocks, args.seed)
               for gap in (5, 12, 25, 50, 125, 250)]
    if args.json:
        print(json.dumps(reports, indent=2, allow_nan=False))
        return
    print('Held-out tracker positions; NOT corrected ground truth or a possession score.')
    print('Independent experiments by gap duration; fixed seed {}; max {} blocks.'.format(args.seed, args.blocks))
    print(' gap_s blocks coverage  P3_mean_m P3_median_m P3_p95_m original_mean_m confidence')
    for report in reports:
        if not report['blocks']:
            print('{:6.2f}      0 (no eligible visible stretches)'.format(report['gap_frames'] / FPS))
            continue
        p3 = report['p3']
        print('{:6.2f} {:6d} {:8.1%} {:10.3f} {:11.3f} {:8.3f} {:15.3f} {:10.3f}'.format(
            report['gap_seconds'], report['blocks'], p3['coverage'], p3['mean_m'],
            p3['median_m'], p3['p95_m'], report['scaffold']['mean_m'], report['mean_conf']))


if __name__ == '__main__':
    main()
