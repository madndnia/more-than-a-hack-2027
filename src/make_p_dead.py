"""Dead-ball probability per video frame  ->  data/derived/p_dead.csv   (Person 2's deliverable).

    python src/make_p_dead.py                 # writes data/derived/p_dead.csv
    python src/make_p_dead.py --out other.csv

Output: header `frame,p_dead`, one row per video frame 0..n-1, P(ball is dead) in [0.001, 0.999],
no NaN. Soft evidence for the HMM decoder, not a hard label.

How it works
------------
Everything is built in log-odds: start from a football prior, then add or subtract evidence.

  prior          dead ball is a minority of graded time (P_BASE).
  out of bounds  ball seen outside the pitch lines -> strongly dead. Carried forward while the ball
                 is missing, because a ball that left the pitch is usually not tracked again until
                 the restart.
  stationary     ball stays inside a small area for ~2 s -> dead. Uses the extent of the ball's
                 movement over a window, so tracker jitter does not matter.
  moving fast    ball covering ground quickly -> alive.
  set pieces     Smart Tagging SET PIECES: a soft bonus inside the tagged window and in the seconds
                 before it. Evidence, not a label.
  team speed     mean speed of all tracked players is low (walking) -> slightly more dead.
  ball missing   NO dead-ball evidence. Occlusion happens in the middle of play all the time, so a
                 missing ball stays at the prior (or is carried by the out-of-bounds flag).

Then: frames 110808..113741 are forced to 0.999 (graded and all D, per FAQ.md), ungraded frames
get 0.5, and everything is clipped to [0.001, 0.999].

IMPORTANT: every number in the PARAMS block is a football-sense default, not a fitted value, and
none may be tuned on the public 5-minute ground truth (RULES.md). Adjust them by looking at the
plausibility diagnostics this script prints, never by scoring against the public window.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402

# ------------------------------------------------------------------------------------------------
# PARAMS -- football-sense defaults. See the note above: do not tune on the public ground truth.
# ------------------------------------------------------------------------------------------------
FPS = 25

P_BASE = 0.30              # prior P(dead) on a graded frame
CLIP = (0.001, 0.999)

# Graded blocks (video frames, inclusive) and the all-D stretch at the start of the second half.
FALLBACK_BLOCKS = [(1, 11185, 90651), (2, 110808, 190197)]
FORCED_DEAD = (110808, 113741)
FORCED_VALUE = 0.999
UNGRADED_VALUE = 0.5

# Ball gaps up to this long (frames) are bridged by linear interpolation; longer ones stay missing.
MAX_GAP = 25

# Out of bounds: metres beyond the touchline / goal line before we call it out (tracker noise).
OOB_MARGIN_M = 0.5
OOB_CARRY = 20 * FPS       # keep the out-of-bounds state through missing frames, at most 20 s
W_OOB = 4.0

# Stationary ball: it stays within STILL_EXTENT_M for STILL_WIN frames (~2 s).
STILL_WIN = 51
STILL_MIN_PERIODS = 40
STILL_EXTENT_M = 1.5
W_STILL = 2.0

# Ball speed from displacement over +/- SPEED_HALF frames (~0.5 s in total).
SPEED_HALF = 6
MOVE_MS, W_MOVE = 2.0, 1.0   # faster than this: alive, small push
FAST_MS, W_FAST = 4.0, 2.0   # faster than this: alive, strong push

# Smart Tagging set pieces: bonus inside the tag, ramp over the PRE frames before it.
TAG_CODE = "set pieces"
W_TAG_IN = 1.0
TAG_PRE = 6 * FPS
W_TAG_PRE = 1.0

# Team-wide speed (mean over tracked players, m/s), smoothed over ~2 s. Walking -> dead.
SPD_WIN, SPD_MIN_PERIODS = 51, 10
SPD_CENTER, SPD_SCALE, W_SPD = 1.5, 0.5, 0.8

# Diagnostics: a "dead stretch" is a run of p > 0.5 at least this long.
DIAG_MIN_STRETCH = 3 * FPS

FIELD_SEP, GROUP_SEP, MISSING = ":", ";", "N"


def logit(p):
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def read_atd(raw_path, n_frames, pitch):
    """Streams the raw ATD once. Returns ball x/y in metres (NaN where missing) and the mean speed
    of the tracked players per frame (NaN where none)."""
    length, width = pitch
    bx = np.full(n_frames, np.nan)
    by = np.full(n_frames, np.nan)
    mean_speed = np.full(n_frames, np.nan)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1          # frame_id counts from 1, submissions from 0
            if not 0 <= frame < n_frames:
                continue

            ball = parts[2]
            if ball and ball[0] != MISSING:
                fields = ball.split(",")
                try:
                    bx[frame] = float(fields[0]) * length
                    by[frame] = float(fields[1]) * width
                except (ValueError, IndexError):
                    pass

            speeds = []
            for group in parts[1].split(GROUP_SEP):
                if not group or group[0] == MISSING:
                    continue
                fields = group.split(",")
                if len(fields) < 3:
                    continue
                try:
                    s = float(fields[2])
                except ValueError:
                    continue
                if s == s:                      # not NaN
                    speeds.append(s)
            if speeds:
                mean_speed[frame] = sum(speeds) / len(speeds)
    return bx, by, mean_speed


def bridge_gaps(values, max_gap):
    """Linear interpolation across missing runs of at most max_gap frames; longer runs stay NaN."""
    s = pd.Series(values)
    missing = s.isna().values
    group = (~missing).cumsum()
    gap_len = pd.Series(missing).groupby(group).transform("sum").values
    filled = s.interpolate(limit_area="inside").to_numpy(copy=True)
    filled[missing & (gap_len > max_gap)] = np.nan
    return filled


def ball_features(bx, by, pitch):
    """Boolean/float arrays over frames: observed, out of bounds (carried), still, speed."""
    length, width = pitch
    n = len(bx)
    observed = ~np.isnan(bx)

    # out of bounds, carried through missing frames until the ball is seen in bounds again
    oob_now = observed & ((bx < -OOB_MARGIN_M) | (bx > length + OOB_MARGIN_M) |
                          (by < -OOB_MARGIN_M) | (by > width + OOB_MARGIN_M))
    state = pd.Series(np.where(observed, oob_now.astype(float), np.nan))
    oob = (state.ffill(limit=OOB_CARRY).fillna(0.0).values == 1.0)

    fx, fy = bridge_gaps(bx, MAX_GAP), bridge_gaps(by, MAX_GAP)
    have = ~np.isnan(fx)

    def extent(v):
        r = pd.Series(v).rolling(STILL_WIN, center=True, min_periods=STILL_MIN_PERIODS)
        return (r.max() - r.min()).values

    box = np.hypot(extent(fx), extent(fy))
    still = have & ~np.isnan(box) & (box < STILL_EXTENT_M)

    ahead_x, back_x = pd.Series(fx).shift(-SPEED_HALF).values, pd.Series(fx).shift(SPEED_HALF).values
    ahead_y, back_y = pd.Series(fy).shift(-SPEED_HALF).values, pd.Series(fy).shift(SPEED_HALF).values
    speed = np.hypot(ahead_x - back_x, ahead_y - back_y) / (2.0 * SPEED_HALF / FPS)
    return observed, oob, still, speed


def tag_bonus(n, events):
    """Per-frame log-odds bonus from SET PIECES tags: full inside the tag, a ramp before it."""
    bonus = np.zeros(n)
    for event in events:
        if (event["code"] or "").strip().lower() != TAG_CODE:
            continue
        start = max(0, event["start_frame"])
        end = min(n, event["end_frame"])
        if end <= start:
            continue
        bonus[start:end] = np.maximum(bonus[start:end], W_TAG_IN)
        pre_start = max(0, start - TAG_PRE)
        if start > pre_start:
            ramp = np.linspace(0.0, W_TAG_PRE, start - pre_start, endpoint=False)
            bonus[pre_start:start] = np.maximum(bonus[pre_start:start], ramp)
    return bonus


def build_logit(bx, by, mean_speed, events, pitch):
    n = len(bx)
    observed, oob, still, speed = ball_features(bx, by, pitch)
    z = np.full(n, logit(P_BASE))

    z += np.where(oob, W_OOB, 0.0)
    z += np.where(still & ~oob, W_STILL, 0.0)

    moving = ~oob & ~np.isnan(speed)
    z -= np.where(moving & (speed > FAST_MS), W_FAST,
                  np.where(moving & (speed > MOVE_MS), W_MOVE, 0.0))

    z += tag_bonus(n, events)

    team = pd.Series(mean_speed).rolling(SPD_WIN, center=True, min_periods=SPD_MIN_PERIODS).mean().values
    z += np.nan_to_num(np.clip(-(team - SPD_CENTER) / SPD_SCALE, -1.0, 1.0) * W_SPD, nan=0.0)
    return z, observed, oob, still


def graded_blocks():
    try:
        from games import half_video_blocks
        return list(half_video_blocks(GAME))
    except Exception:
        return FALLBACK_BLOCKS


def apply_rules(p, blocks):
    n = len(p)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    p = np.where(graded, p, UNGRADED_VALUE)
    p[FORCED_DEAD[0]:FORCED_DEAD[1] + 1] = FORCED_VALUE
    return np.clip(p, *CLIP), graded


def stretches(mask, min_len):
    """Lengths (frames) of runs of True that last at least min_len."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    lengths = edges[1::2] - edges[0::2]
    return lengths[lengths >= min_len]


def diagnostics(p, blocks, observed, oob, still):
    print("\nplausibility diagnostics (no ground truth involved)")
    for half, first, last in blocks:
        seg = slice(first, last + 1)
        dead = p[seg] > 0.5
        runs = stretches(dead, DIAG_MIN_STRETCH)
        med = np.median(runs) / FPS if len(runs) else float("nan")
        print("  half {}: ball observed {:.0f}% | p>0.5 on {:.0f}% of frames | "
              "{} dead stretches >= {} s, median {:.1f} s".format(
                  half, 100 * observed[seg].mean(), 100 * dead.mean(),
                  len(runs), DIAG_MIN_STRETCH // FPS, med))
        print("          out-of-bounds frames {:.1f}% | stationary frames {:.1f}%".format(
            100 * oob[seg].mean(), 100 * still[seg].mean()))
    print("  What to look for: dozens of dead stretches per half (throw-ins, goal kicks, free kicks),")
    print("  not hundreds of tiny ones and not a handful of huge ones. Dead share well above half of")
    print("  the graded time, or almost none, means a threshold in PARAMS is off.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "derived", "p_dead.csv"))
    args = parser.parse_args()

    pitch = GAMES[GAME]["pitch"]
    n = source_frames("tactical", GAME)
    raw_path = require(["atd_raw"], GAME, None)[0]

    print("reading {} ...".format(os.path.basename(raw_path)))
    bx, by, mean_speed = read_atd(raw_path, n, pitch)
    events = load_tags()
    print("  {} frames, ball seen on {:.1f}%, {} tag instances".format(
        n, 100 * np.mean(~np.isnan(bx)), len(events)))

    z, observed, oob, still = build_logit(bx, by, mean_speed, events, pitch)
    blocks = graded_blocks()
    p, _ = apply_rules(sigmoid(z), blocks)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame({"frame": np.arange(n), "p_dead": p}).to_csv(args.out, index=False, float_format="%.6f")
    assert len(p) == n and not np.isnan(p).any() and p.min() >= CLIP[0] and p.max() <= CLIP[1]
    print("wrote {}  rows={}  min={:.3f} max={:.3f}".format(args.out, n, p.min(), p.max()))

    diagnostics(p, blocks, observed, oob, still)


if __name__ == "__main__":
    main()