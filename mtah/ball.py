"""P3: gap-aware ball reconstruction, with conservative isolated-spike rejection.

Long gaps retain low-confidence linear fallbacks for existing consumers. These
are not evidence of the true trajectory. No possession labels are used.
"""
import numpy as np

from mtah import contract
from mtah.loader import load_frames


def reconstruct(frames, fps=contract.FPS, short_gap_seconds=1.0,
                edge_hold_seconds=0.5, spike_speed=55.0):
    """Return contract fields plus a boolean ``rejected`` audit mask.

    ``observed`` preserves the loader flag, including rejected detections.
    Confidence is a heuristic reliability weight, not a calibrated probability.
    Positions are not clipped to the pitch: out-of-play observations are valid.
    """
    if (not np.isfinite(fps) or fps <= 0
            or not np.isfinite(short_gap_seconds) or short_gap_seconds <= 0
            or not np.isfinite(edge_hold_seconds) or edge_hold_seconds < 0
            or not np.isfinite(spike_speed) or spike_speed <= 0):
        raise ValueError("fps, gap duration and spike speed must be positive; edge hold nonnegative")
    raw_x = np.asarray(frames["ball_x"], dtype=float)
    raw_y = np.asarray(frames["ball_y"], dtype=float)
    half = np.asarray(frames["half"])
    observed = np.asarray(frames["ball_observed"], dtype=bool).copy()
    if raw_x.ndim != 1 or any(a.shape != raw_x.shape for a in (raw_y, half, observed)):
        raise ValueError("ball coordinates, half and observed must be matching 1-D arrays")
    x = np.full(raw_x.shape, np.nan)
    y = np.full(raw_x.shape, np.nan)
    conf = np.zeros(raw_x.shape)
    rejected = np.zeros(raw_x.shape, dtype=bool)
    valid = observed & np.isfinite(raw_x) & np.isfinite(raw_y)
    rejected[observed & ~valid] = True

    # Split contiguous runs: do not bridge halves or ungraded intervals.
    boundaries = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1, len(half)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if start == stop or half[start] not in (1, 2):
            continue
        seen = np.flatnonzero(valid[start:stop]) + start
        if len(seen) >= 3:
            a, b, c = seen[:-2], seen[1:-1], seen[2:]
            incoming = np.hypot(raw_x[b] - raw_x[a], raw_y[b] - raw_y[a]) * fps
            outgoing = np.hypot(raw_x[c] - raw_x[b], raw_y[c] - raw_y[b]) * fps
            bridge = np.hypot(raw_x[c] - raw_x[a], raw_y[c] - raw_y[a]) * fps / 2
            deviation = np.hypot(raw_x[b] - (raw_x[a] + raw_x[c]) / 2,
                                 raw_y[b] - (raw_y[a] + raw_y[c]) / 2)
            spikes = ((b - a == 1) & (c - b == 1)
                      & (incoming > spike_speed) & (outgoing > spike_speed)
                      & (bridge < 20.0) & (deviation > 3.0))
            rejected[b[spikes]] = True
        seen = seen[~rejected[seen]]
        if not len(seen):
            continue
        x[seen], y[seen], conf[seen] = raw_x[seen], raw_y[seen], 1.0
        for left, right in zip(seen[:-1], seen[1:]):
            idx = np.arange(left + 1, right)
            if not len(idx):
                continue
            fraction = (idx - left) / float(right - left)
            x[idx] = raw_x[left] + fraction * (raw_x[right] - raw_x[left])
            y[idx] = raw_y[left] + fraction * (raw_y[right] - raw_y[left])
            duration = len(idx) / fps
            if duration <= short_gap_seconds:
                weight = 0.8 * np.exp(-duration / short_gap_seconds)
            else:
                weight = 0.1 * np.exp(-(duration - short_gap_seconds) / 5.0)
            speed = np.hypot(raw_x[right] - raw_x[left], raw_y[right] - raw_y[left]) * fps / (right - left)
            conf[idx] = weight * min(1.0, spike_speed / max(speed, 1e-12))
        # Brief constant-position holds; no indefinite edge extrapolation.
        hold = int(edge_hold_seconds * fps)
        edges = ((seen[0], np.arange(max(start, seen[0] - hold), seen[0])),
                 (seen[-1], np.arange(seen[-1] + 1, min(stop, seen[-1] + hold + 1))))
        for anchor, idx in edges:
            x[idx], y[idx] = raw_x[anchor], raw_y[anchor]
            conf[idx] = 0.2 * np.exp(-np.abs(idx - anchor) / max(hold, 1))
    return dict(x=x, y=y, observed=observed, conf=conf, rejected=rejected)


def build():
    result = reconstruct(load_frames())
    print("ball: {} observed, {} rejected, {} estimated, {} unavailable".format(
        int(result["observed"].sum()), int(result["rejected"].sum()),
        int(((result["conf"] > 0) & (~result["observed"] | result["rejected"])).sum()),
        int((result["conf"] == 0).sum())))
    return contract.save("ball_clean", **result)


if __name__ == "__main__":
    print(build())
