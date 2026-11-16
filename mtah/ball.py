"""Person 3 -- ball reconstruction.  Output: ball_clean (x, y in metres, observed, conf).

DUMMY v0 (Person 1 scaffold): linear interpolation of the ball inside each half, conf 1 where the
tracker saw the ball and 0.3 where it was interpolated. Person 3 replaces build() and keeps the
output contract (mtah/contract.py, SPEC["ball_clean"]).
"""
import numpy as np

from mtah import contract
from mtah.loader import load_frames


def _interp(values, half):
    out = values.copy()
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        seen = idx[~np.isnan(values[idx])]
        if len(seen) >= 2:
            out[idx] = np.interp(idx, seen, values[seen])
    return out


def build():
    F = load_frames()
    observed = F["ball_observed"]
    x = _interp(F["ball_x"], F["half"])
    y = _interp(F["ball_y"], F["half"])
    conf = np.where(observed, 1.0, 0.3)
    conf[np.isnan(x)] = 0.0        # outside the halves nothing to interpolate from
    return contract.save("ball_clean", x=x, y=y, observed=observed, conf=conf)


if __name__ == "__main__":
    print(build())
