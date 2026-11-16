"""The team contract: what every shared file is called, what it contains, and how it is checked.

Everybody reads and writes shared artifacts ONLY through save() / load() below. That is what lets
five people work in parallel: nobody imports anybody else's code, only their output file.

Every artifact is a .npz in data/derived/ holding 1-D arrays of length N_FRAMES, indexed by
tactical video frame 0..199318 (= submission row). data/derived/ is gitignored.

    from mtah import contract
    contract.save("p_dead", p_dead=my_array)          # checked before it is written
    p = contract.load("p_dead")["p_dead"]             # checked again when it is read
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]  # 199319
FPS = GAMES[GAME]["fps"]                   # 25
PITCH = GAMES[GAME]["pitch"]               # (105.3, 68.0) metres

DERIVED_DIR = os.environ.get("MTAH_DERIVED_DIR") or os.path.join(ROOT, "data", "derived")
SUBMISSIONS_DIR = os.path.join(ROOT, "submissions")

# Field kinds:
#   "prob"  float in [0, 1], no NaN
#   "float" any finite float, NaN allowed only where noted by allow_nan
#   "bool"  boolean flag
SPEC = {
    # Person 3 -- ball reconstruction. x/y in METRES, filled on every graded frame.
    "ball_clean": {
        "owner": "P3",
        "fields": {"x": "float", "y": "float", "observed": "bool", "conf": "prob"},
        "allow_nan": ("x", "y"),   # only allowed where conf == 0
    },
    # Person 2 -- dead-ball detector. P(ball is dead) per frame.
    "p_dead": {
        "owner": "P2",
        "fields": {"p_dead": "prob"},
    },
    # Person 4 -- H vs A given the ball is alive. p_home + p_away == 1.
    "p_team": {
        "owner": "P4",
        "fields": {"p_home": "prob", "p_away": "prob", "conf": "prob"},
    },
    # Person 5 -- soft prior from Smart Tagging.
    #   prior_home / prior_away: team evidence from phase tags (0.5 / 0.5 = no evidence)
    #   restart: 1 inside a set-piece window (restart anchor), else 0
    "tag_prior": {
        "owner": "P5",
        "fields": {"prior_home": "prob", "prior_away": "prob", "restart": "prob"},
    },
}


def path(name):
    if name not in SPEC:
        raise KeyError("unknown artifact {!r}; the contract has {}".format(name, sorted(SPEC)))
    return os.path.join(DERIVED_DIR, name + ".npz")


def check(name, arrays):
    """Raise ValueError with a readable reason if `arrays` breaks the contract for `name`."""
    spec = SPEC[name]
    missing = sorted(set(spec["fields"]) - set(arrays))
    if missing:
        raise ValueError("{}: missing fields {}".format(name, missing))
    allow_nan = spec.get("allow_nan", ())
    for field, kind in spec["fields"].items():
        a = np.asarray(arrays[field])
        if a.shape != (N_FRAMES,):
            raise ValueError("{}.{}: shape {} != ({},)".format(name, field, a.shape, N_FRAMES))
        if kind == "bool":
            if a.dtype != bool:
                raise ValueError("{}.{}: dtype {} is not bool".format(name, field, a.dtype))
            continue
        if not np.issubdtype(a.dtype, np.floating):
            raise ValueError("{}.{}: dtype {} is not float".format(name, field, a.dtype))
        nan = np.isnan(a)
        if nan.any() and field not in allow_nan:
            raise ValueError("{}.{}: {} NaN values".format(name, field, int(nan.sum())))
        if kind == "prob":
            finite = a[~nan]
            if finite.size and (finite.min() < 0 or finite.max() > 1):
                raise ValueError("{}.{}: values outside [0, 1] (min {:.3f}, max {:.3f})".format(
                    name, field, finite.min(), finite.max()))
    if name == "p_team":
        s = arrays["p_home"] + arrays["p_away"]
        if np.abs(s - 1).max() > 1e-4:
            raise ValueError("p_team: p_home + p_away must be 1 (max error {:.4f})".format(
                np.abs(s - 1).max()))
    if name == "ball_clean":
        bad = np.isnan(arrays["x"]) & (arrays["conf"] > 0)
        if bad.any():
            raise ValueError("ball_clean: {} frames have NaN position but conf > 0".format(
                int(bad.sum())))


def save(name, **arrays):
    """Check, then write data/derived/<name>.npz. Extra fields are allowed and kept."""
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    for field, kind in SPEC[name]["fields"].items():
        if field in arrays and kind != "bool":
            arrays[field] = arrays[field].astype(np.float32)
    check(name, arrays)
    os.makedirs(DERIVED_DIR, exist_ok=True)
    np.savez(path(name), **arrays)
    return path(name)


def load(name):
    """Read and re-check data/derived/<name>.npz. Returns a dict of arrays."""
    p = path(name)
    if not os.path.isfile(p):
        raise IOError("{} not found. Run `python run_all.py` (or the owner's step) first.".format(p))
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files}
    check(name, arrays)
    return arrays
