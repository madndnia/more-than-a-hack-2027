"""One command from the data bundle to the submission file.

    python run_all.py --team TEAMNAME            # -> submissions/TEAMNAME.csv, validated
    python run_all.py --team TEAMNAME --force    # also rebuild the loader cache

Steps (each owner only edits their own module; this file is Person 1's):
    1. mtah.loader          (P1)  data/derived/frames.npz
    2. mtah.ball            (P3)  data/derived/ball_clean.npz
    3. src/make_p_dead.py   (P2)  data/derived/p_dead.csv
       mtah.p2_adapter      (P5)  p_dead.csv -> data/derived/p_dead.npz (checked, with sha256)
    4. mtah.team            (P4)  data/derived/p_team.npz        needs ball_clean
    5. mtah.tags            (P5)  data/derived/tag_prior.npz     built for the record; the decoder
                                  does not use it (disabled by P5, see CONTRACT.md)
    6. mtah.decoder         (P5)  P4 LLR + P2 dead mask -> Viterbi -> submissions/<team>.csv
    7. validate_submission.py + diagnostics + sha256 of the file

Deterministic on purpose: fixed seeds, no wall-clock or random input. Two runs on the same data
must print the same sha256 -- that is the reproducibility rule in RULES.md.
"""
import argparse
import hashlib
import os
import random
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

from mtah import ball, decoder, diagnostics, loader, p2_adapter, tags, team  # noqa: E402
from mtah.contract import DERIVED_DIR, N_FRAMES, SUBMISSIONS_DIR  # noqa: E402

P2_SCRIPT = os.path.join(ROOT, "src", "make_p_dead.py")
P2_CSV = os.path.join(DERIVED_DIR, "p_dead.csv")


def write_submission(path, labels):
    assert len(labels) == N_FRAMES
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_p2():
    """P2 ships a standalone script that writes a CSV; run it, then adapt it to the NPZ contract."""
    result = subprocess.run([sys.executable, P2_SCRIPT, "--out", P2_CSV])
    if result.returncode != 0:
        sys.exit("P2 (src/make_p_dead.py) failed")
    return p2_adapter.adapt(P2_CSV, replace=True)


def step(name, fn):
    t0 = time.time()
    print("[{}] ...".format(name), flush=True)
    out = fn()
    print("[{}] done in {:.1f}s".format(name, time.time() - t0), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regenerate the submission from the data bundle")
    ap.add_argument("--team", default="TEAMNAME", help="team name; the CSV is named after it")
    ap.add_argument("--force", action="store_true", help="rebuild the loader cache too")
    args = ap.parse_args()

    step("loader  P1", lambda: loader.build(force=args.force))
    step("ball    P3", ball.build)
    step("dead    P2", run_p2)
    step("team    P4", team.build)
    step("tags    P5", tags.build)
    labels, info = step("decoder P5", lambda: decoder.decode(return_info=True))
    print("decoder: switch cost {:.4f}, unresolved alive stretches {}".format(
        info["switch_cost"], info["unresolved"]))

    out = write_submission(os.path.join(SUBMISSIONS_DIR, args.team + ".csv"), labels)
    print("\nwrote", os.path.relpath(out, ROOT))

    result = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation",
                                                          "validate_submission.py"), out])
    if result.returncode != 0:
        sys.exit("VALIDATION FAILED -- do not submit this file")

    print()
    diagnostics.print_table([(os.path.basename(out), labels)])
    print("\nsha256  {}".format(sha256(out)))
