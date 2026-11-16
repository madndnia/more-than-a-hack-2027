"""Team-wide diagnostics that need NO ground truth (agree on these, never tune on the public GT).

    python -m mtah.diagnostics submissions/TEAM.csv [submissions/OTHER.csv ...]

For each file, over the graded halves only:
    poss        number of possessions (maximal H or A runs, D excluded)
    dur_s       mean possession length in seconds
    D%          share of D frames
    H_share     home share of alive frames
    short<2s    share of possessions shorter than 2 s (flicker detector)
Reference points from the README: ground truth on the public 5 min has 17 possessions of mean
14.2 s; baseline_raw invents ~131 there with 2.0 s mean. A full match is typically a few hundred
possessions of ~10-20 s. Use these as plausibility checks, not as targets.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

FPS = 25
RUNS = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]


def stats(labels):
    codes = pm.label_codes(np.asarray(labels))
    filled = pm.fill_abstentions(codes, RUNS)
    spells = pm.possession_spells(filled, RUNS)
    lengths = np.array([b - a for a, b, _ in spells]) / FPS
    graded = np.concatenate([filled[a:b] for a, b in RUNS])
    alive = np.sum((graded == pm.H) | (graded == pm.A))
    return {
        "poss": len(spells),
        "dur_s": float(lengths.mean()) if len(lengths) else 0.0,
        "D%": 100.0 * float(np.mean(graded == pm.D)),
        "H_share": float(np.sum(graded == pm.H)) / alive if alive else 0.0,
        "short<2s": 100.0 * float(np.mean(lengths < 2)) if len(lengths) else 0.0,
    }


def print_table(named):
    print("{:<28} {:>6} {:>7} {:>6} {:>8} {:>9}".format(
        "file", "poss", "dur_s", "D%", "H_share", "short<2s"))
    for name, labels in named:
        s = stats(labels)
        print("{:<28} {:>6} {:>7.1f} {:>6.1f} {:>8.2f} {:>8.1f}%".format(
            name[:28], s["poss"], s["dur_s"], s["D%"], s["H_share"], s["short<2s"]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    named = []
    for path in sys.argv[1:]:
        named.append((os.path.basename(path), pm.read_label_csv(path)["label"].values))
    print_table(named)
