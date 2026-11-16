"""Person 2 -- dead-ball detector.  Output: p_dead (P(D) per frame).

DUMMY v0 (Person 1 scaffold): 1 outside the halves, 0.6 where the ball is not tracked, 0.1 where
it is. Person 2 replaces build() and keeps the output contract (SPEC["p_dead"]).
"""
import numpy as np

from mtah import contract
from mtah.loader import load_frames


def build():
    F = load_frames()
    p = np.where(F["ball_observed"], 0.1, 0.6)
    p[~F["graded"]] = 1.0
    return contract.save("p_dead", p_dead=p)


if __name__ == "__main__":
    print(build())
