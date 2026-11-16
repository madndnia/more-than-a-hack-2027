"""Person 4 -- H vs A possession model.  Output: p_team (p_home, p_away, conf).

DUMMY v0 (Person 1 scaffold): team of the nearest tracked player to ball_clean, 0.8/0.2, and
0.5/0.5 where there is no ball. Person 4 replaces build() and keeps the output contract
(SPEC["p_team"]).
"""
import numpy as np

from mtah import contract
from mtah.loader import load_frames


def build():
    F = load_frames()
    ball = contract.load("ball_clean")
    dx = F["px"] - ball["x"][:, None].astype(np.float32)
    dy = F["py"] - ball["y"][:, None].astype(np.float32)
    d2 = dx * dx + dy * dy
    has = ~np.all(np.isnan(d2), axis=1)
    nearest = np.zeros(len(d2), dtype=np.int64)
    nearest[has] = np.nanargmin(d2[has], axis=1)
    home_near = F["slot_team"][nearest] == 0
    p_home = np.where(has, np.where(home_near, 0.8, 0.2), 0.5)
    conf = np.where(has, ball["conf"], 0.0)
    return contract.save("p_team", p_home=p_home, p_away=1.0 - p_home, conf=conf)


if __name__ == "__main__":
    print(build())
