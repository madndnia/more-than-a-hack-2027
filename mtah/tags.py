"""Person 5 -- Smart Tagging prior.  Output: tag_prior (prior_home, prior_away, restart).

DUMMY v0 (Person 1 scaffold): +0.15 towards the team of every active phase tag, restart = 1 inside
SET PIECES. N/A team tags give no team evidence. Person 5 replaces build() and keeps the output
contract (SPEC["tag_prior"]).
"""
import numpy as np

from mtah import contract
from mtah.loader import load_frames


def build():
    F = load_frames()
    lean = 0.15 * (F["tag_home"].astype(float) - F["tag_away"].astype(float))
    prior_home = np.clip(0.5 + lean, 0.05, 0.95)
    restart = F["tag_set_piece"].astype(float)
    return contract.save("tag_prior", prior_home=prior_home, prior_away=1.0 - prior_home,
                         restart=restart)


if __name__ == "__main__":
    print(build())
