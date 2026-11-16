"""Person 5 -- decoder.  Input: p_dead, p_team, tag_prior.  Output: H/A/D/X label per frame.

DUMMY v0 (Person 1 scaffold): per-frame argmax, then a +/-50-frame majority vote inside each half
(roughly baseline_smooth50). X outside the halves -- those frames are never graded.
Person 5 replaces decode() with the HMM/Viterbi and keeps the signature.
"""
import numpy as np

from mtah import contract
from mtah.loader import load_frames

WINDOW = 50


def _majority(codes, window):
    # codes: 0=H 1=A 2=D ; centred majority vote, cumulative sums so window size is free
    n = len(codes)
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    counts = []
    for k in range(3):
        c = np.concatenate(([0], np.cumsum(codes == k)))
        counts.append(c[hi] - c[lo])
    return np.argmax(np.stack(counts), axis=0)


def decode():
    F = load_frames()
    p_dead = contract.load("p_dead")["p_dead"]
    team = contract.load("p_team")
    prior = contract.load("tag_prior")

    p_home = 0.7 * team["p_home"] + 0.3 * prior["prior_home"]
    codes = np.where(p_dead > 0.5, 2, np.where(p_home >= 0.5, 0, 1))

    labels = np.full(len(codes), "X", dtype="<U1")
    for h in (1, 2):
        idx = np.flatnonzero(F["half"] == h)
        if len(idx):
            labels[idx] = np.array(["H", "A", "D"])[_majority(codes[idx], WINDOW)]
    return labels


if __name__ == "__main__":
    labels = decode()
    print({k: int((labels == k).sum()) for k in "HADX"})
