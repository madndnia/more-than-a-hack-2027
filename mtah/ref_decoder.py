import argparse
import os
import subprocess
import sys

import numpy as np

from mtah import contract, diagnostics, team
from mtah.loader import load_frames

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEAD_SMOOTH = 12


def viterbi_two_state(e, switch):
    return _viterbi_forward(np.asarray(e)[::-1], switch)[::-1]


def _viterbi_forward(e, switch):
    n = len(e)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    from_h = np.zeros(n, dtype=bool)
    from_a = np.zeros(n, dtype=bool)
    vh, va = e[0] / 2.0, -e[0] / 2.0
    for t in range(1, n):
        stay_h, move_h = vh, va - switch
        stay_a, move_a = va, vh - switch
        from_h[t] = stay_h >= move_h
        from_a[t] = stay_a >= move_a
        vh = (stay_h if from_h[t] else move_h) + e[t] / 2.0
        va = (stay_a if from_a[t] else move_a) - e[t] / 2.0
    path = np.empty(n, dtype=np.int8)
    path[-1] = 1 if vh >= va else -1
    for t in range(n - 1, 0, -1):
        if path[t] == 1:
            path[t - 1] = 1 if from_h[t] else -1
        else:
            path[t - 1] = -1 if from_a[t] else 1
    return path


def decode(llr, dead, half, switch, return_info=False):
    labels = np.full(len(half), "X", dtype="<U1")
    alive = (half > 0) & ~dead
    labels[(half > 0) & dead] = "D"
    _, t = team.runs(alive.astype(np.int8), half)
    decided, empty = [], []
    for a, b, v, h in zip(t["first"], t["last"], t["value"], t["half"]):
        if v != 1:
            continue
        if np.any(llr[a:b + 1] != 0):
            path = viterbi_two_state(llr[a:b + 1], switch)
            labels[a:b + 1] = np.where(path > 0, "H", "A")
            decided.append((a, b, h, labels[a], labels[b]))
        else:
            empty.append((a, b, h))

    unresolved = 0
    for a, b, h in empty:
        before = [d for d in decided if d[2] == h and d[1] < a]
        after = [d for d in decided if d[2] == h and d[0] > b]
        if before:
            labels[a:b + 1] = before[-1][4]
        elif after:
            labels[a:b + 1] = after[0][3]
        else:
            unresolved += 1

    if return_info:
        return labels, dict(empty_runs=len(empty),
                            empty_frames=sum(b - a + 1 for a, b, _ in empty),
                            unresolved=unresolved)
    return labels


def dead_mask(p_dead, half):
    return team.centred_mean((p_dead >= 0.5).astype(float), DEAD_SMOOTH, half) >= 0.5


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def main():
    ap = argparse.ArgumentParser(description="P4 reference decoder: Viterbi on p_team LLRs")
    ap.add_argument("--switch", type=float, default=None,
                    help="switch cost in nats (default: team.default_switch_cost(ep_llr))")
    ap.add_argument("--name", default="P4REF", help="output file name, without .csv")
    ap.add_argument("--no-noball", action="store_true", help="ball channel only")
    args = ap.parse_args()

    F = load_frames()
    half = F["half"]
    pt = contract.load("p_team")
    ep_llr = float(pt["ep_llr"]) if "ep_llr" in pt else team.EP_LLR_PRIOR
    switch = args.switch if args.switch is not None else team.default_switch_cost(ep_llr)
    llr = pt["llr_ball"].astype(np.float64)
    if not args.no_noball:
        llr = llr + pt["llr_noball"].astype(np.float64)

    dead = dead_mask(contract.load("p_dead")["p_dead"], half)
    labels, dinfo = decode(llr, dead, half, switch, return_info=True)

    _, t = team.runs(((half > 0) & ~dead).astype(np.int8), half)
    alive_len = (t["last"] - t["first"] + 1)[t["value"] == 1]
    graded = int(np.sum(half > 0))
    print("alive stretches between D: {}, shorter than 2 s: {}".format(
        len(alive_len), int(np.sum(alive_len < 2 * team.FPS))))
    print("alive stretches without evidence: {} ({:.1f}% of graded frames), unresolved {}".format(
        dinfo["empty_runs"], 100.0 * dinfo["empty_frames"] / max(graded, 1), dinfo["unresolved"]))
    print("ep_llr {:.2f} nats, switch cost {:.2f} nats{}".format(
        ep_llr, switch, ", ball channel only" if args.no_noball else ""))

    out = os.path.join(contract.SUBMISSIONS_DIR, args.name + ".csv")
    write_submission(out, labels)
    print("wrote", os.path.relpath(out, ROOT))
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                    out], check=False)

    named = [(os.path.basename(out), labels)]
    for other in sorted(os.listdir(contract.SUBMISSIONS_DIR)):
        path = os.path.join(contract.SUBMISSIONS_DIR, other)
        if other.endswith(".csv") and path != out:
            named.append((other, diagnostics.pm.read_label_csv(path)["label"].values))
    print()
    diagnostics.print_table(named)


if __name__ == "__main__":
    main()
