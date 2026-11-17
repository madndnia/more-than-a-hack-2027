"""P5 experimental Smart Tagging prior: at most one nat per phase, OFF in decoder.

    python -m mtah.tags

Reads P1's normalized tags_events.csv and cached half mask, never raw assets or GT.
"""
import argparse
import csv

import numpy as np

from mtah import loader, p5_io

ATTACK = frozenset(("PLAYERS IN THE BOX", "BALL IN FINAL THIRD", "BALL IN THE BOX",
                    "CREATING CHANCES", "LONG BALL", "PROGRESSION",
                    "ATTACKING TRANSITION", "BUILD UP"))
DEFENCE = frozenset(("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                     "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"))


def read_events(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"start_frame", "end_frame", "code", "team"}
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError("tag events need unique start_frame,end_frame,code,team columns")
        result = []
        for row in reader:
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError("malformed tag event row")
            try:
                start, end = int(row["start_frame"]), int(row["end_frame"])
            except ValueError:
                raise ValueError("tag event boundaries must be integer video frames") from None
            result.append(dict(start_frame=start, end_frame=end, code=row["code"], team=row["team"]))
    return result


def build_prior(events, half, phase_cap=1.0):
    """Distribute a signed budget over live-half frames in each half-open phase.

    Attack supports the named team; defence supports its opponent. N/A/unknown
    phases give no team evidence. Set-piece windows are restart metadata only.
    Overlapping phases add, each with its own <=1-nat absolute evidence budget.
    Audit arrays describe contributing phases; llr_tag retains float64 precision.
    """
    half = np.asarray(half)
    if half.ndim != 1 or half.dtype.kind not in "iu" or not np.isin(half, (0, 1, 2)).all():
        raise ValueError("half must contain integer 0, 1, 2 values")
    if not np.isfinite(phase_cap) or not 0 <= phase_cap <= 1:
        raise ValueError("phase_cap must be finite and in [0, 1] nats")
    n = len(half)
    live = half > 0
    prefix = np.r_[0, np.cumsum(live)]
    delta = np.zeros(n + 1)
    coverage = np.zeros(n + 1, dtype=np.int64)
    restart_delta = np.zeros(n + 1, dtype=np.int64)
    starts, ends, counts, budgets = [], [], [], []
    for event in events:
        start, end = event["start_frame"], event["end_frame"]
        if (not isinstance(start, (int, np.integer)) or isinstance(start, (bool, np.bool_))
                or not isinstance(end, (int, np.integer)) or isinstance(end, (bool, np.bool_))
                or end < start):
            raise ValueError("tag phases require integer half-open boundaries with end >= start")
        a, b = max(0, min(n, int(start))), max(0, min(n, int(end)))
        count = int(prefix[b] - prefix[a])
        if count == 0:
            continue
        code = " ".join(str(event["code"]).upper().split())
        side = str(event["team"]).strip().upper()
        if code == "SET PIECES":
            restart_delta[a] += 1
            restart_delta[b] -= 1
            continue
        sign = {"H": 1, "A": -1, "N/A": 0, "": 0}.get(side)
        if sign is None:
            raise ValueError("tag teams must use P1's normalized H/A/N/A codes")
        if sign == 0 or code not in ATTACK | DEFENCE or phase_cap == 0:
            continue
        if code in DEFENCE:
            sign = -sign
        budget = sign * float(phase_cap)
        delta[a] += budget / count
        delta[b] -= budget / count
        coverage[a] += 1
        coverage[b] -= 1
        starts.append(a)
        ends.append(b)
        counts.append(count)
        budgets.append(budget)
    llr = np.cumsum(delta[:-1])
    llr[~live | (np.cumsum(coverage[:-1]) == 0)] = 0.0
    prior_home = 1.0 / (1.0 + np.exp(-np.clip(llr, -30.0, 30.0)))
    return dict(prior_home=prior_home, prior_away=1.0 - prior_home,
                restart=(live & (np.cumsum(restart_delta[:-1]) > 0)).astype(float),
                llr_tag=llr, phase_start=np.asarray(starts, dtype=np.int32),
                phase_end=np.asarray(ends, dtype=np.int32),
                phase_frames=np.asarray(counts, dtype=np.int32),
                phase_budget=np.asarray(budgets, dtype=float))


def build(events_path=None):
    from mtah.decoder import load_half
    half = load_half()
    events = read_events(loader.EVENTS_PATH if events_path is None else events_path)
    return p5_io.save_npz("tag_prior", replace=True, **build_prior(events, half))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", help="P1 normalized events CSV (default: data/derived/tags_events.csv)")
    args = parser.parse_args(argv)
    try:
        build(args.events)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, "Tag prior failed: {}\n".format(error))
    print("Tag prior OK; phase budget <= 1 nat; decoder default remains OFF")


if __name__ == "__main__":
    main()
