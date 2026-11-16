"""Shared loader (Person 1): ATD + homography + Smart Tagging -> one per-frame table.

Everybody else starts from this instead of reading the raw files themselves, so the off-by-one
fixes live in exactly one place:

    ATD          video_frame = frame_id - 1
    Smart Tag    video_frame = round(seconds * 25)
    homography   frame is already the video frame

Build once (about a minute, it streams the raw ATD), then everything reads the cache:

    python -m mtah.loader            # builds data/derived/frames.npz and prints a summary
    python -m mtah.loader --force    # rebuild even if the cache is fresh

    from mtah.loader import load_frames, frames_df
    F = load_frames()        # dict of arrays, see FIELDS below
    df = frames_df()         # pandas, per-frame scalar columns only (no player matrices)

FIELDS (every 1-D array has length 199319, index = video frame = submission row)
    frame            0..199318
    half             1, 2, or 0 outside the two halves
    graded           True on the 158,857 frames that are scored
    ball_x, ball_y   ball position in METRES (NaN where the tracker lost it -- 41% of graded)
    ball_c3          third field of the raw ball group, as delivered (NaN if absent)
    ball_observed    ~isnan(ball_x)
    px, py, ps       (199319, n_slots) player x/y in metres and speed in m/s, NaN if not tracked
    slot_team        (n_slots,) 0 = home (H, Barcelona), 1 = away (A, Athletic)
    n_home, n_away   number of tracked players of each team on the frame
    hom_valid        homography exists on that frame
    hom_err          homography reprojection error (lower = better)
    tag_home         number of active Smart Tagging phases tagged Barcelona (set pieces excluded)
    tag_away         same for Athletic
    tag_na           number of active instances whose Team is N/A
    tag_set_piece    True inside any SET PIECES instance
    tag_sp_type      index into SP_TYPES of the active set piece, -1 if none
    tag_sp_team      0 = H, 1 = A, -1 = unknown/N/A for the active set piece

Also writes data/derived/tags_events.csv: one row per Smart Tagging instance with frames.
"""

import argparse
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, asset_paths, half_video_blocks, require  # noqa: E402
from mtah.contract import DERIVED_DIR, N_FRAMES, PITCH  # noqa: E402

FRAMES_PATH = os.path.join(DERIVED_DIR, "frames.npz")
EVENTS_PATH = os.path.join(DERIVED_DIR, "tags_events.csv")

SP_TYPES = ["Throw-in", "Corner Kick", "Goal Kick", "Free Kick",
            "Kick Off (After Goal)", "Kick Off (Start)"]
HOME, AWAY, UNKNOWN = 0, 1, -1


# -- ATD --------------------------------------------------------------------------------------

def team_names(meta_path):
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def slot_teams(meta_path):
    """0/1 per track slot; the first team in the metadata is home (FIFA/EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _parse_group(group):
    vals = group.split(",")
    out = [np.nan, np.nan, np.nan]
    for i, v in enumerate(vals[:3]):
        try:
            out[i] = float(v)
        except ValueError:
            pass
    return out


def read_atd(raw_path, n_slots):
    """Stream the raw ATD. Returns ball (N, 3) and players (N, n_slots, 3), normalised 0-1."""
    ball = np.full((N_FRAMES, 3), np.nan, dtype=np.float64)
    players = np.full((N_FRAMES, n_slots, 3), np.nan, dtype=np.float32)
    n_bad = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                n_bad += 1
                continue
            row = int(parts[0]) - 1          # ATD frame_id counts from 1, video from 0
            if not 0 <= row < N_FRAMES:
                n_bad += 1
                continue
            ball[row] = _parse_group(parts[2])
            groups = parts[1].split(";")
            flat = parts[1].replace(";", ",").split(",")
            if len(groups) == n_slots and len(flat) == 3 * n_slots:
                # fast path: every slot has exactly x,y,s ("NaN" parses as nan)
                players[row] = np.array(flat, dtype=np.float64).reshape(n_slots, 3)
            else:
                for slot, group in enumerate(groups[:n_slots]):
                    players[row, slot] = _parse_group(group)
    if n_bad:
        print("  warning: skipped {} malformed ATD lines".format(n_bad))
    return ball, players


# -- Smart Tagging ----------------------------------------------------------------------------

def _team_code(team, home_name):
    """Map a Smart Tagging Team value to H/A. Names differ between files (Athletic/Atletic),
    so match the home club by its first word and treat any other real club as away."""
    t = (team or "").strip().lower()
    if not t or t == "n/a":
        return UNKNOWN
    key = (home_name or "barcelona").strip().lower().split()[0]
    return HOME if key in t or "barcelona" in t else AWAY


def tag_arrays(events, home_name):
    tag_home = np.zeros(N_FRAMES, dtype=np.int16)
    tag_away = np.zeros(N_FRAMES, dtype=np.int16)
    tag_na = np.zeros(N_FRAMES, dtype=np.int16)
    tag_set_piece = np.zeros(N_FRAMES, dtype=bool)
    tag_sp_type = np.full(N_FRAMES, -1, dtype=np.int8)
    tag_sp_team = np.full(N_FRAMES, -1, dtype=np.int8)
    types_lower = [t.lower() for t in SP_TYPES]

    for ev in events:
        a = max(0, ev["start_frame"])
        b = min(N_FRAMES, ev["end_frame"])      # half-open, as the starter loader returns
        if b <= a:
            continue
        team = _team_code(ev["tags"].get("Team"), home_name)
        is_sp = (ev["code"] or "").strip().lower() == "set pieces"
        if team == UNKNOWN:
            tag_na[a:b] += 1
        if is_sp:
            tag_set_piece[a:b] = True
            sp_type = (ev["tags"].get("Type") or "").strip().lower()
            if sp_type in types_lower:
                tag_sp_type[a:b] = types_lower.index(sp_type)
            tag_sp_team[a:b] = team
        elif team == HOME:
            tag_home[a:b] += 1
        elif team == AWAY:
            tag_away[a:b] += 1
    return dict(tag_home=tag_home, tag_away=tag_away, tag_na=tag_na,
                tag_set_piece=tag_set_piece, tag_sp_type=tag_sp_type, tag_sp_team=tag_sp_team)


def write_events_csv(events, home_name, path):
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["start_frame", "end_frame", "code", "team", "side", "type", "half"])
        for ev in events:
            team = _team_code(ev["tags"].get("Team"), home_name)
            w.writerow([ev["start_frame"], ev["end_frame"], ev["code"],
                        {HOME: "H", AWAY: "A", UNKNOWN: "N/A"}[team],
                        ev["tags"].get("Side", ""), ev["tags"].get("Type", ""),
                        ev["tags"].get("Half", "")])


# -- build ------------------------------------------------------------------------------------

def _inputs():
    return require(["atd_meta", "atd_raw", "homography", "smart_tagging"])


def is_fresh():
    if not os.path.isfile(FRAMES_PATH):
        return False
    built = os.path.getmtime(FRAMES_PATH)
    return all(os.path.getmtime(p) < built for p in _inputs())


def build(force=False, verbose=True):
    if not force and is_fresh():
        if verbose:
            print("frames.npz is up to date (use --force to rebuild)")
        return FRAMES_PATH

    import load_homography
    import load_smart_tagging

    t0 = time.time()
    meta_path, raw_path, _, _ = _inputs()
    home_name, away_name = team_names(meta_path)
    teams = slot_teams(meta_path)
    if verbose:
        print("teams: H = {}, A = {}   ({} track slots)".format(home_name, away_name, len(teams)))
        print("reading ATD ...")
    ball, players = read_atd(raw_path, len(teams))

    length, width = PITCH
    ball_x = ball[:, 0] * length
    ball_y = ball[:, 1] * width
    px = players[:, :, 0] * np.float32(length)
    py = players[:, :, 1] * np.float32(width)
    ps = players[:, :, 2]
    tracked = ~np.isnan(px)
    n_home = tracked[:, teams == HOME].sum(axis=1).astype(np.int8)
    n_away = tracked[:, teams == AWAY].sum(axis=1).astype(np.int8)

    half = np.zeros(N_FRAMES, dtype=np.int8)
    for h, first, last in half_video_blocks(GAME):
        half[first:last + 1] = h

    if verbose:
        print("reading homography and Smart Tagging ...")
    _, hom_err, hom_valid = load_homography.load()
    if len(hom_valid) != N_FRAMES:
        raise ValueError("homography has {} rows, expected {}".format(len(hom_valid), N_FRAMES))
    events = load_smart_tagging.load()
    tags = tag_arrays(events, home_name)

    os.makedirs(DERIVED_DIR, exist_ok=True)
    np.savez(
        FRAMES_PATH,
        frame=np.arange(N_FRAMES, dtype=np.int32),
        half=half, graded=half > 0,
        ball_x=ball_x, ball_y=ball_y, ball_c3=ball[:, 2], ball_observed=~np.isnan(ball_x),
        px=px, py=py, ps=ps, slot_team=teams, n_home=n_home, n_away=n_away,
        hom_valid=hom_valid.astype(bool), hom_err=hom_err.astype(np.float32),
        sp_types=np.array(SP_TYPES), team_names=np.array([home_name, away_name]),
        **tags)
    write_events_csv(events, home_name, EVENTS_PATH)
    if verbose:
        print("wrote {} ({:.0f} s)".format(os.path.relpath(FRAMES_PATH, ROOT), time.time() - t0))
    return FRAMES_PATH


def load_frames():
    if not os.path.isfile(FRAMES_PATH):
        build()
    with np.load(FRAMES_PATH) as z:
        return {k: z[k] for k in z.files}


def frames_df():
    import pandas as pd
    F = load_frames()
    return pd.DataFrame({k: v for k, v in F.items() if v.ndim == 1 and len(v) == N_FRAMES})


def summary(F):
    g = F["graded"]
    print("\nframes.npz summary")
    print("  frames {}   graded {}   halves {}".format(len(g), int(g.sum()),
          [(h, int((F["half"] == h).sum())) for h in (1, 2)]))
    print("  ball observed: {:.1f}% of all, {:.1f}% of graded   (DATA.md: 56.9% / 59.3%)".format(
        100 * F["ball_observed"].mean(), 100 * F["ball_observed"][g].mean()))
    print("  tracked players per graded frame: mean {:.1f}   (DATA.md: 20.5)".format(
        (F["n_home"][g] + F["n_away"][g]).mean()))
    print("  homography valid: {:.1f}%".format(100 * F["hom_valid"].mean()))
    print("  Smart Tagging coverage of graded frames: home {:.1f}%  away {:.1f}%  "
          "set piece {:.1f}%".format(100 * (F["tag_home"][g] > 0).mean(),
                                     100 * (F["tag_away"][g] > 0).mean(),
                                     100 * F["tag_set_piece"][g].mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build data/derived/frames.npz")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build(force=args.force)
    summary(load_frames())
