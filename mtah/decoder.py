"""P5 submission entry point using P4's reference decoder, without rebuilding P2.

    python -m mtah.decoder --p-dead-csv data/derived/p_dead.csv

Reads existing frames/p_team artifacts. Smart Tagging is disabled: arbitrary
external priors have no verified per-phase budget. No second smoothing or duration
filter is used.
"""
import argparse
import csv
import os
from pathlib import Path
import tempfile

import numpy as np

from evaluation import possession_metrics as pm
from mtah import contract, loader, ref_decoder, team

PRIOR_DISABLED = ("tag prior is disabled: an external prior's per-phase budget is not verified; "
                  "team agreement and a verified integration are required before enabling it")


def validate_half(half):
    half = np.asarray(half)
    if half.ndim != 1 or half.dtype.kind not in "iu" or not np.isin(half, (0, 1, 2)).all():
        raise ValueError("half must be a 1-D integer array containing only 0, 1, 2")
    bounds = []
    for h in (1, 2):
        indices = np.flatnonzero(half == h)
        if indices.size:
            if indices[-1] - indices[0] + 1 != indices.size:
                raise ValueError("each half must occupy one contiguous block")
            bounds.append((h, indices[0], indices[-1]))
    if len(bounds) == 2 and bounds[0][2] >= bounds[1][1]:
        raise ValueError("half 1 must precede half 2")
    return half


def load_half():
    # Deliberately do not call load_frames(): it builds the loader when absent.
    if not os.path.isfile(loader.FRAMES_PATH):
        raise FileNotFoundError("frames.npz is missing; obtain the existing P1 loader cache first")
    with np.load(loader.FRAMES_PATH, allow_pickle=False) as data:
        half = validate_half(data["half"])
    if half.shape != (contract.N_FRAMES,):
        raise ValueError("frames.npz: half has the wrong frame count")
    return half


def _finite_vector(values, name, n):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError("{} must have one finite value per frame".format(name))
    return result


def decode_arrays(p_team, p_dead, half, tag_llr=None, return_info=False):
    """Pure wrapper: the default path is exactly P4's summed LLR/dead-mask path.

    All-zero alive stretches follow P4: borrow from the same half, otherwise
    remain X and count as unresolved. A missing channel is an error, not zero.
    """
    if tag_llr is not None:
        raise ValueError(PRIOR_DISABLED)
    half = validate_half(half)
    n = len(half)
    llr = (_finite_vector(p_team["llr_ball"], "llr_ball", n)
           + _finite_vector(p_team["llr_noball"], "llr_noball", n))
    if not np.isfinite(llr).all():
        raise ValueError("summed P4 evidence is not finite")
    ep = np.asarray(p_team.get("ep_llr", team.EP_LLR_PRIOR), dtype=float)
    if ep.shape != () or not np.isfinite(ep) or ep <= 0:
        raise ValueError("ep_llr must be a finite positive scalar")
    switch = team.default_switch_cost(float(ep))
    dead_probability = _finite_vector(p_dead, "p_dead", n)
    if np.any((dead_probability < 0) | (dead_probability > 1)):
        raise ValueError("p_dead must be in [0, 1]")
    dead = ref_decoder.dead_mask(dead_probability, half)
    labels, info = ref_decoder.decode(llr, dead, half, switch, return_info=True)
    info.update(switch_cost=float(switch), ep_llr=float(ep), tag_prior=False)
    return (labels, info) if return_info else labels


def decode(p_dead_csv=None, replace_npz=False, use_tag_prior=False, return_info=False):
    """Keep the no-argument pipeline interface; never rebuild another component."""
    if use_tag_prior:
        raise ValueError(PRIOR_DISABLED)
    half = load_half()
    p_team = contract.load("p_team")
    if p_dead_csv is not None:
        from mtah.p2_adapter import adapt
        adapt(p_dead_csv, replace=replace_npz)
    p_dead = contract.load("p_dead")["p_dead"]
    return decode_arrays(p_team, p_dead, half, return_info=return_info)


def validate_file(path):
    reason = pm.validate_submission(pm.read_label_csv(path), contract.N_FRAMES)
    if reason:
        raise ValueError("invalid submission: " + reason)


def write_submission(path, labels, overwrite=False):
    """Validate before atomically publishing a canonical UTF-8 frame,label CSV.

    Even with overwrite=True, refuse to replace anything except a valid existing
    submission. In particular, p_dead.csv and NPZ inputs cannot be overwritten.
    """
    labels = np.asarray(labels)
    if labels.shape != (contract.N_FRAMES,) or not np.isin(labels, list("HADX")).all():
        raise ValueError("labels must have N_FRAMES entries, each exactly H, A, D or X")
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError("output exists; choose another path or pass --overwrite")
        validate_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=str(path.parent), suffix=".csv", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write("frame,label\n")
            handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
        validate_file(temporary)
        if path.exists():
            if not overwrite:
                raise FileExistsError("output was created by another run")
            validate_file(path)
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p-dead-csv", help="adapt P2 CSV; without this, read the existing NPZ")
    parser.add_argument("--replace-npz", action="store_true", help="explicitly replace a stale P2 NPZ")
    parser.add_argument("--output", default=os.path.join(contract.SUBMISSIONS_DIR, "submission.csv"))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing valid submission")
    parser.add_argument("--use-tag-prior", action="store_true",
                        help="currently rejected: external per-phase budgets are not verified")
    args = parser.parse_args(argv)
    if args.replace_npz and args.p_dead_csv is None:
        parser.error("--replace-npz requires --p-dead-csv")
    try:
        labels, info = decode(args.p_dead_csv, args.replace_npz, args.use_tag_prior, return_info=True)
        write_submission(args.output, labels, overwrite=args.overwrite)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, "P5 failed: {}\n".format(error))
    print("P5 OK: {} rows; tag prior {}; unresolved alive stretches {}".format(
        len(labels), "ON" if info["tag_prior"] else "OFF", info["unresolved"]))
    print("ep_llr {:.6f}; switch cost {:.6f}".format(info["ep_llr"], info["switch_cost"]))


if __name__ == "__main__":
    main()
