import argparse
import math
import warnings

import numpy as np

from mtah import contract
from mtah.loader import load_frames

FPS = contract.FPS
HOME, AWAY = 0, 1

GAP_REACH = 13
INTERP_W = 0.5

TAU_D = 0.5
R_IN = 1.5
R_OUT = 3.0
R_COUNT = 3.0
GHOST_WIN = 75
GHOST_W = 0.5

V_IN = 7.0
V_OUT = 12.0
SPEED_SPAN = 2
SPEED_MED = 5

HOLD = 6
HOLD_LEN = 2 * HOLD + 1
E_MIN = 0.1
CLEAR_S = 0.5
EP_GAP = 50
BLIP_MAX = 25
BLIP_FLANK = 75
EP_PRIOR_A, EP_PRIOR_B = 1.0, 19.0
EP_LLR_PRIOR = float(np.log(19.0))
EP_LLR_MIN = 1.0
EP_LLR_MAX = float(np.log(99.0))
WEAK_EP_FRAC = 0.35
MIN_CONTROL = HOLD_LEN

NB_MIN_TRACKED = 8
NB_VEL_SPAN = 12
NB_SMOOTH = 25
NB_EDGE = NB_SMOOTH + NB_VEL_SPAN
NB_TRAIN_GAP = 75
NB_MIN_RUNS = 20
NB_L2 = 1.0
LLR_NB_MAX = 2.0
TAU_MAX = 250
NB_SEGMENT_MAX = 1.0
NB_CAL_MIN_LEN = 25
NB_CAL_MIN = 20
NB_CAL_L2 = 0.1
NB_LOC_MIN_LEN = 50
NB_LOC_MIN = 20
NB_LOC_PASS = 0.55
NB_LOC_P = 0.05

FEATURES = ("f_vel", "f_spread")
BIG = 10 ** 9


def labelled_blocks(half):
    blocks = []
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if len(idx):
            blocks.append((h, int(idx[0]), int(idx[-1]) + 1))
    return blocks


def half_blocks(half):
    return [(lo, hi) for _, lo, hi in labelled_blocks(half)]


def ramp_down(v, full, zero):
    out = np.clip((zero - v) / (zero - full), 0.0, 1.0)
    return np.where(np.isnan(out), 0.0, out)


def centred_mean(a, w, half, nan_aware=False):
    out = np.full(len(a), np.nan if nan_aware else 0.0)
    for lo, hi in half_blocks(half):
        seg = np.asarray(a[lo:hi], dtype=np.float64)
        valid = ~np.isnan(seg)
        csum = np.concatenate(([0.0], np.cumsum(np.where(valid, seg, 0.0))))
        i = np.arange(hi - lo)
        l, h = np.maximum(0, i - w), np.minimum(hi - lo, i + w + 1)
        s = csum[h] - csum[l]
        if nan_aware:
            ccnt = np.concatenate(([0], np.cumsum(valid)))
            k = ccnt[h] - ccnt[l]
            out[lo:hi] = np.where(k > 0, s / np.maximum(k, 1), np.nan)
        else:
            out[lo:hi] = s / (h - l)
    return out


def running_median(a, k):
    pad = k // 2
    padded = np.concatenate((np.full(pad, np.nan), a, np.full(pad, np.nan)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(windows, axis=1)


def central_diff(a, span, half):
    out = np.full(len(a), np.nan)
    for lo, hi in half_blocks(half):
        seg = a[lo:hi]
        if hi - lo > 2 * span:
            out[lo + span:hi - span] = (seg[2 * span:] - seg[:-2 * span]) / (2 * span)
    return out


def last_seen_index(a, limit):
    rows = np.arange(a.shape[0], dtype=np.int32)[:, None]
    last = np.maximum.accumulate(np.where(np.isnan(a), np.int32(-1), rows), axis=0)
    return np.where((last >= 0) & (rows - last <= limit), last, -1)


def take_rows(a, idx):
    out = np.take_along_axis(a, np.maximum(idx, 0), axis=0)
    return np.where(idx >= 0, out, np.nan).astype(a.dtype)


def nearest_index_distance(mask, half):
    out = np.full(len(mask), BIG, dtype=np.int64)
    for lo, hi in half_blocks(half):
        m = mask[lo:hi]
        pos = np.arange(hi - lo)
        last = np.maximum.accumulate(np.where(m, pos, -BIG))
        nxt = np.minimum.accumulate(np.where(m, pos, 2 * BIG)[::-1])[::-1]
        out[lo:hi] = np.minimum(pos - last, nxt - pos)
    return out


def flanking_labels(lab, half):
    n = len(lab)
    prev_lab, next_lab = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    prev_d, next_d = np.full(n, BIG, dtype=np.int64), np.full(n, BIG, dtype=np.int64)
    for lo, hi in half_blocks(half):
        L = lab[lo:hi]
        pos = np.arange(hi - lo)
        has = L != 0
        last = np.maximum.accumulate(np.where(has, pos, -1))
        ok = last >= 0
        prev_lab[lo:hi][ok] = L[last[ok]]
        prev_d[lo:hi][ok] = (pos - last)[ok]
        nxt = np.minimum.accumulate(np.where(has, pos, BIG)[::-1])[::-1]
        ok = nxt < BIG
        next_lab[lo:hi][ok] = L[nxt[ok]]
        next_d[lo:hi][ok] = (nxt - pos)[ok]
    return prev_lab, prev_d, next_lab, next_d


def cap_groups(llr, ids, cap):
    out = np.asarray(llr, dtype=np.float64).copy()
    m = ids >= 0
    if not m.any():
        return out, 0
    g, v = ids[m], out[m]
    n_groups = int(g.max()) + 1
    if np.ndim(cap):
        cap = np.asarray(cap, dtype=np.float64)
        n_groups = max(n_groups, len(cap))
    else:
        cap = np.full(n_groups, float(cap))
    pos = np.bincount(g, weights=np.where(v > 0, v, 0.0), minlength=n_groups)
    neg = -np.bincount(g, weights=np.where(v < 0, v, 0.0), minlength=n_groups)
    sp = np.minimum(1.0, cap / np.maximum(pos, 1e-12))
    sn = np.minimum(1.0, cap / np.maximum(neg, 1e-12))
    out[m] = np.where(v > 0, v * sp[g], v * sn[g])
    return out, int(np.sum((pos > cap) | (neg > cap)))


def binomial_p_one_sided(successes, n):
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(successes, n + 1)) / 2.0 ** n


def runs(values, half):
    ids = np.full(len(values), -1, dtype=np.int64)
    firsts, lasts, vals, halves = [], [], [], []
    k = 0
    for h, lo, hi in labelled_blocks(half):
        v = values[lo:hi]
        new = np.ones(hi - lo, dtype=bool)
        new[1:] = v[1:] != v[:-1]
        ids[lo:hi] = np.cumsum(new) - 1 + k
        starts = np.flatnonzero(new)
        firsts.append(starts + lo)
        lasts.append(np.r_[starts[1:] - 1, hi - lo - 1] + lo)
        vals.append(v[starts])
        halves.append(np.full(len(starts), h))
        k += len(starts)

    def cat(xs):
        return np.concatenate(xs) if xs else np.array([], dtype=np.int64)

    return ids, dict(first=cat(firsts), last=cat(lasts), value=cat(vals), half=cat(halves))


def episodes(sign, max_gap, half):
    ids = np.full(len(sign), -1, dtype=np.int64)
    nz = np.flatnonzero(sign != 0)
    if nz.size == 0:
        empty = np.array([], dtype=np.int64)
        return ids, dict(first=empty, last=empty, sign=empty, half=empty)
    s, h = sign[nz], half[nz]
    new = np.ones(nz.size, dtype=bool)
    new[1:] = (s[1:] != s[:-1]) | (np.diff(nz) - 1 > max_gap) | (h[1:] != h[:-1])
    ids[nz] = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:] - 1, nz.size - 1]
    return ids, dict(first=nz[starts], last=nz[ends], sign=s[starts], half=h[starts])


def n_runs(mask):
    m = np.asarray(mask, dtype=np.int8)
    return int(m[0]) + int(np.sum((m[1:] == 1) & (m[:-1] == 0))) if m.size else 0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def balanced_weights(t):
    n_pos, n_neg = t.sum(), len(t) - t.sum()
    if n_pos == 0 or n_neg == 0:
        return None
    return np.where(t > 0, 0.5 * len(t) / n_pos, 0.5 * len(t) / n_neg)


def fit_logistic(X, y, l2=NB_L2, iters=100):
    X = np.asarray(X, dtype=np.float64)
    t = (np.asarray(y) > 0).astype(np.float64)
    n, k = X.shape
    sw = balanced_weights(t)
    if sw is None:
        return np.zeros(k), 0.0
    A = np.hstack([X, np.ones((n, 1))])
    R = np.eye(k + 1) * l2
    R[-1, -1] = 0.0
    beta = np.zeros(k + 1)
    for _ in range(iters):
        p = sigmoid(A @ beta)
        grad = A.T @ (sw * (t - p)) - R @ beta
        hess = (A * (sw * p * (1 - p))[:, None]).T @ A + R
        step = np.linalg.solve(hess + 1e-9 * np.eye(k + 1), grad)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta[:-1], float(beta[-1])


def fit_scale(s, y, l2=NB_CAL_L2, iters=50):
    s = np.asarray(s, dtype=np.float64)
    t = (np.asarray(y) > 0).astype(np.float64)
    sw = balanced_weights(t)
    if sw is None:
        return 0.0
    a = 0.0
    for _ in range(iters):
        p = sigmoid(a * s)
        step = (np.sum(sw * (t - p) * s) - l2 * a) / (np.sum(sw * p * (1 - p) * s * s) + l2)
        a += step
        if abs(step) < 1e-9:
            break
    return float(a)


def autocorr_time(z, valid, half, max_lag=TAU_MAX):
    _, table = runs(np.asarray(valid, dtype=np.int8), half)
    segs = [z[a:b + 1] for a, b, v in zip(table["first"], table["last"], table["value"])
            if v == 1 and b - a + 1 > 10]
    if not segs:
        return 1.0, False
    mean = np.mean(np.concatenate(segs))
    segs = [s - mean for s in segs]
    var = sum(float(np.sum(s * s)) for s in segs) / sum(len(s) for s in segs)
    if var <= 0:
        return 1.0, False
    tau = 1.0
    for lag in range(1, max_lag):
        num = sum(float(np.sum(s[:-lag] * s[lag:])) for s in segs if len(s) > lag)
        den = sum(len(s) - lag for s in segs if len(s) > lag)
        if den == 0:
            return float(np.clip(tau, 1.0, max_lag)), True
        rho = num / den / var
        if rho < 0.05:
            return float(np.clip(tau, 1.0, max_lag)), False
        tau += 2.0 * rho
    return float(np.clip(tau, 1.0, max_lag)), True


def control_value(n_frames):
    s = np.r_[np.ones(4 * HOLD_LEN), -np.ones(n_frames), np.zeros(4 * HOLD_LEN)]
    hold = centred_mean(s, HOLD, np.ones(len(s), dtype=np.int8))
    hold = np.where(np.abs(hold) >= E_MIN, hold, 0.0)
    return min(float(-hold[hold < 0].sum()) / HOLD_LEN, 1.0)


def default_switch_cost(ep_llr):
    floor = max(NB_SEGMENT_MAX, WEAK_EP_FRAC * ep_llr)
    must = control_value(MIN_CONTROL) * ep_llr
    return 0.5 * (floor + must) if must > floor else must


def ball_speed(x, y, half):
    v = np.hypot(central_diff(x, SPEED_SPAN, half), central_diff(y, SPEED_SPAN, half)) * FPS
    out = np.full(len(x), np.nan)
    for lo, hi in half_blocks(half):
        out[lo:hi] = running_median(v[lo:hi], SPEED_MED)
    return out


def ball_reliability(x, observed, conf, half):
    dist = nearest_index_distance(np.asarray(observed, dtype=bool), half)
    rel = np.where(observed, 1.0, INTERP_W * np.clip(1.0 - dist / GAP_REACH, 0.0, 1.0))
    rel = np.minimum(rel, np.nan_to_num(conf, nan=0.0))
    rel[np.isnan(x) | (half == 0)] = 0.0
    return rel


def kernel_direction(d_inf, weight, home):
    dmin = d_inf.min(axis=1)
    shift = np.where(np.isfinite(dmin), dmin, 0.0)[:, None]
    with np.errstate(invalid="ignore"):
        k = weight * np.exp(-(d_inf - shift) / TAU_D)
    k = np.nan_to_num(k, nan=0.0, posinf=0.0)
    kh, ka = k[:, home].sum(axis=1), k[:, ~home].sum(axis=1)
    tot = kh + ka
    return np.where(tot > 0, (kh - ka) / np.where(tot > 0, tot, 1.0), 0.0), dmin


def finite_or_nan(v):
    return np.where(np.isfinite(v), v, np.nan)


def at_the_ball(px, py, slot_team, bx, by):
    home = slot_team == HOME
    tracked = ~np.isnan(px)
    idx = last_seen_index(px, GHOST_WIN)
    gx, gy = take_rows(px, idx), take_rows(py, idx)
    ghost = ~tracked & (idx >= 0)
    age = np.arange(px.shape[0], dtype=np.int32)[:, None] - idx
    w_ghost = np.where(ghost, GHOST_W * np.clip(1.0 - age / GHOST_WIN, 0.0, 1.0), 0.0)
    d = np.hypot(gx - bx[:, None].astype(np.float32), gy - by[:, None].astype(np.float32))
    d = np.where(np.isnan(d), np.inf, d)

    d_tr = np.where(tracked, d, np.inf)
    dir_tr, dmin_tr = kernel_direction(d_tr, tracked.astype(np.float32), home)
    w_all = np.where(tracked, 1.0, w_ghost).astype(np.float32)
    dir_all, _ = kernel_direction(np.where(w_all > 0, d, np.inf), w_all, home)

    same_side = dir_tr * dir_all > 0
    direction = np.where(same_side,
                         np.sign(dir_tr) * np.minimum(np.abs(dir_tr), np.abs(dir_all)), 0.0)
    near = d_tr < R_COUNT
    return dict(direction=direction, dmin=finite_or_nan(dmin_tr),
                d_home=finite_or_nan(d_tr[:, home].min(axis=1)),
                d_away=finite_or_nan(d_tr[:, ~home].min(axis=1)),
                n_home_near=near[:, home].sum(axis=1), n_away_near=near[:, ~home].sum(axis=1),
                ghost_blocked=(dir_tr != 0) & ~same_side)


def significant_episodes(hold_s, ep_ids, ep):
    m = ep_ids >= 0
    clear = (np.abs(hold_s) >= CLEAR_S)[m].astype(float)
    return np.bincount(ep_ids[m], weights=clear, minlength=len(ep["first"])) > 0


def episode_llr(e):
    return float(np.clip(np.log((1.0 - e) / e), EP_LLR_MIN, EP_LLR_MAX))


def estimate_episode_llr(ep, significant):
    idx = np.flatnonzero(significant)
    if idx.size == 0:
        e = EP_PRIOR_A / (EP_PRIOR_A + EP_PRIOR_B)
        return e, 0, episode_llr(e)
    first, last = ep["first"][idx], ep["last"][idx]
    sign, half = ep["sign"][idx], ep["half"][idx]
    blip = np.zeros(idx.size, dtype=bool)
    if idx.size >= 3:
        i = np.arange(1, idx.size - 1)
        blip[i] = ((last[i] - first[i] + 1 < BLIP_MAX)
                   & (sign[i - 1] == -sign[i]) & (sign[i + 1] == -sign[i])
                   & (half[i - 1] == half[i]) & (half[i + 1] == half[i])
                   & (first[i] - last[i - 1] <= BLIP_FLANK)
                   & (first[i + 1] - last[i] <= BLIP_FLANK))
    n_blip = int(blip.sum())
    e = (n_blip + EP_PRIOR_A) / (idx.size + EP_PRIOR_A + EP_PRIOR_B)
    return e, n_blip, episode_llr(e)


def ball_channel(direction, dmin_tr, speed, rel, half):
    q = ramp_down(dmin_tr, R_IN, R_OUT) * ramp_down(speed, V_IN, V_OUT) * rel
    hold_s = centred_mean(direction * q, HOLD, half)
    hold_q = centred_mean(q, HOLD, half)
    s_eff = np.where(np.abs(hold_s) >= E_MIN, hold_s, 0.0)
    ep_ids, ep = episodes(np.sign(s_eff).astype(np.int8), EP_GAP, half)
    significant = significant_episodes(hold_s, ep_ids, ep)
    e, n_blip, ep_llr = estimate_episode_llr(ep, significant)
    look = ep_llr * s_eff
    caps = np.where(significant, ep_llr, WEAK_EP_FRAC * ep_llr)
    llr, n_capped = cap_groups(look / HOLD_LEN, ep_ids, caps)
    return dict(hold_s=hold_s, hold_q=hold_q, look_ball=look, llr_ball=llr, ep_ids=ep_ids, ep=ep,
                significant=significant, episodes_capped=n_capped, ep_error=e, n_blips=n_blip,
                ep_llr=ep_llr)


def attack_direction(px, slot_team, half):
    home = slot_team == HOME
    out = np.zeros(len(half))
    by_half = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for h, lo, hi in labelled_blocks(half):
            mh = np.nanmean(px[lo:hi][:, home])
            ma = np.nanmean(px[lo:hi][:, ~home])
            d = 1.0 if mh < ma else -1.0
            out[lo:hi] = d
            by_half[h] = (d, float(mh), float(ma))
    return out, by_half


def team_centroid_spread(px, py, mask):
    x, y = px[:, mask], py[:, mask]
    n = np.sum(~np.isnan(x), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cx = np.nanmean(x, axis=1)
        spread = np.sqrt(np.nanvar(x, axis=1) + np.nanvar(y, axis=1))
    ok = n >= NB_MIN_TRACKED
    return np.where(ok, cx, np.nan), np.where(ok, spread, np.nan), n


def noball_features(px, py, slot_team, half, direction):
    home = slot_team == HOME
    cx_h, sp_h, n_h = team_centroid_spread(px, py, home)
    cx_a, sp_a, n_a = team_centroid_spread(px, py, ~home)
    vel = central_diff(0.5 * (cx_h + cx_a), NB_VEL_SPAN, half) * FPS * direction
    X = np.stack([centred_mean(vel, NB_SMOOTH, half, nan_aware=True), sp_h - sp_a], axis=1)
    X[half == 0] = np.nan
    for lo, hi in half_blocks(half):
        for j in range(X.shape[1]):
            col = X[lo:hi, j]
            v = col[~np.isnan(col)]
            mad = 1.4826 * float(np.median(np.abs(v - np.median(v)))) if v.size else 0.0
            X[lo:hi, j] = col / (mad if mad > 1e-9 else 1.0)
    return X, n_h, n_a


def stretches(ep_ids, ep, significant, half):
    n = len(half)
    delta = np.zeros(n + 1, dtype=np.int64)
    sig = np.flatnonzero(significant)
    np.add.at(delta, ep["first"][sig], 1)
    np.add.at(delta, ep["last"][sig] + 1, -1)
    ids, t = runs((np.cumsum(delta[:-1]) > 0).astype(np.int8), half)
    sig_sign = np.zeros(max(len(ep["first"]), 1), dtype=np.int8)
    sig_sign[sig] = ep["sign"][sig]
    frame_sign = np.where(ep_ids >= 0, sig_sign[np.maximum(ep_ids, 0)], 0)
    silent = t["value"] == 0
    left = np.zeros(len(silent), dtype=np.int8)
    right = np.zeros(len(silent), dtype=np.int8)
    for i in np.flatnonzero(silent):
        a, b = t["first"][i], t["last"][i]
        if a > 0 and half[a - 1] == half[a]:
            left[i] = frame_sign[a - 1]
        if b + 1 < n and half[b + 1] == half[b]:
            right[i] = frame_sign[b + 1]
    t.update(silent=silent, left=left, right=right, length=t["last"] - t["first"] + 1)
    return ids, t


def enough_pseudo_labels(m, y):
    return min(n_runs(m & (y > 0)), n_runs(m & (y < 0))) >= NB_MIN_RUNS


def trimmed_bounds(t):
    a = t["first"] + NB_EDGE
    b = t["last"] + 1 - NB_EDGE
    return a, b, np.maximum(b - a, 0)


def fit_out_of_fold(X, train, y, valid, half):
    z = np.zeros(len(half))
    present = [h for h, _, _ in labelled_blocks(half)]
    if len(present) == 2 and all(enough_pseudo_labels(train & (half == h), y) for h in present):
        folds = {}
        for h in present:
            other = train & (half != h) & (half > 0)
            w, b = fit_logistic(X[other], y[other])
            m = valid & (half == h)
            z[m] = X[m] @ w + b
            folds[h] = (w, b)
        return z, folds, True
    w, b = fit_logistic(X[train], y[train])
    z[valid] = X[valid] @ w + b
    return z, {0: (w, b)}, False


def localisation_test(llr, t):
    agree = total = 0
    ta, tb, tlen = trimmed_bounds(t)
    candidates = (t["silent"] & (t["left"] != 0) & (t["right"] != 0)
                  & (t["left"] != t["right"]) & (tlen >= NB_LOC_MIN_LEN))
    for i in np.flatnonzero(candidates):
        a, b = ta[i], tb[i]
        mid = (a + b) // 2
        move = llr[mid:b].sum() - llr[a:mid].sum()
        if move != 0:
            total += 1
            agree += int(np.sign(move) == t["right"][i])
    if total == 0:
        return float("nan"), 0, 1.0
    return agree / total, total, binomial_p_one_sided(agree, total)


def calibrate_temperature(raw, st):
    ta, tb, tlen = trimmed_bounds(st)
    csum = np.concatenate(([0.0], np.cumsum(raw)))
    sums = np.where(tlen > 0, csum[np.minimum(tb, len(raw))] - csum[np.minimum(ta, len(raw))], 0.0)
    labelled = (st["silent"] & (st["left"] != 0) & (st["left"] == st["right"])
                & (tlen >= NB_CAL_MIN_LEN))
    n_h = int(np.sum(labelled & (st["left"] > 0)))
    n_a = int(np.sum(labelled & (st["left"] < 0)))
    if min(n_h, n_a) < NB_CAL_MIN:
        return None, n_h, n_a
    a = float(np.clip(fit_scale(sums[labelled], st["left"][labelled]), 0.0, 1.0))
    return a, n_h, n_a


def noball_channel(X, hold_s, hold_rel, half, stretch_ids, st):
    valid = np.all(~np.isnan(X), axis=1) & (half > 0)
    clear = np.where(hold_s >= CLEAR_S, 1, np.where(hold_s <= -CLEAR_S, -1, 0)).astype(np.int8)
    clear[half == 0] = 0
    pl, pd, nl, nd = flanking_labels(clear, half)
    train = (valid & (clear == 0) & (pl != 0) & (pl == nl)
             & (pd <= NB_TRAIN_GAP) & (nd <= NB_TRAIN_GAP))
    y = pl
    zeros = np.zeros(len(half))
    out = dict(valid=valid, clear=clear, train=train, y=y, z=zeros, w_noball=zeros,
               llr_noball=zeros, look_noball=zeros)
    info = dict(n_home=int(np.sum(train & (y > 0))), n_away=int(np.sum(train & (y < 0))),
                runs_home=n_runs(train & (y > 0)), runs_away=n_runs(train & (y < 0)),
                enabled=False, reason="", folds={}, oof=False, tau=1.0, tau_ceiling=False,
                a=0.0, calibrated=False, cal_home=0, cal_away=0, capped=0,
                loc_agree=float("nan"), loc_n=0, loc_p=1.0)
    if not enough_pseudo_labels(train, y):
        info["reason"] = "too few pseudo-labelled stretches to fit"
        return out, info

    z_raw, folds, oof = fit_out_of_fold(X, train, y, valid, half)
    z = np.clip(z_raw, -LLR_NB_MAX, LLR_NB_MAX)
    tau, ceiling = autocorr_time(z, valid, half)
    tau = max(tau, float(HOLD_LEN))
    w_nb = np.clip(1.0 - hold_rel, 0.0, 1.0)
    w_nb[half == 0] = 0.0
    raw = w_nb * z / tau
    out.update(z=z, w_noball=w_nb)
    info.update(folds=folds, oof=oof, tau=tau, tau_ceiling=ceiling)

    a, cal_h, cal_a = calibrate_temperature(raw, st)
    info.update(cal_home=cal_h, cal_away=cal_a)
    if a is None:
        info["reason"] = "too few labelled stretches to calibrate"
        return out, info

    llr, n_capped = cap_groups(raw * a, stretch_ids, NB_SEGMENT_MAX)
    agree, n_loc, p_loc = localisation_test(llr, st)
    info.update(a=a, calibrated=True, capped=n_capped, loc_agree=agree, loc_n=n_loc, loc_p=p_loc)
    if n_loc < NB_LOC_MIN:
        info["reason"] = "too few change-of-possession stretches to test localisation"
        return out, info
    if not (agree >= NB_LOC_PASS and p_loc < NB_LOC_P):
        info["reason"] = "fails the localisation test (not distinguishable from chance)"
        return out, info

    info.update(enabled=True, reason="passes")
    out.update(llr_noball=llr, look_noball=a * w_nb * z)
    return out, info


def compute(F, ball):
    half = np.asarray(F["half"])
    slot_team = np.asarray(F["slot_team"])
    px, py = F["px"], F["py"]
    bx = np.asarray(ball["x"], dtype=np.float64)
    by = np.asarray(ball["y"], dtype=np.float64)

    rel = ball_reliability(bx, np.asarray(ball["observed"], dtype=bool), ball["conf"], half)
    speed = ball_speed(bx, by, half)
    atb = at_the_ball(px, py, slot_team, bx, by)
    ch1 = ball_channel(atb["direction"], atb["dmin"], speed, rel, half)

    attack, dir_info = attack_direction(px, slot_team, half)
    X, n_h, n_a = noball_features(px, py, slot_team, half, attack)
    stretch_ids, st = stretches(ch1["ep_ids"], ch1["ep"], ch1["significant"], half)
    ch2, nb_info = noball_channel(X, ch1["hold_s"], centred_mean(rel, HOLD, half), half,
                                  stretch_ids, st)

    p_home = sigmoid(ch1["look_ball"] + ch2["look_noball"])
    p_home[half == 0] = 0.5
    out = dict(p_home=p_home, conf=np.abs(2.0 * p_home - 1.0),
               llr_ball=ch1["llr_ball"], llr_noball=ch2["llr_noball"], w_noball=ch2["w_noball"],
               hold_s=ch1["hold_s"], hold_q=ch1["hold_q"], ball_rel=rel, ball_speed=speed,
               d_home=atb["d_home"], d_away=atb["d_away"],
               n_home_near=atb["n_home_near"], n_away_near=atb["n_away_near"],
               ghost_blocked=atb["ghost_blocked"], n_home_tracked=n_h, n_away_tracked=n_a,
               **{k: ch2[k] for k in ("z", "valid", "clear", "train", "y")})
    info = dict(episodes_capped=ch1["episodes_capped"], n_episodes=len(ch1["ep"]["first"]),
                n_significant=int(ch1["significant"].sum()), ep_error=ch1["ep_error"],
                n_blips=ch1["n_blips"], ep_llr=ch1["ep_llr"], direction=dir_info,
                n_stretches=int(st["silent"].sum()),
                **{"nb_" + k: v for k, v in nb_info.items()})
    return out, info


def build(report=False):
    F = load_frames()
    out, info = compute(F, contract.load("ball_clean"))
    if report:
        print_report(F, out, info)
    f32 = np.float32
    return contract.save(
        "p_team", p_home=out["p_home"], p_away=1.0 - out["p_home"], conf=out["conf"],
        llr_ball=out["llr_ball"].astype(f32), llr_noball=out["llr_noball"].astype(f32),
        w_noball=out["w_noball"].astype(f32), hold_s=out["hold_s"].astype(f32),
        hold_q=out["hold_q"].astype(f32), ball_rel=out["ball_rel"].astype(f32),
        ball_speed=np.nan_to_num(out["ball_speed"], nan=-1.0).astype(f32),
        d_home=np.nan_to_num(out["d_home"], nan=-1.0).astype(f32),
        d_away=np.nan_to_num(out["d_away"], nan=-1.0).astype(f32),
        n_home_near=out["n_home_near"].astype(f32), n_away_near=out["n_away_near"].astype(f32),
        ep_llr=np.array(info["ep_llr"], dtype=f32))


def print_report(F, out, info):
    g = np.asarray(F["graded"], dtype=bool)

    def pct(m):
        return 100.0 * float(np.mean(m))

    print("\np_team report (graded frames)")
    obs = np.asarray(F["ball_observed"], dtype=bool) & g
    sp = out["ball_speed"][obs]
    sp = sp[~np.isnan(sp)]
    if sp.size:
        print("  ball speed where observed: median {:.1f} m/s, p90 {:.1f}, p99 {:.1f}".format(
            *np.percentile(sp, [50, 90, 99])))
    if "ball_c3" in F:
        c3 = np.asarray(F["ball_c3"])
        both = obs & ~np.isnan(c3) & ~np.isnan(out["ball_speed"])
        if both.sum() > 100:
            r = float(np.corrcoef(c3[both], out["ball_speed"][both])[0, 1])
            print("  ball_c3 (unused 3rd raw field): median {:.2f}, p99 {:.2f}; "
                  "corr with ball speed {:.2f}".format(np.median(c3[both]),
                                                       np.percentile(c3[both], 99), r))
    rel = out["ball_rel"][g]
    print("  ball reliability: observed {:.1f}%   trusted interpolation {:.1f}%   none {:.1f}%"
          .format(pct(rel >= 1.0), pct((rel > 0) & (rel < 1.0)), pct(rel == 0)))
    print("  tracked per team: home {:.1f}  away {:.1f} on average; both >= {}: {:.1f}%".format(
        out["n_home_tracked"][g].mean(), out["n_away_tracked"][g].mean(), NB_MIN_TRACKED,
        pct((out["n_home_tracked"][g] >= NB_MIN_TRACKED)
            & (out["n_away_tracked"][g] >= NB_MIN_TRACKED))))
    hs = out["hold_s"][g]
    print("  ball evidence |hold_s| >= {}: {:.1f}%   clear: home {:.1f}%  away {:.1f}%   "
          "ghost-blocked {:.1f}%".format(E_MIN, pct(np.abs(hs) >= E_MIN), pct(hs >= CLEAR_S),
                                         pct(hs <= -CLEAR_S), pct(out["ghost_blocked"][g])))
    print("  episodes {} (significant {}), blips {} -> episode error {:.3f} -> ep_llr {:.2f} nats"
          " [{:.1f}, {:.1f}]; episodes capped {}; switch cost {:.2f}".format(
              info["n_episodes"], info["n_significant"], info["n_blips"], info["ep_error"],
              info["ep_llr"], EP_LLR_MIN, EP_LLR_MAX, info["episodes_capped"],
              default_switch_cost(info["ep_llr"])))
    for h, (d, mh, ma) in info["direction"].items():
        print("  half {}: home attacks towards {} x   (mean x home {:.1f} m, away {:.1f} m)".format(
            h, "high" if d > 0 else "low", mh, ma))
    print("  no-ball: features on {:.1f}% of graded frames; pseudo-labels home {} frames in {} "
          "stretches, away {} in {} (need >= {} per class)".format(
              pct(out["valid"][g]), info["nb_n_home"], info["nb_runs_home"], info["nb_n_away"],
              info["nb_runs_away"], NB_MIN_RUNS))
    for h, (w, b) in info["nb_folds"].items():
        name = "fitted on the half other than {}".format(h) if h else "in-sample"
        print("    model {}: {}   intercept {:+.2f}".format(
            name, "  ".join("{} {:+.2f}".format(n, v) for n, v in zip(FEATURES, w)), b))
    if info["nb_folds"]:
        print("    autocorrelation time {}{:.0f} frames ({:.1f} s)".format(
            ">= " if info["nb_tau_ceiling"] else "", info["nb_tau"], info["nb_tau"] / FPS))
        print("    temperature a = {:.2f} ({}; labelled stretches home {} away {})".format(
            info["nb_a"], "out-of-fold" if info["nb_oof"] else "in-sample",
            info["nb_cal_home"], info["nb_cal_away"]))
        print("    localisation: right way in {:.1f}% of {} stretches (p = {:.3f}; pass >= {:.0f}% "
              "and p < {})".format(100 * info["nb_loc_agree"], info["nb_loc_n"],
                                   info["nb_loc_p"], 100 * NB_LOC_PASS, NB_LOC_P))
    print("  no-ball channel {}: {}".format("ON" if info["nb_enabled"] else "OFF",
                                            info["nb_reason"]))
    print("  total |LLR| over graded frames: ball {:.0f} nats   no-ball {:.0f} nats".format(
        np.abs(out["llr_ball"][g]).sum(), np.abs(out["llr_noball"][g]).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="P4: H vs A possession evidence -> p_team")
    ap.add_argument("--report", action="store_true", help="print aggregate sanity numbers")
    print(build(report=ap.parse_args().report))
