import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import team
from mtah.ref_decoder import decode, viterbi_two_state

Y_PLAY = 34.0
Y_PARK = 5.0


class Scene:
    def __init__(self, n=3000, n_home=3, n_away=3, halves=((100, 2900),)):
        self.n = n
        self.half = np.zeros(n, dtype=np.int8)
        for h, (a, b) in enumerate(halves, start=1):
            self.half[a:b] = h
        s = n_home + n_away
        self.slot_team = np.array([0] * n_home + [1] * n_away, dtype=np.int8)
        self.px = np.full((n, s), np.nan, dtype=np.float32)
        self.py = np.full((n, s), np.nan, dtype=np.float32)
        live = self.half > 0
        for k in range(s):
            self.px[live, k] = 10.0 + 6.0 * k
            self.py[live, k] = Y_PARK
        self.bx = np.full(n, np.nan)
        self.by = np.full(n, np.nan)
        self.observed = None
        self.conf = None

    def put(self, slot, frames, x, y=Y_PLAY):
        self.px[frames, slot] = x
        self.py[frames, slot] = y

    def lose(self, slot, frames):
        self.px[frames, slot] = np.nan
        self.py[frames, slot] = np.nan

    def ball(self, frames, x, y=Y_PLAY):
        self.bx[frames] = x
        self.by[frames] = y

    def run(self):
        observed = ~np.isnan(self.bx) if self.observed is None else self.observed
        conf = observed.astype(float) if self.conf is None else self.conf
        F = dict(half=self.half, slot_team=self.slot_team, px=self.px, py=self.py)
        ball = dict(x=self.bx, y=self.by, observed=observed, conf=conf)
        return team.compute(F, ball)


def frames(a, b):
    return np.arange(a, b)


def test_dribble_is_strong_home_evidence():
    sc = Scene()
    f = frames(500, 600)
    x = 30 + 4.0 * (f - 500) / 25
    sc.put(0, f, x)
    sc.ball(f, x + 0.5)
    out, info = sc.run()
    mid = frames(520, 580)
    assert out["llr_ball"][f].sum() > 0.9 * info["ep_llr"]
    assert out["p_home"][mid].min() > 0.9
    assert out["conf"][mid].min() > 0.8


def test_pass_flying_past_an_opponent_is_not_control():
    sc = Scene()
    f = frames(500, 530)
    sc.ball(f, 30 + 18.0 * (f - 500) / 25)
    sc.put(3, frames(400, 700), 30 + 18.0 * 15 / 25, Y_PLAY + 0.5)
    out, _ = sc.run()
    assert abs(out["llr_ball"][f].sum()) < 0.3


def test_close_duel_favours_the_closer_player():
    sc = Scene()
    f = frames(500, 600)
    sc.put(0, f, 50.0)
    sc.put(3, f, 51.9)
    sc.ball(f, 50.5)
    out, _ = sc.run()
    assert out["hold_s"][frames(520, 580)].min() > team.CLEAR_S


def test_equal_duel_is_neutral():
    sc = Scene()
    f = frames(500, 600)
    sc.put(0, f, 49.0)
    sc.put(3, f, 51.0)
    sc.ball(f, 50.0)
    out, _ = sc.run()
    assert np.abs(out["hold_s"][frames(520, 580)]).max() < team.E_MIN
    assert np.abs(out["llr_ball"][f]).sum() < 1e-9
    assert out["conf"][frames(520, 580)].max() < 1e-9


def test_ghost_blocks_the_opponent():
    sc = Scene()
    f = frames(500, 700)
    sc.put(0, f, 70.0)
    sc.ball(f, 70.3)
    sc.put(3, f, 72.0)
    sc.lose(0, frames(600, 700))
    out, _ = sc.run()
    fresh = frames(610, 660)
    assert np.abs(out["llr_ball"][fresh]).sum() == 0.0
    assert out["ghost_blocked"][fresh].all()


def test_ghost_alone_at_the_ball_is_no_evidence():
    sc = Scene()
    f = frames(500, 700)
    sc.put(0, f, 70.0)
    sc.ball(f, 70.3)
    sc.lose(0, frames(600, 700))
    out, _ = sc.run()
    assert np.abs(out["llr_ball"][frames(610, 700)]).sum() == 0.0


def test_ghost_never_creates_evidence_for_its_team():
    sc = Scene()
    sc.put(0, frames(400, 600), 60.0)
    sc.lose(0, frames(600, 900))
    f = frames(600, 760)
    x = 50 + 6.0 * (f - 600) / 25
    sc.put(3, f, x - 0.4)
    sc.ball(f, x)
    out, _ = sc.run()
    assert out["llr_ball"][f].max() <= 1e-12
    assert out["llr_ball"][f].sum() < 0


def test_long_interpolated_gap_is_not_a_ball():
    sc = Scene()
    sc.ball(frames(495, 500), 20.0)
    sc.ball(frames(800, 805), 80.0)
    gap = frames(500, 800)
    sc.bx[gap] = np.interp(gap, [499, 800], [20.0, 80.0])
    sc.by[gap] = Y_PLAY
    sc.observed = np.zeros(sc.n, dtype=bool)
    sc.observed[495:500] = sc.observed[800:805] = True
    sc.conf = np.where(sc.observed, 1.0, 0.3)
    mid = frames(640, 660)
    sc.put(0, mid, sc.bx[mid] + 0.3)
    out, _ = sc.run()
    assert np.all(out["ball_rel"][mid] == 0.0)
    assert np.all(out["llr_ball"][mid] == 0.0)


def test_one_control_episode_is_capped():
    sc = Scene()
    f = frames(300, 1300)
    x = 20 + 1.5 * (f - 300) / 25
    sc.put(0, f, x)
    sc.ball(f, x + 0.4)
    out, info = sc.run()
    total = out["llr_ball"][f].sum()
    assert total <= info["ep_llr"] + 1e-6
    assert total > 0.9 * info["ep_llr"]
    assert info["episodes_capped"] >= 1


def test_cap_is_not_bypassed_by_short_silences():
    sc = Scene()
    f = frames(300, 1300)
    x = 20 + 1.5 * (f - 300) / 25
    sc.put(0, f, x)
    sc.ball(f, x + 0.4)
    for start in range(375, 1300, 75):
        sc.bx[start:start + 25] = np.nan
        sc.by[start:start + 25] = np.nan
    out, info = sc.run()
    assert info["n_episodes"] == 1
    assert out["llr_ball"][f].sum() <= info["ep_llr"] + 1e-6


def test_blips_set_the_value_of_an_episode():
    sc = Scene(n=6000, halves=((100, 5900),))
    for k in range(40):
        a = 200 + 140 * k
        sc.put(0, frames(a, a + 50), 40.0)
        sc.ball(frames(a, a + 50), 40.4)
        sc.put(3, frames(a + 50, a + 65), 40.8)
        sc.ball(frames(a + 50, a + 65), 40.4)
        sc.put(0, frames(a + 50, a + 65), 30.0)
        sc.put(0, frames(a + 65, a + 115), 40.0)
        sc.ball(frames(a + 65, a + 115), 40.4)
        sc.put(3, frames(a + 65, a + 115), 52.0)
    _, info = sc.run()
    assert info["n_blips"] >= 30
    assert 0.2 < info["ep_error"] < 0.5
    assert info["ep_llr"] == team.EP_LLR_MIN


def test_episode_value_is_monotone_and_starts_at_the_prior():
    empty = dict(first=np.array([], dtype=np.int64), last=np.array([], dtype=np.int64),
                 sign=np.array([], dtype=np.int8), half=np.array([], dtype=np.int8))
    _, _, llr0 = team.estimate_episode_llr(empty, np.zeros(0, dtype=bool))
    assert abs(llr0 - team.EP_LLR_PRIOR) < 1e-12

    def with_blips(n_eps, n_blips):
        first, last, sign = [], [], []
        t = 0
        for k in range(n_eps):
            is_blip = k % 2 == 1 and (k // 2) < n_blips
            length = 10 if is_blip else 100
            first.append(t)
            last.append(t + length - 1)
            sign.append(-1 if is_blip else 1)
            t += length + 5
        ep = dict(first=np.array(first), last=np.array(last), sign=np.array(sign, dtype=np.int8),
                  half=np.ones(n_eps, dtype=np.int8))
        return team.estimate_episode_llr(ep, np.ones(n_eps, dtype=bool))

    values = [with_blips(501, k)[2] for k in (0, 1, 5, 50)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert with_blips(501, 1)[1] == 1 and values[1] <= values[0]


def test_nothing_crosses_halftime():
    half = np.zeros(400, dtype=np.int8)
    half[0:200], half[200:400] = 1, 2
    a = np.zeros(400)
    a[190:200] = 1.0
    assert np.all(team.centred_mean(a, 6, half)[200:] == 0.0)
    sign = np.r_[np.zeros(190), np.ones(210)].astype(np.int8)
    _, ep = team.episodes(sign, 1000, half)
    assert len(ep["first"]) == 2


def test_cap_groups_is_two_sided_and_keeps_shape():
    llr = np.array([1.0, 1.0, 1.0, -0.5, -0.5, 2.0])
    out, n = team.cap_groups(llr, np.zeros(6, dtype=np.int64), 1.5)
    assert abs(out[out > 0].sum() - 1.5) < 1e-12
    assert abs(out[out < 0].sum() + 1.0) < 1e-12
    assert out[5] > out[0] and n == 1


def test_binomial_p_value():
    assert team.binomial_p_one_sided(0, 10) == 1.0
    assert abs(team.binomial_p_one_sided(10, 10) - 1 / 1024) < 1e-12
    assert team.binomial_p_one_sided(22, 40) > 0.05
    assert team.binomial_p_one_sided(27, 40) < 0.05


def test_distances_per_team_are_output():
    sc = Scene()
    f = frames(500, 600)
    sc.put(0, f, 50.0)
    sc.put(3, f, 52.0)
    sc.ball(f, 50.5)
    out, _ = sc.run()
    assert np.allclose(out["d_home"][f], 0.5, atol=1e-4)
    assert np.allclose(out["d_away"][f], 1.5, atol=1e-4)


def test_logistic_fits_are_finite_on_separable_data():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([-1, -1, 1, 1])
    w, b = team.fit_logistic(X, y)
    assert np.all(np.isfinite(w)) and np.isfinite(b) and w[0] > 0
    a = team.fit_scale(X[:, 0], y)
    assert np.isfinite(a) and a > 0


def test_attack_direction_per_half():
    sc = Scene(n=1000, halves=((0, 500), (500, 1000)))
    sc.px[0:500, :3], sc.px[0:500, 3:] = 30.0, 70.0
    sc.px[500:, :3], sc.px[500:, 3:] = 70.0, 30.0
    d, info = team.attack_direction(sc.px, sc.slot_team, sc.half)
    assert d[100] == 1.0 and d[700] == -1.0
    assert info[1][0] == 1.0 and info[2][0] == -1.0


def noball_scene(n_cycles=44, long_hole=0, informative=True, quiet_informative=True):
    CTRL, QUIET = 150, 120
    pattern = [(1, False), (1, True), (1, False), (1, True), (-1, True),
               (-1, False), (-1, True), (-1, False), (-1, True), (1, True)]
    lengths = [QUIET if q else CTRL for _, q in pattern]
    n_cycle = sum(lengths)
    n = 100 + n_cycle * n_cycles + long_hole + 300
    mid = 100 + n_cycle * (n_cycles // 2)
    sc = Scene(n=n, n_home=8, n_away=8, halves=((100, mid), (mid, n - 100)))
    rng = np.random.default_rng(0)
    base_x = np.r_[np.linspace(15, 45, 8), np.linspace(40, 75, 8)]
    base_y = np.r_[np.linspace(8, 60, 8), np.linspace(12, 56, 8)]
    drift = np.zeros(n)
    phase_team = np.zeros(n, dtype=np.int8)
    quiet = np.zeros(n, dtype=bool)
    speed = 2.0 if informative else 0.0
    x0 = 0.0
    for c in range(n_cycles):
        start = 100 + c * n_cycle
        for (tm, is_quiet), length in zip(pattern, lengths):
            seg = frames(start, start + length)
            start += length
            phase_team[seg], quiet[seg] = tm, is_quiet
            v = speed if (quiet_informative or not is_quiet) else 0.0
            drift[seg] = x0 + tm * v * (seg - seg[0]) / 25
            x0 = drift[seg[-1]]
    t0 = 100 + n_cycle * n_cycles
    hole = frames(t0, t0 + long_hole)
    if long_hole:
        phase_team[hole], quiet[hole] = 1, True
        drift[hole] = x0 + 0.5 * (hole - hole[0]) / 25
        x0 = drift[hole[-1]]
    tail = frames(t0 + long_hole, n - 100)
    phase_team[tail], drift[tail] = 1, x0
    live = sc.half > 0
    for s in range(16):
        sc.px[live, s] = base_x[s] + drift[live] + rng.normal(0, 0.2, live.sum())
        sc.py[live, s] = base_y[s] + rng.normal(0, 0.2, live.sum())
    ctrl = live & ~quiet
    holder = np.where(phase_team > 0, 3, 11)
    sc.bx[ctrl] = sc.px[ctrl, holder[ctrl]] + 0.4
    sc.by[ctrl] = sc.py[ctrl, holder[ctrl]]
    return sc, quiet, phase_team, hole


def test_noball_model_learns_who_is_advancing():
    sc, quiet, phase_team, _ = noball_scene()
    out, info = sc.run()
    live = sc.half > 0
    assert info["nb_enabled"], info["nb_reason"]
    assert info["nb_oof"] and set(info["nb_folds"]) == {1, 2}
    assert all(w[0] > 0 for w, _ in info["nb_folds"].values())
    assert out["llr_noball"][quiet & (phase_team > 0) & live].mean() > 0
    assert out["llr_noball"][quiet & (phase_team < 0) & live].mean() < 0
    assert info["nb_calibrated"] and 0 < info["nb_a"] <= 1
    assert info["nb_loc_n"] >= team.NB_LOC_MIN and info["nb_loc_agree"] > 0.9


def test_p_home_falls_back_on_the_noball_channel_where_the_ball_is_missing():
    sc, quiet, phase_team, _ = noball_scene()
    out, info = sc.run()
    assert info["nb_enabled"], info["nb_reason"]
    inner_h = quiet & (phase_team > 0) & (sc.half > 0) & (out["ball_rel"] == 0)
    inner_a = quiet & (phase_team < 0) & (sc.half > 0) & (out["ball_rel"] == 0)
    assert out["p_home"][inner_h].mean() > 0.5 > out["p_home"][inner_a].mean()


def test_uninformative_noball_channel_switches_itself_off():
    sc, *_ = noball_scene(informative=False)
    out, info = sc.run()
    assert not info["nb_enabled"]
    assert np.all(out["llr_noball"] == 0)


def test_noball_channel_cannot_pass_on_what_the_ball_already_said():
    sc, *_ = noball_scene(quiet_informative=False)
    out, info = sc.run()
    assert not info["nb_enabled"], info["nb_reason"]
    assert np.all(out["llr_noball"] == 0)


def test_noball_evidence_in_a_long_hole_is_bounded():
    sc, _, _, hole = noball_scene(long_hole=60 * 25)
    out, info = sc.run()
    assert info["nb_enabled"], info["nb_reason"]
    total = out["llr_noball"][hole].sum()
    assert 0 < total <= 1.0 + 1e-6


def test_weak_ball_brushes_do_not_cut_a_silent_stretch():
    sc, _, _, hole = noball_scene(long_hole=60 * 25)
    for k in range(3):
        f = frames(hole[0] + 300 * (k + 1), hole[0] + 300 * (k + 1) + 10)
        sc.ball(f, sc.px[f, 3] + 2.4, sc.py[f, 3])
    out, info = sc.run()
    assert info["nb_enabled"], info["nb_reason"]
    brush = np.abs(out["hold_s"][hole])
    assert brush.max() >= team.E_MIN and brush.max() < team.CLEAR_S
    assert out["llr_noball"][hole].sum() <= 1.0 + 1e-6


def test_switch_is_placed_where_the_other_team_takes_control():
    e = np.r_[np.full(20, 0.3), np.zeros(60), np.full(20, -0.3)]
    path = viterbi_two_state(e, 2.0)
    assert np.all(path[:80] == 1) and np.all(path[80:] == -1)


def test_alive_stretch_without_evidence_continues_the_last_possession():
    half = np.r_[np.zeros(10), np.ones(300), np.zeros(10)].astype(np.int8)
    dead = np.zeros(320, dtype=bool)
    dead[150:160] = True
    llr = np.zeros(320)
    llr[20:40] = -0.3
    labels = decode(llr, dead, half, 2.0)
    assert set(labels[10:150]) == {"A"}
    assert set(labels[160:310]) == {"A"}
    assert set(labels[150:160]) == {"D"}
    assert "X" not in set(labels[half > 0])


def test_leading_empty_stretch_takes_the_next_possession():
    half = np.r_[np.zeros(10), np.ones(300), np.zeros(10)].astype(np.int8)
    dead = np.zeros(320, dtype=bool)
    dead[150:160] = True
    llr = np.zeros(320)
    llr[200:220] = 0.3
    labels = decode(llr, dead, half, 2.0)
    assert set(labels[10:150]) == {"H"} and set(labels[160:310]) == {"H"}


def test_empty_stretches_never_borrow_across_halftime():
    half = np.r_[np.ones(100), np.zeros(20), 2 * np.ones(100)].astype(np.int8)
    llr = np.zeros(220)
    llr[10:30] = -0.3
    llr[150:170] = 0.3
    dead = np.zeros(220, dtype=bool)
    dead[125:135] = True
    labels = decode(llr, dead, half, 2.0)
    assert set(labels[120:125]) == {"H"}


def test_viterbi_switches_on_strong_evidence_only():
    e = np.r_[np.full(100, 0.1), np.full(100, -0.1)]
    path = viterbi_two_state(e, 2.0)
    assert np.all(path[:100] == 1) and np.all(path[100:] == -1)
    weak = np.r_[np.full(100, 0.1), np.full(100, -0.005)]
    assert np.all(viterbi_two_state(weak, 2.0) == 1)


def test_noball_channel_alone_never_switches_possession():
    sc, _, _, hole = noball_scene(long_hole=60 * 25)
    out, info = sc.run()
    assert info["nb_enabled"], info["nb_reason"]
    e = np.zeros(len(hole) + 200)
    e[:100] = -0.2
    e[100:100 + len(hole)] = out["llr_noball"][hole]
    assert np.all(viterbi_two_state(e, team.default_switch_cost(info["ep_llr"])) == -1)


def test_switch_cost_sits_between_what_may_and_may_not_switch():
    must = team.control_value(team.MIN_CONTROL)
    assert 0.6 < must < 0.9
    for ep_llr in (2.81, team.EP_LLR_PRIOR, team.EP_LLR_MAX):
        c = team.default_switch_cost(ep_llr)
        assert team.NB_SEGMENT_MAX < c < must * ep_llr
        assert team.WEAK_EP_FRAC * ep_llr < c


def interception(n_control):
    sc = Scene()
    f = frames(300, 500)
    sc.put(0, f, 40.0)
    sc.ball(f, 40.4)
    g = frames(500, 500 + n_control)
    sc.put(3, g, 40.8)
    sc.ball(g, 40.4)
    sc.put(0, g, 34.0)
    out, info = sc.run()
    labels = decode(out["llr_ball"] + out["llr_noball"], np.zeros(sc.n, dtype=bool), sc.half,
                    team.default_switch_cost(info["ep_llr"]))
    return labels, out, g


def test_half_a_second_of_control_switches_possession():
    labels, out, g = interception(team.MIN_CONTROL)
    assert labels[400] == "H" and labels[g[-1]] == "A"


def test_a_brief_touch_does_not_switch_possession():
    labels, out, g = interception(7)
    assert np.abs(out["hold_s"][g]).max() >= team.CLEAR_S
    assert set(labels[300:600]) == {"H"}


def test_weak_episode_cannot_switch_on_its_own():
    sc = Scene()
    f = frames(500, 750)
    x = 30 + 0.5 * (f - 500) / 25
    sc.put(3, f, x - 2.4)
    sc.ball(f, x)
    out, info = sc.run()
    assert np.abs(out["hold_s"][f]).max() < team.CLEAR_S
    total = out["llr_ball"][f].sum()
    assert -0.35 * info["ep_llr"] - 1e-6 <= total < 0


def test_reference_decoder_end_to_end():
    sc, quiet, phase_team, _ = noball_scene()
    out, info = sc.run()
    live = sc.half > 0
    labels = decode(out["llr_ball"] + out["llr_noball"], np.zeros(sc.n, dtype=bool), sc.half,
                    0.5 * info["ep_llr"])
    truth = np.where(phase_team > 0, "H", "A")
    ctrl = live & ~quiet
    assert np.mean(labels[ctrl] == truth[ctrl]) > 0.95
    assert np.mean(labels[live] == truth[live]) > 0.85


def test_outputs_are_valid():
    sc = Scene()
    f = frames(500, 600)
    sc.put(0, f, 40.0)
    sc.ball(f, 40.5)
    out, _ = sc.run()
    for k in ("p_home", "conf", "w_noball", "ball_rel"):
        assert np.all((out[k] >= 0) & (out[k] <= 1)), k
    assert np.all(np.isfinite(out["llr_ball"])) and np.all(np.isfinite(out["llr_noball"]))
    assert np.all(out["p_home"][sc.half == 0] == 0.5)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok    ", name)
        except Exception as exc:
            failed += 1
            print("FAIL  ", name, "--", type(exc).__name__, exc)
    print("\n{} passed, {} failed".format(len(tests) - failed, failed))
    sys.exit(1 if failed else 0)
