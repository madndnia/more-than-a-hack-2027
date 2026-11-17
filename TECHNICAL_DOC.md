# ronaldo2008 — Technical document, More than a Hack 2027 Phase 1

Repository: https://github.com/madndnia/more-than-a-hack-2027

## 1. Approach
A modular pipeline. One shared loader turns the provided assets into a single per-frame table;
three independent estimators (dead ball, ball reconstruction, home vs away) and a Smart Tagging
prior each write one per-frame probability file; a temporal decoder turns those into labels.

We chose this structure for two reasons. First, the metric (spellF1) rewards whole, correctly
bounded possessions rather than per-frame accuracy, so the decisive step is a temporal decoder that
can enforce realistic possession lengths — which is easiest when its inputs are clean probabilities
rather than hard labels. Second, it let five people work in parallel: every component has a fixed
output contract (`CONTRACT.md`, enforced by `mtah/contract.py`), so nobody depended on anybody
else's code, only on their output file.

P5 reuses P4's reference H/A Viterbi on the sum of its two LLR channels, with P4's default switch
cost and dead-ball mask. This preserves P4's episode evidence budget and avoids a second smoothing
or duration model. It supersedes the earlier three-state decoder proposal. Smart Tagging is a
separately built experimental prior. It is disabled by default, and explicit attempts to enable
an unverified external prior are also rejected in the current P5 implementation.

## 2. Assets used and dropped
| asset | used? | how |
|---|---|---|
| tactical ATD (players + ball) | yes | main input; read by `mtah/loader.py`, positions converted to metres on a 105.3 × 68.0 m pitch |
| tactical ATD metadata | yes | team of each of the 46 track slots (first team = home) |
| Smart Tagging | yes | team evidence from phase tags, set pieces as restart anchors (P5) |
| tactical homography | loaded, not used | validity and reprojection error are in the per-frame table; no component needed pixel coordinates |
| tactical video | no | the pipeline reads tracking and tagging only |
| broadcast video / ATD / homography | no | not synchronised to the tactical timeline and only 65.8% usable homography; not worth the alignment effort within the time available |

## 3. Training data
None. No labelled data from any game was used. Every component is rule-based with football-sense
defaults; the only fitted model (P4's no-ball logistic regression) is trained on pseudo-labels
produced by P4's own ball channel, decided by an algorithm and not by a person. No labels from
BAR-ATH were used for training, tuning or validation, and no component reads the public ground
truth.

## 4. Components

### 4.1 Loader and frame alignment
`mtah/loader.py` streams the raw ATD once (about 11 s) and writes `data/derived/frames.npz`, one
row per tactical video frame (199,319). All alignment lives in this one place:

| source | conversion to video frame |
|---|---|
| ATD | `frame_id - 1` |
| Smart Tagging | `round(seconds × 25)` |
| homography | `frame` as delivered |

The table holds ball and player positions in metres (NaN where the tracker lost them), team of
each slot, tracked-player counts per team, homography validity and error, the graded-half mask,
and per-frame Smart Tagging indicators (active home / away / N/A phases, set piece and its type).

**Verification.** The loader's own summary reproduces every figure published in DATA.md: ball
observed on 56.9% of all frames and 59.3% of graded frames, a mean of 20.5 tracked players per
graded frame, homography valid on 100% of frames, 158,857 graded frames in two halves of 79,467
and 79,390. Re-running `starter/baseline.py` reproduces the published baseline scores exactly
(baseline_smooth50: macroF1 0.6207, spellF1 0.3256 on the public window), which confirms the data
bundle and the alignment before any of our own logic is involved.

### 4.2 Ball reconstruction (P3)
`mtah/ball.py` reconstructs the two ball coordinates jointly and independently within each
contiguous half. Finite observed positions are retained without pitch clipping or general
smoothing, preserving out-of-play evidence and abrupt direction changes. An isolated observation
is excluded as an anchor only when both adjacent frames are observed, both incoming and outgoing
speeds exceed 55 m/s, the two neighbours imply less than 20 m/s, and the point deviates more than
3 m from their midpoint. Missing positions are linearly interpolated between accepted anchors.
Gaps up to 1 s receive a confidence weight of `0.8 * exp(-gap_seconds)`; longer gaps retain a
linear fallback with weight `0.1 * exp(-(gap_seconds - 1) / 5)`. Implied endpoint speeds above
55 m/s further reduce this weight. Boundary positions are held for at most 0.5 s with low,
decaying confidence; otherwise coordinates remain NaN with zero confidence. The original
`observed` mask is preserved, and an extra `rejected` mask records excluded observations.

These weights and thresholds are engineering defaults, not calibrated probabilities or parameters
tuned on possession ground truth. Synthetic tests cover interpolation, isolated spikes, fast
passes, direction changes, invalid coordinates, half boundaries, deterministic output and the
full-size artifact contract. On the provided match, the component completed in approximately
0.2 s; the conservative spike filter rejected no observed points. The complete pipeline produced
a valid 199,319-row CSV, and a second run reproduced the identical SHA256. This verifies
integration and reproducibility, not reconstruction accuracy or a scoring improvement.
Long-gap paths cannot recover unseen passes or touches;
multi-frame tracking errors and plausible-looking false detections can survive the conservative
filter. Downstream consumers must use confidence to avoid treating long-gap fallbacks as reliable
positions; the current scaffold decoder does not yet use the propagated team confidence.

**Independent reconstruction check.** `python scripts/check_ball_quality.py` hides both coordinates
and the observed flag for 200 nonoverlapping blocks per duration, sampled with seed 0 from fully
observed stretches inside a half. Both visible endpoints are reserved. Each duration is evaluated
as a separate experiment, without possession labels or parameter changes based on the results.
All masked frames were reconstructed. Euclidean errors against the hidden tracker observations:

| hidden duration (s) | mean error (m) | median error (m) | 95th percentile (m) |
|---|---|---|---|
| 0.20 | 0.050 | 0.036 | 0.146 |
| 0.48 | 0.211 | 0.130 | 0.663 |
| 1.00 | 0.714 | 0.431 | 2.128 |
| 2.00 | 1.778 | 1.101 | 5.547 |
| 5.00 | 5.915 | 4.695 | 15.339 |
| 10.00 | 11.004 | 9.135 | 27.269 |

The original linear interpolation has identical position errors on these blocks; no position
accuracy improvement is claimed. The new component adds gap-dependent confidence, conservative
spike rejection and bounded edge holds. Fully observed stretches can be easier than naturally
missing segments, and the reference tracker positions are automatic, not corrected ground truth.
These results do not validate possession accuracy or confidence calibration.

### 4.3 Dead-ball detector (P2)
`src/make_p_dead.py` writes `data/derived/p_dead.csv`, P(ball is dead) per video frame, built in
log-odds: a football prior plus evidence terms, then a sigmoid. It reads the raw ATD and Smart
Tagging directly.

| evidence | rule | log-odds |
|---|---|---|
| prior | dead ball is a minority of graded time, P = 0.30 | logit(0.30) |
| out of bounds | ball observed more than 0.5 m beyond a touchline or goal line; the state is carried through missing frames for up to 20 s, because a ball that left the pitch is rarely tracked again before the restart | +4.0 |
| stationary | ball stays inside 1.5 m for about 2 s (51-frame centred window, gaps up to 1 s bridged) | +2.0 |
| moving | ball speed over ±0.24 s above 2 m/s / above 4 m/s | −1.0 / −2.0 |
| set piece | inside a Smart Tagging SET PIECES instance, and a ramp over the 6 s before it | up to +1.0 |
| team speed | mean speed of all tracked players over ~2 s; walking pace pushes towards dead | ±0.8 |
| ball missing | **no** dead-ball evidence: occlusion happens in open play all the time | 0 |

Frames 110,808–113,741 (graded, before the second-half kickoff, all D according to FAQ.md) are
set to 0.999; ungraded frames get 0.5; everything is clipped to [0.001, 0.999]. Every weight is a
football-sense default chosen from the plausibility diagnostics the script prints (number and
length of dead stretches per half), never by scoring against the public window. The decoder turns
the probabilities into D with P4's dead mask: `p_dead ≥ 0.5`, then a centred majority over a short
window inside each half. Compared with the scaffold, which called every frame without a tracked
ball dead, the D share of graded frames fell from 41.6% to 18.3%, a plausible share of stoppage time for a
professional match.

### 4.4 Home vs away model (P4)
`mtah/team.py` writes `p_team.npz`: for every frame, how strongly the tracking says that home or
away controls the ball, as log-likelihood ratios a temporal decoder can add up. **Ball channel.**
P(player *i* is at the ball) = softmax(−d<sub>i</sub> / 0.5 m) over tracked players, summed per
team; this one quantity covers the nearest distance of each team, the number of players near the
ball and duels (equal distances give no evidence). It counts only when the ball is slow (full below
7 m/s, none above 12 m/s), a tracked player is within 1.5–3 m and the ball was observed (or
interpolated less than 0.5 s from an observation), and it is averaged over ±0.5 s, so a touch or a
pass flying past an opponent is not control. Tracks lost next to the ball can only cancel evidence,
never create it. The evidence of one control episode is capped at ln((1−e)/e), where e is the
share of episodes that look like errors (an isolated short episode between two of the other team,
Beta(1, 19) prior): e = 5.6% and 2.82 nats on this match. **No-ball channel.** Block velocity
towards the opponent's goal and the difference in team spread, combined by a logistic regression
fitted on pseudo-labels from the ball channel (no ground truth), out-of-fold by half, with a
temperature calibrated on whole ball-less stretches and at most 1 nat per stretch. It is kept only
if, inside ball-less stretches where possession changed, its evidence moves towards the new
possessor more often than chance; on this match it does (73.9% of 46 stretches, p = 0.001), so it
is on. `p_home` adds both channels for a decoder that reads one value per frame;
`mtah/ref_decoder.py` is a reference Viterbi on the summed LLRs (switch cost between what may not
and what must switch possession: 1.54 nats here) with 36 synthetic tests in `tests/test_team.py`.
On the full match it gives 210 possessions, mean 18.0 s, 11% shorter than 2 s.

**Where it does not work.** The ball is missing on 38.7% of graded frames and clear control is
visible on only 26.5%; possession changes inside those gaps rest on the no-ball channel, which is
weak by design (49 nats of evidence over the match against 1,582 from the ball) and can only place
a change, never make one alone. The reference decoder therefore finds about 76 changes in open
play, which is probably too few. The tracking has no ball height, so an aerial ball dropping slowly
next to a player looks like control. All thresholds are physical priors, not fitted; the result is
not very sensitive to the switch cost (1.91 → 1.54 nats moved the count from 198 to 210
possessions), which points to missing ball evidence rather than the decoder as the limit. The third
raw ball field (`ball_c3`) is not speed (correlation 0.06) and was left unused.

### 4.5 Smart Tagging prior and decoder (P5)
`mtah/decoder.py` reads existing artifacts and delegates to `mtah/ref_decoder.py` with
`llr_ball + llr_noball`, `team.default_switch_cost(ep_llr)` and a single `dead_mask` call. It does
not reuse `p_home`, resmooth labels or impose minimum durations. Missing LLR channels, nonfinite
values and malformed half masks fail explicitly. Outside the two half blocks it writes X. A
zero-evidence alive stretch follows P4's same-half neighbour rule; if the whole half has no
evidence, it remains X and is reported as unresolved rather than borrowing the other half's team.

P2 supplies `frame,p_dead` CSV whereas the model contract reads NPZ. `mtah/p2_adapter.py` checks
exact frame alignment and finite [0,1] probabilities without sorting, filling or clipping. It
preserves the CSV and stores its SHA-256 with the float32 contract artifact. P5's local
`mtah/p5_io.py` uses the shared contract's dtype rules and checks without changing
`mtah/contract.py`: it writes a temporary NPZ in the destination directory, flushes and fsyncs it,
then reopens it and verifies every field, dtype and value (including provenance) before publication.
New artifacts use an exclusive hard link so a concurrent writer's output cannot be overwritten;
explicit replacement uses `os.replace`. A failed write, verification or publication leaves an
existing artifact intact and removes the temporary file. The same writer is used for P5's own
`tag_prior.npz`. A different or unprovenanced adapter NPZ still requires `--replace-npz`; no actual
P2 artifact is replaced during synthetic checks. The P5 entry point never calls `dead.build()` or
the full pipeline.

`mtah/tags.py` reads P1's normalized events CSV, retaining phase codes instead of using flattened
home/away phase counts. Attacking phases support their named team; defending phases support the
opponent. Each phase has at most one nat of absolute LLR distributed over its active half frames,
regardless of duration. Overlapping phases retain separate budgets and may cancel. N/A and unknown
phase codes give no team evidence. Set pieces populate restart metadata without forcing a team or
dead-ball interval. `llr_tag` and per-phase budget audit arrays accompany the compatible probability
fields. **`tag_prior` is OFF by default and is not read by the submission decoder.**

The one-nat-per-phase bound is a property of this generator and its synthetic checks. For an
arbitrary supplied NPZ, finite `llr_tag` values and self-reported phase-budget arrays do not prove
that each original phase respected the bound: their provenance and consistency with source events
have not been independently verified. Therefore both `use_tag_prior=True` and the CLI's retained
`--use-tag-prior` flag fail explicitly before input adaptation, and `decode_arrays` rejects a
supplied `tag_llr`. This supersedes earlier descriptions of an available experimental opt-in.
Enabling tags requires separate team agreement and a verified integration; it is not achieved by
supplying a file or setting the old flag.

The output writer requires 199,319 exact H/A/D/X labels, emits canonical `frame,label` CSV, applies
the official format checker and publishes the validated file atomically. Existing outputs need
explicit overwrite and must themselves be valid submissions. Synthetic checks cover reference
equivalence, half boundaries, unresolved stretches, adapter rejection/preservation, phase budgets,
and a deterministic full-length CLI run checked by `evaluation/validate_submission.py`. These are
format and behaviour checks, not a claim of accuracy on this match. No public GT was used by P5.

## 5. Possession rules implemented
P4 supplies control evidence for loose balls and challenges; P5 retains its temporal decision rule
without adding a second touch/duration filter. D is determined by P4's dead-mask operation on P2
probabilities. Smart Tagging restarts remain metadata while the experimental prior is disabled.

## 6. Validation without the private ground truth
The public five minutes contain only 17 possessions, which cannot rank two versions of a system
(EVALUATION.md), so we did not tune on them. They were used once per integration, by one team
member, only to catch bugs — misalignment or swapped teams.

Every change was instead compared with team-wide diagnostics computed on the full match without
any ground truth (`python -m mtah.diagnostics`): number of possessions, mean possession length,
share of dead-ball frames, home share of alive frames, and share of possessions shorter than 2 s.
Each value was checked for plausibility against football knowledge before a change was merged.

| version | possessions | mean length | D share | home share | < 2 s |
|---|---|---|---|---|---|
| v1 (dummy components) | 441 | 8.4 s | 41.6% | 0.59 | 16.8% |
| final (submitted) | 245 | 21.2 s | 18.3% | 0.53 | 8.6% |

## 7. Limitations and what we would do with more time
- **Loader:** the broadcast recording is not used; mapping it onto the tactical timeline could
  recover ball positions where the tactical tracker loses the ball.
- **Home vs away (P4):** better ball reconstruction would help most, since possession changes
  are missed where the ball is lost. With labelled tracking from other matches the hand-set
  thresholds and the no-ball model could be fitted and validated properly instead of on
  pseudo-labels.
- **Decoder and tags (P5):** the default dead-ball decisions depend entirely on P2 and P4's mask;
  this component does not improve a constant P2 input. An entirely evidence-free half stays
  unresolved (X). Automatic tag phases have uncertain boundaries and correlated overlaps, so even
  the generator's one-nat-per-phase cap is not evidence that enabling tags improves possession
  recovery, nor a guarantee for arbitrary externally supplied priors. Tags remain disabled.
  Exclusive NPZ publication needs filesystem hard-link support; unsupported filesystems fail
  safely rather than falling back to a partial or overwriting write. Atomic publication is not a
  guarantee of durability after every possible storage or power failure. 
- **Ball reconstruction:** long gaps remain weak linear fallbacks, and isolated-spike rejection
  does not detect sustained false tracks. Masked-block checks show error increasing strongly with
  gap duration; future work would test generalisation to naturally missing segments and assess
  downstream effects without tuning on the public possession labels.
- **Dead ball (P2):** all weights are hand-set priors; a ball that stops while still in play
  (a goalkeeper holding it, a slow build-up) can be called dead, and a quick restart that the
  tracker never loses can be missed. Labelled tracking from other matches would let the weights be
  fitted instead of chosen.
- **Overall:** the final file averages 21.2 s per possession against 14.2 s published in the README for the
  public window, which suggests some possession changes are merged rather than invented. We did not adjust for
  this, because the only way to do so in time was to tune on the public labels.

## 8. Reproduction
```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # numpy 2.5.5, pandas 3.1.1 (pinned)
# unpack the data bundle into ./data (data/tactical/...) or set MTAH_DATA_DIR
python run_all.py --team ronaldo2008   # -> submissions/ronaldo2008.csv, validated
```
The run is deterministic (fixed seeds, no randomness in the pipeline) and prints the sha256 of the
output file. We verified it by cloning the public repository into an empty folder, creating a new
virtual environment from `requirements.txt` and running the command above: the file was
byte-identical, sha256 `3a783a500a6a828fdc42271c5df6dcdfdc1b289375b0fd43b04d3f1da8cfb528`.
`scripts/check_reproducible.sh` automates the same check. The match data is not in the repository (`.gitignore`), in
line with the terms of use.

`run_all.py` runs, in order: loader (P1), ball (P3), `src/make_p_dead.py` plus the CSV adapter
(P2), team evidence (P4), tag prior (P5, built but not used) and the decoder (P5). Run P4's synthetic suite with `python tests/test_team.py` and P5's with
`python -m unittest discover -s tests -p test_p5.py -v`. P5 tests write only synthetic temporary
artifacts and do not read the public ground truth or replace a competition-data artifact.
