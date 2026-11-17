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
| tactical homography | loaded, _TODO: used?_ | validity and reprojection error are in the per-frame table |
| tactical video | no | not needed for the final pipeline; _TODO if anyone used it for debugging_ |
| broadcast video / ATD / homography | no | not synchronised to the tactical timeline and only 65.8% usable homography; not worth the alignment effort within the time available |

## 3. Training data
_TODO: "None — no labelled data was used; every component is rule-based or unsupervised." If any
team member used labelled data from OTHER games, list its source, who labelled it and how much._
No labels from BAR-ATH were used for training, tuning or validation.

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
_TODO_

### 4.3 Dead-ball detector (P2)
_TODO_

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
| final | _TODO_ | | | | |

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
  guarantee of durability after every possible storage or power failure. Real-match accuracy and
  reproduction on the pinned dependencies still require P1's integration.
- _TODO: one paragraph per component (everyone)._

## 8. Reproduction
```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # numpy 2.5.5, pandas 3.1.1 (pinned)
# unpack the data bundle into ./data (data/tactical/...) or set MTAH_DATA_DIR
python run_all.py --team ronaldo2008   # -> submissions/ronaldo2008.csv, validated
```
The run is deterministic (fixed seeds, no randomness in the pipeline) and prints the sha256 of the
output file. A fresh clone with a fresh environment reproduces the identical file; this is checked
by `scripts/check_reproducible.sh`. The match data is not in the repository (`.gitignore`), in
line with the terms of use.

For P5 integration with an existing P2 CSV, use the standalone command in `CONTRACT.md`, not the
full rebuild above. Run P4's synthetic suite with `python tests/test_team.py` and P5's with
`python -m unittest discover -s tests -p test_p5.py -v`. P5 tests write only synthetic temporary
artifacts and do not read the public ground truth or replace a competition-data artifact.
