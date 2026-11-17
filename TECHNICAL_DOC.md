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

_TODO (P5): one paragraph on the final decoder and why it was chosen over alternatives._

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
_TODO_

### 4.4 Home vs away model (P4)
_TODO_

### 4.5 Smart Tagging prior and decoder (P5)
_TODO_

## 5. Possession rules implemented
_TODO (P4 + P5): loose balls, challenges, restarts — where the line was drawn and why._

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
- **Ball reconstruction:** long gaps remain weak linear fallbacks, and isolated-spike rejection
  does not detect sustained false tracks. Masked-block checks show error increasing strongly with
  gap duration; future work would test generalisation to naturally missing segments and assess
  downstream effects without tuning on the public possession labels.
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
