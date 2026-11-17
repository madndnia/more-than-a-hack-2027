# Team contract

Model artifacts are `.npz` files in `data/derived/` (gitignored) with 1-D arrays of length
**199,319**, indexed by tactical video frame 0..199318 (= submission row). Read and write them only
through `mtah/contract.py` (`contract.save(...)` / `contract.load(...)`), which checks shape, dtype,
NaN and value ranges on both sides. A file that breaks the contract fails at save time, not three
steps later in somebody else's code.

| artifact | owner | module | fields |
|---|---|---|---|
| `frames.npz` | P1 | `mtah/loader.py` | per-frame table, see the loader docstring |
| `ball_clean.npz` | P3 | `mtah/ball.py` | `x`, `y` (metres), `observed` (bool), `conf` [0,1] |
| `p_dead.csv` | P2 | P2's generator | `frame,p_dead`, exactly 0..199318 in order |
| `p_dead.npz` | P5 adapter | `mtah/p2_adapter.py` | `p_dead` [0,1], `source_sha256` provenance |
| `p_team.npz` | P4 | `mtah/team.py` | `p_home`, `p_away` (sum to 1), `conf` [0,1] |
| `tag_prior.npz` | P5 | `mtah/tags.py` | `prior_home`, `prior_away`, `restart` [0,1] |
| `submissions/<team>.csv` | P5 | `mtah/decoder.py` | `frame,label`, H/A/D/X |

Extra fields are allowed and kept. P4 additionally writes `llr_ball`, `llr_noball` (one finite
value per frame) and scalar `ep_llr`. P5 requires both LLR channels and uses P4's
`default_switch_cost(ep_llr)`; if the scalar is absent it uses P4's `EP_LLR_PRIOR`.
There is no majority vote, second smoothing pass, or added duration filter. `dead_mask` is called
once; H/A decoding and empty-stretch handling are delegated unchanged to `mtah/ref_decoder.py`.

`tag_prior.npz` retains its three original fields and adds `llr_tag` plus audit arrays
`phase_start`, `phase_end`, `phase_frames`, `phase_budget`. Each phase contributes at most one nat
of absolute LLR, distributed over its frames inside the halves. Overlapping phases add their
separate budgets. Defence supports the opponent, N/A gives no team evidence, and `restart` is
metadata only. **The decoder does not load this artifact by default.** `--use-tag-prior` is an
experimental opt-in for use only after team agreement; the standalone prior builder does not
enable it.

## P5-only execution (ready P1/P4 artifacts required)

Obtain the existing `frames.npz`, P4's `p_team.npz`, and P2's `p_dead.csv` first. This command
adapts the CSV and writes a validated `submissions/submission.csv` without rebuilding any owner:

```bash
python -m mtah.decoder --p-dead-csv data/derived/p_dead.csv --output submissions/submission.csv
python evaluation/validate_submission.py submissions/submission.csv
```

The adapter can also run separately: `python -m mtah.p2_adapter --csv data/derived/p_dead.csv`.
It checks the exact CSV header, row count, integer frame sequence and finite [0,1] values before
writing via the NPZ contract, then verifies the round trip. It never changes the CSV. NPZ
probabilities use the contract's float32 precision; the source CSV SHA-256 is stored with them.
The same CSV is a no-op on a verified existing NPZ. A different or unprovenanced NPZ requires
the explicit `--replace-npz` flag. There is no sorting, clipping or missing-frame repair.

If `--p-dead-csv` is omitted, the decoder reads the existing `p_dead.npz` without writing it.
An existing submission is protected by default; `--overwrite` can replace only a valid existing
submission, never a P2 CSV or NPZ. New CSVs are validated before atomic publication. Missing
input artifacts cause an error, not an automatic loader or component build.

Optional prior preparation: `python -m mtah.tags` reads P1's `tags_events.csv` and the existing
half mask. It does not change the default decoder result.

`run_all.py` remains P1's full rebuild command and calls `mtah.dead.build()`, which replaces the
NPZ. Do not use it to consume an independently prepared P2 result. P5 uses the standalone
commands above instead.

Synthetic checks (no public GT or match assets):

```bash
python tests/test_team.py
python -m unittest discover -s tests -p test_p5.py -v
```

## Frame alignment (done once, in the loader)

| source | to video frame |
|---|---|
| ATD | `frame_id - 1` |
| Smart Tagging | `round(seconds * 25)` |
| homography | `frame` as is |

Graded frames: 11185..90651 and 110808..190197 (`F["graded"]`). Home = Barcelona = `H`.

## Workflow

- One branch per person: `p1-integration`, `p2-dead`, `p3-ball`, `p4-team`, `p5-decoder`.
- Only P1 merges into `main`, and only after `python run_all.py` still passes validation.
- One change per merge. Before/after on `python -m mtah.diagnostics`, **never** on the public GT.
- Only P1 runs `scripts/public_window_check.py`, and only as a bug check.
- No competition data in cloud LLMs or team chats.
