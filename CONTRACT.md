# Team contract

Every shared artifact is a `.npz` in `data/derived/` (gitignored) with 1-D arrays of length
**199,319**, indexed by tactical video frame 0..199318 (= submission row). Read and write them only
through `mtah/contract.py` (`contract.save(...)` / `contract.load(...)`), which checks shape, dtype,
NaN and value ranges on both sides. A file that breaks the contract fails at save time, not three
steps later in somebody else's code.

| artifact | owner | module | fields |
|---|---|---|---|
| `frames.npz` | P1 | `mtah/loader.py` | per-frame table, see the loader docstring |
| `ball_clean.npz` | P3 | `mtah/ball.py` | `x`, `y` (metres), `observed` (bool), `conf` [0,1] |
| `p_dead.npz` | P2 | `mtah/dead.py` | `p_dead` [0,1] |
| `p_team.npz` | P4 | `mtah/team.py` | `p_home`, `p_away` (sum to 1), `conf` [0,1] |
| `tag_prior.npz` | P5 | `mtah/tags.py` | `prior_home`, `prior_away`, `restart` [0,1] |
| `submissions/<team>.csv` | P5 | `mtah/decoder.py` | `frame,label`, H/A/D/X |

Every module already contains a working **dummy v0**, so `python run_all.py` produces a valid file
from the first hour. Replace the body of your own `build()` / `decode()`, keep the output fields.
Extra fields are allowed and kept.

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
