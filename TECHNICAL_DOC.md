# <TEAM NAME> — Technical document, More than a Hack 2027 Phase 1

## 1. Approach (P1 + P5)
Modular pipeline: a per-frame loader, three independent estimators (dead ball, ball
reconstruction, H vs A) and a Smart Tagging prior, combined by a temporal decoder.
_Why this structure rather than an end-to-end model:_ TODO

## 2. Assets used and dropped (P1)
| asset | used? | how |
|---|---|---|
| tactical ATD | yes | loader, frame_id - 1 |
| homography (tactical) | TODO | |
| Smart Tagging | yes | P5 prior, set pieces as restart anchors |
| broadcast video / ATD | TODO | |
| tactical video | TODO | |

## 3. Training data
TODO — none / list every labelled dataset, source, who labelled it, how much. None from BAR-ATH.

## 4. Components
### 4.1 Loader and alignment (P1)
### 4.2 Ball reconstruction (P3)
### 4.3 Dead-ball detector (P2)
### 4.4 H vs A model (P4)
### 4.5 Smart Tagging prior and decoder (P5)

## 5. Possession rules implemented
Loose balls, challenges, restarts: TODO (P4 + P5)

## 6. Validation without the private ground truth
Team diagnostics (possession count, mean length, D share, short-spell share), checked for
football plausibility. The public 5 minutes were used only to check alignment and file format.

## 7. Limitations and what we would do with more time
TODO (everyone, one paragraph per component)

## 8. Reproduction
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put the data bundle in ./data or set MTAH_DATA_DIR
python run_all.py --team <TEAM NAME>                   # -> submissions/<TEAM NAME>.csv
```
Deterministic: fixed seeds, pinned versions; the run prints the sha256 of the file.
