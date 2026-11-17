from pathlib import Path
import numpy as np
import pandas as pd

N_FRAMES = 199_319
FORCED_DEAD = (110_808, 113_741)   # inclusive, 2,934 frames, all D
DUMMY_VALUE = 0.3
FORCED_VALUE = 0.999

root = Path(__file__).resolve().parents[1]
out = root / "data" / "derived" / "p_dead.csv"
out.parent.mkdir(parents=True, exist_ok=True)

p = np.full(N_FRAMES, DUMMY_VALUE)
p[FORCED_DEAD[0]:FORCED_DEAD[1] + 1] = FORCED_VALUE
p = np.clip(p, 0.001, 0.999)

df = pd.DataFrame({"frame": np.arange(N_FRAMES), "p_dead": p})
df.to_csv(out, index=False, float_format="%.6f")

print(f"wrote {out}  rows={len(df)}  "
      f"frames {df.frame.iloc[0]}..{df.frame.iloc[-1]}  "
      f"min={df.p_dead.min():.3f} max={df.p_dead.max():.3f}")