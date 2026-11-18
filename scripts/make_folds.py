"""Create configs/cv_folds.json: 5-fold GroupKFold over the 80 training trips (group = trip).

Trips are assigned greedily, largest first, to the currently lightest fold (balanced by number
of GPS points), after a seeded shuffle for tie-breaking. Fixed seed 42; shared by ALL models.
"""
import json

import numpy as np
import pandas as pd

from pathsnap.paths import CACHE_DIR, CONFIG_DIR, list_trips

SEED, K = 42, 5
sizes = {t: len(pd.read_parquet(CACHE_DIR / "obs" / "train" / f"{t}.parquet", columns=["obs_idx"]))
         for t in list_trips("train")}
rng = np.random.default_rng(SEED)
trips = list(sizes)
rng.shuffle(trips)
trips.sort(key=lambda t: -sizes[t])
load = [0] * K
folds = [[] for _ in range(K)]
for t in trips:
    k = int(np.argmin(load))
    folds[k].append(t)
    load[k] += sizes[t]
CONFIG_DIR.mkdir(exist_ok=True)
(CONFIG_DIR / "cv_folds.json").write_text(json.dumps(
    {"seed": SEED, "n_folds": K, "folds": [sorted(f) for f in folds]}, indent=1))
print("points per fold:", load, "trips per fold:", [len(f) for f in folds])
