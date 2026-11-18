"""Out-of-fold (OOF) classifier scores for the 80 training trips.

Why: the exported model is trained on all 80 training trips, so it scores those trips better than
unseen ones (Top-1 about 94.5 % in-sample vs about 87.5 % in grouped CV). Tuning the HMM decoder on
in-sample scores would pick settings for over-confident emissions. OOF scores for each training trip
come from a model that never saw that trip (same LightGBM recipe, same 5 trip-grouped folds,
`configs/cv_folds.json`), so decoder tuning sees realistic scores.

Used ONLY to tune decoder parameters. The final system always uses the single exported model
(`artifacts/model.joblib`).

    python scripts/make_oof_scores.py          # builds data_cache/oof/train_oof.parquet (about 1 min)

    from pathsnap.oof import oof_scorer
    predict(trip_dir, scorer=oof_scorer(trip_id))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_folds, load_part, stride_subsample, xy
from .features import FEATURE_ORDER
from .models.registry import make_model, proba
from .paths import CACHE_DIR

OOF_PATH = CACHE_DIR / "oof" / "train_oof.parquet"
TRAIN_STRIDE = 3          # same as the exported model


def build_oof(stride: int = TRAIN_STRIDE) -> pd.DataFrame:
    cand, _ = load_part("train")
    parts = []
    for k, val_trips in enumerate(load_folds()):
        held = cand["trip_id"].astype(str).isin(set(val_trips)).to_numpy()
        X, y = xy(stride_subsample(cand[~held], stride))
        model = make_model("lightgbm")
        model.fit(X, y)
        va = cand[held]
        parts.append(pd.DataFrame({"trip_id": va["trip_id"].astype(str).to_numpy(),
                                   "obs_idx": va["obs_idx"].to_numpy(), "arc_id": va["arc_id"].to_numpy(),
                                   "p_oof": proba(model, va[FEATURE_ORDER].to_numpy(np.float32)).astype("float32"),
                                   "fold": k}))
    out = pd.concat(parts, ignore_index=True)
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OOF_PATH, index=False)
    return out


def oof_scorer(trip_id: str):
    """Scorer for `pathsnap.pipeline.predict(..., scorer=...)`: looks up the OOF score of each candidate row."""
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"{OOF_PATH} not found; run scripts/make_oof_scores.py first")
    df = pd.read_parquet(OOF_PATH)
    df = df[df["trip_id"] == str(trip_id)]
    if df.empty:
        raise KeyError(f"trip {trip_id} is not a training trip (no OOF scores)")

    def score(table: pd.DataFrame) -> np.ndarray:
        merged = table[["obs_idx", "arc_id"]].merge(df[["obs_idx", "arc_id", "p_oof"]], on=["obs_idx", "arc_id"], how="left")
        if merged["p_oof"].isna().any():
            raise ValueError("candidate rows without an OOF score: candidate configuration differs from the cached tables")
        return merged["p_oof"].to_numpy(np.float64)

    return score
