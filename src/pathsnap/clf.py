"""Inference-time scoring of candidate arcs. Loads ONLY `artifacts/` (no data, no labels).

    score_candidates(df) -> P(candidate is the reference arc), one float per row
    normalise_per_obs(df, p) -> same probabilities renormalised to sum to 1 within each GPS point
                                (useful as HMM emission probabilities)

`df` is a candidate table from `candidates.build_candidates`; its feature columns are selected
and ordered by `artifacts/feature_order.json`, the order used in training.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .paths import ARTIFACT_DIR


@lru_cache(maxsize=1)
def _load():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    order = json.loads((ARTIFACT_DIR / "feature_order.json").read_text())
    return model, order


def score_candidates(df: pd.DataFrame) -> np.ndarray:
    model, order = _load()
    missing = [c for c in order if c not in df.columns]
    if missing:
        raise KeyError(f"candidate table lacks features: {missing}")
    if len(df) == 0:
        return np.zeros(0)
    return model.predict_proba(df[order].to_numpy(np.float32))[:, 1]


def normalise_per_obs(df: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """Divide each score by the sum of scores of the same (trip, obs). Rows must be grouped by obs."""
    key = df["trip_id"].astype(str) + "_" + df["obs_idx"].astype(str)
    total = pd.Series(p).groupby(key.to_numpy()).transform("sum").to_numpy()
    return p / np.maximum(total, 1e-12)
