"""Load cached candidate / observation tables for development partitions."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .features import FEATURE_ORDER
from .io import _read
from .paths import CACHE_DIR, CONFIG_DIR, list_trips, trip_dir


def load_part(part: str, trips: list[str] | None = None):
    trips = trips or list_trips(part)
    cand = pd.concat([pd.read_parquet(CACHE_DIR / "candidates" / part / f"{t}.parquet") for t in trips], ignore_index=True)
    obs = pd.concat([pd.read_parquet(CACHE_DIR / "obs" / part / f"{t}.parquet") for t in trips], ignore_index=True)
    return cand, obs


def load_folds() -> list[list[str]]:
    return json.loads((CONFIG_DIR / "cv_folds.json").read_text())["folds"]


def stride_subsample(cand: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Keep every `stride`-th GPS point of each trip (all candidates of kept points)."""
    return cand[(cand["obs_idx"].to_numpy() % stride) == 0] if stride > 1 else cand


def xy(cand: pd.DataFrame):
    return cand[FEATURE_ORDER].to_numpy(np.float32), cand["label"].to_numpy(np.int8)


def trip_arcs(part: str, trip_id: str) -> np.ndarray:
    return _read(trip_dir(part, trip_id) / f"{trip_id}.arcs", np.int64)
