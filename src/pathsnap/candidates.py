"""Candidate generation + feature table for one trip (no labels involved)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd

from .features import compute_features
from .geo import TripGraph
from .io import Trip
from .paths import CONFIG_DIR


@dataclass
class CandidateConfig:
    radius_m: float = 50.0       # arcs within this distance of a GPS point are candidates
    k_min: int = 6               # ...and at least this many nearest arcs are always offered
    max_cand: int = 24           # cap per point (nearest first) to bound table size
    fallback_max_m: float = 400.0

    @classmethod
    def load(cls) -> "CandidateConfig":
        p = CONFIG_DIR / "candidate_config.json"
        return cls(**json.loads(p.read_text())) if p.exists() else cls()

    def save(self) -> None:
        (CONFIG_DIR / "candidate_config.json").write_text(json.dumps(asdict(self), indent=2))


def build_candidates(trip: Trip, graph: TripGraph, cfg: CandidateConfig | None = None) -> pd.DataFrame:
    """One row per (obs_idx, arc_id) with the features in FEATURE_ORDER. Uses no ground truth."""
    cfg = cfg or CandidateConfig.load()
    obs, arc, perp, frac = graph.query_candidates(graph.obs_xy, cfg.radius_m, cfg.k_min,
                                                  cfg.max_cand, cfg.fallback_max_m)
    feats = compute_features(graph, obs, arc, perp, frac)
    out = pd.DataFrame({"trip_id": trip.trip_id, "obs_idx": obs.astype("int32"), "arc_id": arc.astype("int32")})
    return pd.concat([out, feats], axis=1)

