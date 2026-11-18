"""Bridge Person A's candidate features/scorer to Person B's decoder contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import CandidateConfig
from .clf import score_candidates
from .features import compute_features
from .geo import TripGraph
from .io import load_trip
from .decoder import Candidate
from .decoder.inference import InferenceComponents


def make_inference_components(trip_dir: str | Path):
    """Return decoder callbacks plus private state for post-inference scoring.

    B's RoadGraph owns projection and candidate geometries. The feature graph is
    constructed in that same projection origin but does not build a redundant
    spatial index. The private state is never included in the API response.
    """
    trip_dir = Path(trip_dir)
    trip = load_trip(trip_dir, with_labels=False)
    config = CandidateConfig.load()
    state: dict[str, object] = {"trip": trip, "candidate_sets": None, "probabilities": None,
                                "graph": None, "feature_graph": None}

    def feature_graph_for(road_graph):
        feature_graph = state["feature_graph"]
        if feature_graph is None:
            feature_graph = TripGraph(
                trip,
                origin=(road_graph.projection.lon0, road_graph.projection.lat0),
                build_index=False,
            )
            state["feature_graph"] = feature_graph
        return feature_graph

    def generate_candidates(projected_points, road_graph):
        feature_graph_for(road_graph)
        candidate_sets: list[list[Candidate]] = []
        for point in projected_points:
            if point is None:
                candidate_sets.append([])
                continue
            radius = config.radius_m
            candidates = road_graph.spatial_index.query(point, radius)
            while len(candidates) < config.k_min and radius < config.fallback_max_m:
                radius = min(radius * 2.0, config.fallback_max_m)
                candidates = road_graph.spatial_index.query(point, radius)
            candidates.sort(key=lambda c: (c.distance_m, c.arc_id))
            candidate_sets.append(candidates[:config.max_cand])
        state["candidate_sets"] = candidate_sets
        state["graph"] = road_graph
        return candidate_sets

    def build_features(observations, candidate_sets, road_graph):
        feature_graph = feature_graph_for(road_graph)
        obs_indices, arc_ids = [], []
        for observation_index, candidates in enumerate(candidate_sets):
            for candidate in candidates:
                obs_indices.append(observation_index)
                arc_ids.append(candidate.arc_id)
        obs = np.asarray(obs_indices, dtype=np.int64)
        arcs = np.asarray(arc_ids, dtype=np.int64)
        if len(obs) == 0:
            frame = pd.DataFrame(columns=["trip_id", "obs_idx", "arc_id"])
            return frame
        perp, frac, _ = feature_graph.project_on_arcs(feature_graph.obs_xy[obs], arcs)
        features = compute_features(feature_graph, obs, arcs, perp, frac)
        frame = pd.DataFrame({
            "trip_id": trip.trip_id,
            "obs_idx": obs.astype(np.int32),
            "arc_id": arcs.astype(np.int32),
        })
        return pd.concat([frame, features], axis=1)

    def score(df):
        probabilities = score_candidates(df)
        state["probabilities"] = probabilities
        return probabilities

    return InferenceComponents(generate_candidates, build_features, score), state
