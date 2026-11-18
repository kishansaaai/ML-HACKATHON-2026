"""End-to-end prediction: Person A's candidates + classifier feeding Person B's HMM decoder.

    from pathsnap.pipeline import predict
    result = predict("Student folder/student_test/00000052")

`predict` reads only `.track`, `.nodes` and `.arcs` (never `.route`), so it is safe to call on
evaluator trips. Reference routes are compared against the result later, by the server-side
scoring layer, with `pathsnap.decoder.metrics.evaluate_predictions`.

One coordinate frame is shared by both halves: the classifier's features are computed in the
`TripGraph` frame (centred on the GPS track median), and the decoder graph is built with the
same origin through `InferenceComponents.projection_factory`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .candidates import CandidateConfig, build_candidates
from .clf import score_candidates
from .decoder.config import DecoderConfig
from .decoder.graph import Candidate, RoadGraph
from .decoder.inference import InferenceComponents, predict_trip
from .decoder.projection import LocalProjection
from .geo import TripGraph
from .io import load_trip


def make_components(trip_dir: str | Path, cand_cfg: CandidateConfig | None = None,
                    scorer: Callable[[pd.DataFrame], np.ndarray] | None = None) -> InferenceComponents:
    """Adapters for one trip. Labels are never loaded (`with_labels=False`).

    `scorer` maps a candidate table to one probability per row. Default: the exported model
    (`score_candidates`). Pass `pathsnap.oof.oof_scorer(trip_id)` ONLY to tune decoder settings on
    training trips with honest, out-of-fold scores.
    """
    trip = load_trip(trip_dir, with_labels=False)
    tgraph = TripGraph(trip)
    table = build_candidates(trip, tgraph, cand_cfg).sort_values("obs_idx", kind="stable").reset_index(drop=True)
    kept: list[int] = []        # table rows that became decoder candidates, in decoder order

    def projection_factory(node_lonlat, gps_lonlat) -> LocalProjection:
        return LocalProjection(tgraph.lon0, tgraph.lat0)

    def generate(points, graph: RoadGraph):
        if len(points) != trip.n_obs:
            raise ValueError(f"track length mismatch: decoder {len(points)} vs classifier {trip.n_obs}")
        sets: list[list[Candidate]] = [[] for _ in points]
        kept.clear()
        obs, arcs = table["obs_idx"].to_numpy(), table["arc_id"].to_numpy()
        frac, perp = table["frac"].to_numpy(), table["perp"].to_numpy()
        for row in range(len(table)):
            arc = graph.arcs.get(int(arcs[row]))
            if arc is None:                           # arc rejected by the decoder's validity checks
                continue
            f = float(frac[row])
            xy = (arc.start[0] + (arc.end[0] - arc.start[0]) * f, arc.start[1] + (arc.end[1] - arc.start[1]) * f)
            sets[int(obs[row])].append(Candidate(int(arcs[row]), f, float(perp[row]), xy))
            kept.append(row)
        return sets

    def build_features(observations, candidate_sets, graph):
        return table.iloc[kept]                       # same order as the candidate lists

    def score(features) -> np.ndarray:
        return (scorer or score_candidates)(features)

    return InferenceComponents(generate, build_features, score, projection_factory)


def predict(trip_dir: str | Path, decoder_config: DecoderConfig | None = None,
            cand_cfg: CandidateConfig | None = None,
            scorer: Callable[[pd.DataFrame], np.ndarray] | None = None) -> dict:
    """Classifier + HMM route prediction for one trip folder (no ground truth is read)."""
    return predict_trip(trip_dir, make_components(trip_dir, cand_cfg, scorer), decoder_config or DecoderConfig())
