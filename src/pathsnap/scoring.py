"""Post-inference server-side scoring. Do not import this from prediction code."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .decoder.metrics import evaluate_predictions
from .io import load_trip
from .labels import derive_point_labels


def score_after_prediction(trip_dir: str | Path, prediction: dict, state: dict) -> dict:
    """Load reference labels only after prediction and calculate server-side metrics."""
    labeled_trip = load_trip(trip_dir, with_labels=True)
    feature_graph = state.get("feature_graph")
    if feature_graph is None:
        raise RuntimeError("inference did not construct its feature graph")
    reference_labels = derive_point_labels(labeled_trip, feature_graph)
    reference_labels = [None if int(value) < 0 else int(value) for value in reference_labels]

    candidate_sets = state.get("candidate_sets")
    probabilities = state.get("probabilities")
    graph = state.get("graph")
    if candidate_sets is None or probabilities is None or graph is None:
        raise RuntimeError("inference did not retain its private scoring context")
    flat = np.asarray(probabilities, dtype=float).tolist()
    nested = []
    offset = 0
    for candidates in candidate_sets:
        nested.append(flat[offset:offset + len(candidates)])
        offset += len(candidates)
    if offset != len(flat):
        raise ValueError("probability count does not match decoder candidates")

    return evaluate_predictions(
        candidate_sets=candidate_sets,
        probabilities=nested,
        classifier_arc_ids=prediction["classifier_arc_ids"],
        hmm_arc_ids=prediction["hmm_arc_ids"],
        reference_labels=reference_labels,
        predicted_route=prediction["route_arc_ids"],
        reference_route=labeled_trip.route.tolist(),
        graph=graph,
        timings=prediction["stage_timings_s"],
    )
