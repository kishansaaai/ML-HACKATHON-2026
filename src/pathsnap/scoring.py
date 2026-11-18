"""Ground-truth scoring called only after pathsnap.pipeline.predict_trip completes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .decoder.metrics import evaluate_predictions, route_sequence_metrics
from .geo import TripGraph
from .io import load_trip
from .labels import derive_point_labels


def score_after_prediction(trip_dir: str | Path, prediction: dict, state: dict) -> dict:
    """Read `.route` after prediction and score the retained pipeline candidates."""
    labeled_trip = load_trip(trip_dir, with_labels=True)
    reference_labels = derive_point_labels(labeled_trip, TripGraph(labeled_trip))
    reference_labels = [None if int(value) < 0 else int(value) for value in reference_labels]
    candidate_sets = state.get("candidate_sets")
    flat_probabilities = state.get("probabilities")
    graph = state.get("graph")
    if candidate_sets is None or flat_probabilities is None or graph is None:
        raise RuntimeError("prediction did not retain its private evaluation context")
    probabilities = np.asarray(flat_probabilities, dtype=float).reshape(-1)
    nested: list[list[float]] = []
    offset = 0
    for candidates in candidate_sets:
        nested.append(probabilities[offset:offset + len(candidates)].tolist())
        offset += len(candidates)
    if offset != len(probabilities):
        raise ValueError("scorer output does not align with the pipeline candidates")
    metrics = evaluate_predictions(
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
    classifier_route = [int(value) for value in prediction["classifier_arc_ids"] if value is not None]
    metrics["classifier_route_metrics"] = route_sequence_metrics(classifier_route, labeled_trip.route.tolist())
    return metrics
