"""Reproducible group-aware decoder parameter search over training trips only."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeVar

from .config import DecoderConfig

T = TypeVar("T")


def grouped_folds(trip_ids: Sequence[str], n_splits: int = 5, seed: int = 42) -> list[tuple[list[str], list[str]]]:
    """Return deterministic trip-disjoint train/validation ID lists."""
    ids = sorted(set(trip_ids))
    if n_splits < 2 or len(ids) < n_splits:
        raise ValueError("n_splits must be at least two and no greater than the number of trips")
    random.Random(seed).shuffle(ids)
    folds = []
    for fold_index in range(n_splits):
        validation = ids[fold_index::n_splits]
        training = [trip_id for trip_id in ids if trip_id not in set(validation)]
        if set(training) & set(validation):
            raise AssertionError("trip leakage between grouped folds")
        folds.append((training, validation))
    return folds


def tune_decoder_group_cv(
    training_trip_payloads: Mapping[str, T],
    configurations: Sequence[DecoderConfig],
    evaluate_trip: Callable[[DecoderConfig, T], Mapping[str, float | int | None]],
    *,
    n_splits: int = 5,
    seed: int = 42,
    output_json: str | Path | None = None,
) -> dict[str, object]:
    """Choose parameters by grouped validation Top-1, then continuity, then log loss.

    `evaluate_trip` must run inference first and evaluate labels supplied separately from prediction.
    Pass training trips only; the function has no dataset loader and never opens route files.
    Every fold/configuration metric is retained in the returned and optional JSON record.
    """
    if not training_trip_payloads or not configurations:
        raise ValueError("provide training trip payloads and at least one decoder configuration")
    folds = grouped_folds(list(training_trip_payloads), n_splits, seed)
    trials = []
    for config_index, config in enumerate(configurations):
        fold_results = []
        for fold_index, (_training_ids, validation_ids) in enumerate(folds):
            values = [evaluate_trip(config, training_trip_payloads[trip_id]) for trip_id in validation_ids]
            top1 = _mean([m.get("hmm_top1_accuracy") for m in values])
            continuity = _mean([_continuity(m.get("hmm_route_continuity")) for m in values])
            log_loss = _mean([m.get("candidate_log_loss") for m in values])
            fold_results.append({"fold": fold_index, "validation_trip_ids": validation_ids,
                                 "hmm_top1_accuracy": top1, "hmm_route_continuity": continuity,
                                 "candidate_log_loss": log_loss})
        trials.append({"trial": config_index, "config": asdict(config), "folds": fold_results,
                       "mean_hmm_top1_accuracy": _mean([f["hmm_top1_accuracy"] for f in fold_results]),
                       "mean_hmm_route_continuity": _mean([f["hmm_route_continuity"] for f in fold_results]),
                       "mean_candidate_log_loss": _mean([f["candidate_log_loss"] for f in fold_results])})
    best = max(trials, key=lambda trial: (trial["mean_hmm_top1_accuracy"],
                                          trial["mean_hmm_route_continuity"],
                                          -trial["mean_candidate_log_loss"]))
    result = {"protocol": "grouped_kfold_training_trips_only", "n_splits": n_splits, "seed": seed,
              "trip_ids": sorted(training_trip_payloads), "selection_rule": "mean HMM Top-1, then route continuity, then lower candidate log loss",
              "best_config": best["config"], "trials": trials}
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _mean(values: Sequence[float | int | None]) -> float:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        raise ValueError("evaluate_trip must return finite validation metrics for every evaluated trip")
    if any(value != value or value in (float("inf"), float("-inf")) for value in finite):
        raise ValueError("evaluate_trip returned a non-finite validation metric")
    return sum(finite) / len(finite)


def _continuity(metric: object) -> float | None:
    if isinstance(metric, Mapping):
        value = metric.get("route_continuity")
        return float(value) if value is not None else None
    return None

