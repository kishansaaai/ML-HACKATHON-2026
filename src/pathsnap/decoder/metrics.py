"""Ground-truth evaluation helpers; deliberately separate from prediction imports."""

from __future__ import annotations

import math
from typing import Sequence

from .graph import Candidate, RoadGraph


def candidate_metrics(
    candidate_sets: Sequence[Sequence[Candidate]],
    probability_sets: Sequence[Sequence[float]],
    reference_arc_ids: Sequence[int | None],
) -> dict[str, object]:
    """Binary candidate classification scores, with one positive arc per labeled observation."""
    if not (len(candidate_sets) == len(probability_sets) == len(reference_arc_ids)):
        raise ValueError("candidate sets, scores, and labels must align by observation")
    y_true: list[int] = []
    y_pred: list[int] = []
    p_true: list[float] = []
    misses = labeled = 0
    for candidates, probs, truth in zip(candidate_sets, probability_sets, reference_arc_ids):
        if truth is None:
            continue
        if len(candidates) != len(probs):
            raise ValueError("candidate and score counts differ for an observation")
        labeled += 1
        if not candidates:
            misses += 1
            continue
        values = [max(0.0, float(p)) for p in probs]
        total = sum(values)
        values = [p / total for p in values] if total > 0 else [1 / len(values)] * len(values)
        ids = [candidate.arc_id for candidate in candidates]
        misses += int(truth not in ids)
        for arc_id, probability in zip(ids, values):
            positive = int(arc_id == truth)
            y_true.append(positive)
            y_pred.append(int(probability >= 0.5))
            p_true.append(probability)
    tp = sum(a == b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    tn = sum(a == b == 0 for a, b in zip(y_true, y_pred))
    # Macro average over binary classes with zero-division defined as zero.
    precision_pos = _ratio(tp, tp + fp)
    recall_pos = _ratio(tp, tp + fn)
    f1_pos = _ratio(2 * precision_pos * recall_pos, precision_pos + recall_pos)
    precision_neg = _ratio(tn, tn + fn)
    recall_neg = _ratio(tn, tn + fp)
    f1_neg = _ratio(2 * precision_neg * recall_neg, precision_neg + recall_neg)
    log_loss = 0.0
    if y_true:
        log_loss = -sum(math.log(max(1e-12, min(1 - 1e-12, p if y else 1 - p))) for y, p in zip(y_true, p_true)) / len(y_true)
    return {
        "candidate_miss_rate": _ratio(misses, labeled),
        "candidate_log_loss": log_loss if y_true else None,
        "candidate_macro_precision": (precision_pos + precision_neg) / 2,
        "candidate_macro_recall": (recall_pos + recall_neg) / 2,
        "candidate_macro_f1": (f1_pos + f1_neg) / 2,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "candidate_classification_count": len(y_true),
        "labeled_observation_count": labeled,
    }


def observation_top1(predicted_arc_ids: Sequence[int | None], reference_arc_ids: Sequence[int | None]) -> float | None:
    if len(predicted_arc_ids) != len(reference_arc_ids):
        raise ValueError("prediction and reference labels must align")
    valid = [(p, y) for p, y in zip(predicted_arc_ids, reference_arc_ids) if y is not None]
    return _ratio(sum(p == y for p, y in valid), len(valid)) if valid else None


def observation_classification_metrics(
    predicted_arc_ids: Sequence[int | None], reference_arc_ids: Sequence[int | None]
) -> dict[str, object]:
    """Multiclass arc-ID metrics over labeled observations, separate from candidate-row scores."""
    if len(predicted_arc_ids) != len(reference_arc_ids):
        raise ValueError("prediction and reference labels must align")
    pairs = [(p, y) for p, y in zip(predicted_arc_ids, reference_arc_ids) if y is not None]
    labels: list[int | str] = sorted({int(y) for _, y in pairs} | {int(p) for p, _ in pairs if p is not None})
    if any(p is None for p, _ in pairs):
        labels.append("missing")
    confusion = [[0 for _ in labels] for _ in labels]
    index = {label: i for i, label in enumerate(labels)}
    for prediction, truth in pairs:
        true_label = int(truth)
        predicted_label: int | str = "missing" if prediction is None else int(prediction)
        confusion[index[true_label]][index[predicted_label]] += 1
    precisions, recalls, f1s = [], [], []
    for i in range(len(labels)):
        tp = confusion[i][i]
        fp = sum(confusion[row][i] for row in range(len(labels)) if row != i)
        fn = sum(confusion[i][col] for col in range(len(labels)) if col != i)
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(_ratio(2 * precision * recall, precision + recall))
    count = len(pairs)
    return {
        "accuracy": _ratio(sum(prediction == truth for prediction, truth in pairs), count) if count else None,
        "macro_precision": sum(precisions) / len(labels) if labels else None,
        "macro_recall": sum(recalls) / len(labels) if labels else None,
        "macro_f1": sum(f1s) / len(labels) if labels else None,
        "confusion_matrix": {"labels": labels, "counts": confusion},
        "labeled_observation_count": count,
    }


def route_continuity(predicted_arc_ids: Sequence[int | None], graph: RoadGraph) -> dict[str, float | int]:
    pairs = list(zip(predicted_arc_ids, predicted_arc_ids[1:]))
    breaks = sum(a is None or b is None or (a != b and not graph.connected(a, b)) for a, b in pairs)
    return {"route_break_count": breaks, "route_continuity": _ratio(len(pairs) - breaks, len(pairs)) if pairs else 1.0}


def route_sequence_metrics(predicted: Sequence[int], reference: Sequence[int]) -> dict[str, float | int]:
    pred_set, ref_set = set(predicted), set(reference)
    intersection = len(pred_set & ref_set)
    precision = _ratio(intersection, len(pred_set))
    recall = _ratio(intersection, len(ref_set))
    union = len(pred_set | ref_set)
    return {
        "route_arc_set_precision": precision,
        "route_arc_set_recall": recall,
        "route_arc_set_f1": _ratio(2 * precision * recall, precision + recall),
        "route_arc_set_jaccard": _ratio(intersection, union),
        "arc_sequence_edit_distance": edit_distance(predicted, reference),
    }


def edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Levenshtein distance in O(min(len(a), len(b))) memory."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, va in enumerate(a, 1):
        current = [i]
        for j, vb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (va != vb)))
        previous = current
    return previous[-1]


def evaluate_predictions(
    candidate_sets: Sequence[Sequence[Candidate]],
    probabilities: Sequence[Sequence[float]],
    classifier_arc_ids: Sequence[int | None],
    hmm_arc_ids: Sequence[int | None],
    reference_labels: Sequence[int | None],
    predicted_route: Sequence[int],
    reference_route: Sequence[int],
    graph: RoadGraph,
    timings: dict[str, float] | None = None,
) -> dict[str, object]:
    """Server-side helper. Call only after inference; never imported by inference.py."""
    if len(classifier_arc_ids) != len(hmm_arc_ids):
        raise ValueError("classifier and HMM predictions must align")
    out = candidate_metrics(candidate_sets, probabilities, reference_labels)
    out.update({
        "classifier_top1_accuracy": observation_top1(classifier_arc_ids, reference_labels),
        "hmm_top1_accuracy": observation_top1(hmm_arc_ids, reference_labels),
        "classifier_observation_metrics": observation_classification_metrics(classifier_arc_ids, reference_labels),
        "hmm_observation_metrics": observation_classification_metrics(hmm_arc_ids, reference_labels),
        "classifier_route_continuity": route_continuity(classifier_arc_ids, graph),
        "hmm_route_continuity": route_continuity(hmm_arc_ids, graph),
        "stitched_route_continuity": route_continuity(predicted_route, graph),
        "stitched_route_metrics": route_sequence_metrics(predicted_route, reference_route),
        "stage_timings_s": timings or {},
    })
    return out


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


