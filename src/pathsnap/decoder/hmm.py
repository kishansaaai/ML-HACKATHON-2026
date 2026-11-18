"""Log-space variable-state HMM/Viterbi decoding for directed road candidates."""

from __future__ import annotations

import math
from typing import Sequence

from .config import DecoderConfig
from .graph import Candidate, RoadGraph


def emission_log_scores(
    candidates: Sequence[Candidate],
    classifier_scores: Sequence[float],
    config: DecoderConfig,
) -> list[float]:
    """Normalize nonnegative classifier scores per observation, then take log."""
    if len(candidates) != len(classifier_scores):
        raise ValueError("candidate and classifier score counts differ")
    scores = [max(0.0, float(p)) if math.isfinite(float(p)) else 0.0 for p in classifier_scores]
    total = sum(scores)
    if scores and total <= 0:
        scores = [1.0] * len(scores)
        total = len(scores)
    if total:
        scores = [p / total for p in scores]
    logs = [math.log(max(config.min_probability, p)) for p in scores]
    if config.emission_sigma_m is not None:
        sigma2 = 2 * config.emission_sigma_m**2
        logs = [score - candidate.distance_m**2 / sigma2 for score, candidate in zip(logs, candidates)]
    return logs


def transition_log_score(
    source: Candidate,
    target: Candidate,
    gps_distance_m: float,
    graph: RoadGraph,
    config: DecoderConfig,
) -> float:
    cutoff = min(config.transition_cutoff_m,
                 max(0.0, gps_distance_m + config.beta_m * abs(config.unreachable_log_penalty)))
    network = graph.network_distance(source, target, cutoff)
    if network is None:
        return config.unreachable_log_penalty
    return -abs(network - max(0.0, gps_distance_m)) / config.beta_m


def viterbi_decode(
    candidate_sets: Sequence[Sequence[Candidate]],
    classifier_scores: Sequence[Sequence[float]],
    timestamps: Sequence[float],
    observation_xy: Sequence[tuple[float, float] | None],
    graph: RoadGraph,
    config: DecoderConfig | None = None,
) -> list[Candidate | None]:
    """Decode each contiguous candidate segment; missing observations and long gaps restart."""
    config = config or graph.config
    n = len(candidate_sets)
    if not (len(classifier_scores) == len(timestamps) == len(observation_xy) == n):
        raise ValueError("candidate sets, scores, timestamps and coordinates must have equal lengths")
    result: list[Candidate | None] = [None] * n
    start = 0
    while start < n:
        while start < n and not candidate_sets[start]:
            start += 1
        if start >= n:
            break
        end = start + 1
        while end < n and candidate_sets[end] and _valid_pair(timestamps[end - 1], timestamps[end], config.gap_threshold_s):
            end += 1
        result[start:end] = _decode_segment(candidate_sets[start:end], classifier_scores[start:end], timestamps[start:end], observation_xy[start:end], graph, config)
        start = end
    return result


def _valid_pair(previous: float, current: float, threshold: float) -> bool:
    return math.isfinite(previous) and math.isfinite(current) and 0 <= current - previous <= threshold


def _decode_segment(
    candidate_sets: Sequence[Sequence[Candidate]],
    scores: Sequence[Sequence[float]],
    timestamps: Sequence[float],
    coords: Sequence[tuple[float, float] | None],
    graph: RoadGraph,
    config: DecoderConfig,
) -> list[Candidate | None]:
    emissions = [emission_log_scores(candidates, p, config) for candidates, p in zip(candidate_sets, scores)]
    dynamic = emissions[0][:]
    backpointers: list[list[int]] = [[-1] * len(candidate_sets[0])]
    for t in range(1, len(candidate_sets)):
        prev_candidates, current_candidates = candidate_sets[t - 1], candidate_sets[t]
        gps_distance = _euclidean(coords[t - 1], coords[t])
        # A path whose transition score is already below the configured finite
        # unreachable penalty cannot improve Viterbi, so it is safe to prune it.
        effective_cutoff = min(config.transition_cutoff_m,
                               max(0.0, gps_distance + config.beta_m * abs(config.unreachable_log_penalty)))
        network_distances = graph.network_distance_matrix(prev_candidates, current_candidates, effective_cutoff)
        next_dynamic: list[float] = []
        pointers: list[int] = []
        for j, current in enumerate(current_candidates):
            best_score, best_prev = -math.inf, 0
            for i, previous in enumerate(prev_candidates):
                network = network_distances[i][j]
                transition = (config.unreachable_log_penalty if network is None
                              else -abs(network - max(0.0, gps_distance)) / config.beta_m)
                score = dynamic[i] + transition
                if score > best_score:
                    best_score, best_prev = score, i
            next_dynamic.append(best_score + emissions[t][j])
            pointers.append(best_prev)
        dynamic = next_dynamic
        backpointers.append(pointers)
    state = max(range(len(dynamic)), key=dynamic.__getitem__)
    path: list[Candidate] = [candidate_sets[-1][state]]
    for t in range(len(candidate_sets) - 1, 0, -1):
        state = backpointers[t][state]
        path.append(candidate_sets[t - 1][state])
    path.reverse()
    return path


def _euclidean(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


