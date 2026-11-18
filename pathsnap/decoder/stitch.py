"""Reconstruct a legal ordered route from per-observation arc positions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .graph import Candidate, RoadGraph


@dataclass(frozen=True, slots=True)
class RouteStitch:
    arc_ids: list[int]
    breaks: list[dict[str, int | str]]


def stitch_route(
    selected: list[Candidate | None],
    graph: RoadGraph,
    timestamps: Sequence[float] | None = None,
    gap_threshold_s: float | None = None,
) -> RouteStitch:
    if timestamps is not None and len(timestamps) != len(selected):
        raise ValueError("timestamps must align with selected states")
    route: list[int] = []
    breaks: list[dict[str, int | str]] = []
    previous: Candidate | None = None
    for index, candidate in enumerate(selected):
        if (index and timestamps is not None and gap_threshold_s is not None
                and (not math.isfinite(timestamps[index - 1]) or not math.isfinite(timestamps[index])
                     or timestamps[index] < timestamps[index - 1]
                     or timestamps[index] - timestamps[index - 1] > gap_threshold_s)):
            if previous is not None and candidate is not None:
                breaks.append({"from_arc_id": previous.arc_id, "to_arc_id": candidate.arc_id, "reason": "time_gap"})
            previous = None
        if candidate is None:
            if previous is not None:
                breaks.append({"from_arc_id": previous.arc_id, "to_arc_id": previous.arc_id, "reason": "missing_candidates"})
            previous = None
            continue
        if previous is None:
            _append(route, candidate.arc_id)
            previous = candidate
            continue
        if previous.arc_id == candidate.arc_id:
            previous = candidate
            continue
        first, second = graph.arcs.get(previous.arc_id), graph.arcs.get(candidate.arc_id)
        if first is None or second is None:
            breaks.append({"from_arc_id": previous.arc_id, "to_arc_id": candidate.arc_id, "reason": "invalid_arc"})
            _append(route, candidate.arc_id)
            previous = candidate
            continue
        middle: list[int] = []
        if first.head != second.tail:
            shortest = graph.shortest_path(first.head, second.tail)
            if shortest is None:
                breaks.append({"from_arc_id": previous.arc_id, "to_arc_id": candidate.arc_id, "reason": "disconnected"})
                _append(route, candidate.arc_id)
                previous = candidate
                continue
            middle = shortest[1]
        for arc_id in middle:
            _append(route, arc_id)
        _append(route, candidate.arc_id)
        previous = candidate
    return RouteStitch(route, breaks)


def _append(route: list[int], arc_id: int) -> None:
    if not route or route[-1] != arc_id:
        route.append(arc_id)

