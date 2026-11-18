"""Ground-truth-free prediction orchestration with explicit Person A adapters."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import DecoderConfig
from .graph import Candidate, RoadGraph, load_road_graph
from .hmm import viterbi_decode
from .projection import LocalProjection
from .stitch import stitch_route


@dataclass(frozen=True, slots=True)
class Observation:
    index: int
    lon: float | None
    lat: float | None
    timestamp: float
    xy: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class InferenceComponents:
    """Adapters for Person A's candidate generation, features, and exported scorer."""

    generate_candidates: Callable[[Sequence[tuple[float, float] | None], RoadGraph], Sequence[Sequence[Candidate]]]
    build_features: Callable[[Sequence[Observation], Sequence[Sequence[Candidate]], RoadGraph], Any]
    score_candidates: Callable[[Any], Sequence[Sequence[float]] | Sequence[float]]


class IntegrationError(RuntimeError):
    """Raised when the upstream candidate/model components have not been supplied."""


_GRAPH_CACHE: OrderedDict[tuple[str, int, int, int, int], RoadGraph] = OrderedDict()
_GRAPH_CACHE_LIMIT = 1


def predict_trip(
    trip_dir: str | Path,
    components: InferenceComponents | None = None,
    config: DecoderConfig | None = None,
) -> dict[str, object]:
    """Predict one arc per valid track observation, without opening a reference route file."""
    if components is None:
        raise IntegrationError("Person A adapters are required: generate_candidates, build_features, and score_candidates")
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    config = config or DecoderConfig()
    timings: dict[str, float] = {}

    start = time.perf_counter()
    raw = _read_track(trip_dir / f"{trip_id}.track")
    node_coords = _read_lonlat_nodes(trip_dir / f"{trip_id}.nodes")
    projection = LocalProjection.from_coordinates(node_coords + [(o.lon, o.lat) for o in raw if o.lon is not None and o.lat is not None])
    observations = [Observation(o.index, o.lon, o.lat, o.timestamp,
                                projection.forward(o.lon, o.lat) if o.lon is not None and o.lat is not None else None) for o in raw]
    timings["projection"] = time.perf_counter() - start

    start = time.perf_counter()
    graph = _get_graph(trip_dir, config, projection)
    timings["graph_and_spatial_index"] = time.perf_counter() - start

    start = time.perf_counter()
    candidate_sets = [list(row) for row in components.generate_candidates([o.xy for o in observations], graph)]
    if len(candidate_sets) != len(observations):
        raise ValueError("candidate generator must return one candidate list per track observation")
    for index, observation in enumerate(observations):
        if observation.xy is None:
            candidate_sets[index] = []
    timings["candidate_generation"] = time.perf_counter() - start

    start = time.perf_counter()
    features = components.build_features(observations, candidate_sets, graph)
    timings["feature_generation"] = time.perf_counter() - start

    start = time.perf_counter()
    probabilities = _nest_scores(components.score_candidates(features), candidate_sets)
    timings["classifier_scoring"] = time.perf_counter() - start

    start = time.perf_counter()
    classifier = [_best_arc(candidates, scores) for candidates, scores in zip(candidate_sets, probabilities)]
    selected = viterbi_decode(candidate_sets, probabilities, [o.timestamp for o in observations], [o.xy for o in observations], graph, config)
    hmm = [candidate.arc_id if candidate is not None else None for candidate in selected]
    timings["hmm_viterbi"] = time.perf_counter() - start

    start = time.perf_counter()
    stitched = stitch_route(selected, graph, [o.timestamp for o in observations], config.gap_threshold_s)
    polyline = graph.route_polyline(stitched.arc_ids)
    timings["route_stitching"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())

    warnings = []
    if graph.invalid_arcs:
        skipped = sum("retained" not in warning for warning in graph.invalid_arcs)
        retained = len(graph.invalid_arcs) - skipped
        if skipped:
            warnings.append(f"{skipped} invalid arc rows were skipped")
        if retained:
            warnings.append(f"{retained} zero-length arc rows were retained")
    missing = sum(not row for row in candidate_sets)
    if missing:
        warnings.append(f"No candidate was generated for {missing} observation(s); predictions are null at those positions")
    if stitched.breaks:
        warnings.append(f"{len(stitched.breaks)} disconnected route transition(s) are represented as route breaks")
    return {
        "trip_id": trip_id,
        "observations": [
            {"index": o.index, "coordinates": [o.lon, o.lat] if o.lon is not None and o.lat is not None else None,
             "timestamp": o.timestamp if math.isfinite(o.timestamp) else None,
             "classifier_arc_id": classifier[i], "hmm_arc_id": hmm[i], "candidate_count": len(candidate_sets[i])}
            for i, o in enumerate(observations)
        ],
        "classifier_arc_ids": classifier,
        "hmm_arc_ids": hmm,
        "route_arc_ids": stitched.arc_ids,
        "route_polyline": polyline,
        "route_breaks": stitched.breaks,
        "stage_timings_s": timings,
        "inference_seconds": timings["total"],
        "warnings": warnings,
    }


def _get_graph(trip_dir: Path, config: DecoderConfig, projection: LocalProjection) -> RoadGraph:
    trip_id = trip_dir.name
    node_file, arc_file, track_file = (trip_dir / f"{trip_id}.{ext}" for ext in ("nodes", "arcs", "track"))
    key = (str(trip_dir.resolve()), node_file.stat().st_mtime_ns, arc_file.stat().st_mtime_ns,
           track_file.stat().st_mtime_ns, hash(config))
    graph = _GRAPH_CACHE.get(key)
    if graph is None:
        graph = load_road_graph(trip_dir, config, projection)
        _GRAPH_CACHE[key] = graph
        while len(_GRAPH_CACHE) > _GRAPH_CACHE_LIMIT:
            _GRAPH_CACHE.popitem(last=False)
    else:
        _GRAPH_CACHE.move_to_end(key)
    return graph


def _read_track(path: Path) -> list[Observation]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        fields = line.strip().split()
        if len(fields) != 3:
            rows.append(Observation(index, None, None, math.nan, None))
            continue
        try:
            lon, lat, timestamp = map(float, fields)
        except ValueError:
            rows.append(Observation(index, None, None, math.nan, None))
            continue
        valid = math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90
        rows.append(Observation(index, lon if valid else None, lat if valid else None, timestamp, None))
    return rows


def _read_lonlat_nodes(path: Path) -> list[tuple[float, float]]:
    points = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            lon, lat = map(float, fields)
        except ValueError:
            continue
        if math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90:
            points.append((lon, lat))
    if not points:
        raise ValueError(f"no valid road nodes in {path}")
    return points


def _nest_scores(scores: Sequence[Sequence[float]] | Sequence[float], candidates: Sequence[Sequence[Candidate]]) -> list[list[float]]:
    values = list(scores)
    if values and isinstance(values[0], (list, tuple)):
        nested = [[float(p) for p in row] for row in values]  # type: ignore[arg-type]
    else:
        flat = [float(p) for p in values]  # type: ignore[arg-type]
        nested, offset = [], 0
        for row in candidates:
            nested.append(flat[offset:offset + len(row)])
            offset += len(row)
        if offset != len(flat):
            raise ValueError("flat score output length does not match flattened candidates")
    if len(nested) != len(candidates) or any(len(p) != len(c) for p, c in zip(nested, candidates)):
        raise ValueError("score_candidates must return one score per candidate in input order")
    return nested


def _best_arc(candidates: Sequence[Candidate], scores: Sequence[float]) -> int | None:
    if not candidates:
        return None
    if len(candidates) != len(scores):
        raise ValueError("candidate and score counts differ")
    def rank(i: int) -> tuple[float, int]:
        score = float(scores[i])
        return (score if math.isfinite(score) else -math.inf, -candidates[i].arc_id)
    return candidates[max(range(len(candidates)), key=rank)].arc_id

