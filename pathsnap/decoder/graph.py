"""Per-trip directed road graph, arc geometry, spatial lookup, and bounded paths."""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Iterable

from .config import DecoderConfig
from .projection import LocalProjection

XY = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Arc:
    arc_id: int
    tail: int
    head: int
    start: XY
    end: XY
    length_m: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One projected position on a directed arc; fraction is measured tail to head."""

    arc_id: int
    fraction: float
    distance_m: float
    xy: XY


class ArcSpatialIndex:
    """Uniform-grid index of segment bounding boxes with stable original arc IDs."""

    def __init__(self, arcs: dict[int, Arc], cell_size_m: float = 128.0, max_cells_per_arc: int = 256):
        if cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        self.cell_size_m = cell_size_m
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.large_arcs: list[int] = []
        self.arcs = arcs
        for arc_id, arc in arcs.items():
            x0, y0, x1, y1 = arc.bbox
            ix0, iy0 = self._cell(x0, y0)
            ix1, iy1 = self._cell(x1, y1)
            count = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
            if count > max_cells_per_arc:
                self.large_arcs.append(arc_id)
                continue
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    self.cells[(ix, iy)].append(arc_id)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return math.floor(x / self.cell_size_m), math.floor(y / self.cell_size_m)

    def query(self, point: XY, radius_m: float) -> list[Candidate]:
        if radius_m < 0:
            raise ValueError("radius_m cannot be negative")
        x, y = point
        ix0, iy0 = self._cell(x - radius_m, y - radius_m)
        ix1, iy1 = self._cell(x + radius_m, y + radius_m)
        ids = set(self.large_arcs)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                ids.update(self.cells.get((ix, iy), ()))
        out = []
        for arc_id in ids:
            arc = self.arcs[arc_id]
            projected, fraction, distance = project_point_segment(point, arc.start, arc.end)
            if distance <= radius_m:
                out.append(Candidate(arc_id, fraction, distance, projected))
        return sorted(out, key=lambda c: (c.distance_m, c.arc_id))

    def nearest(self, point: XY, k: int = 1, initial_radius_m: float = 32.0) -> list[Candidate]:
        """Nearest-K lookup with expanding grid windows; does not invent a candidate."""
        if k < 1:
            return []
        radius = max(initial_radius_m, self.cell_size_m / 2)
        # Expansion eventually covers the finite index; cap iterations to guard bad coordinates.
        for _ in range(32):
            found = self.query(point, radius)
            if len(found) >= k:
                return found[:k]
            radius *= 2
        # Rare fallback for a point far beyond the network extent.
        all_candidates = []
        for arc_id, arc in self.arcs.items():
            projected, fraction, distance = project_point_segment(point, arc.start, arc.end)
            all_candidates.append(Candidate(arc_id, fraction, distance, projected))
        return sorted(all_candidates, key=lambda c: (c.distance_m, c.arc_id))[:k]


def project_point_segment(point: XY, start: XY, end: XY) -> tuple[XY, float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denom = dx * dx + dy * dy
    fraction = 0.0 if denom == 0 else max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denom))
    projected = (start[0] + fraction * dx, start[1] + fraction * dy)
    return projected, fraction, math.hypot(point[0] - projected[0], point[1] - projected[1])


class RoadGraph:
    """Sparse directed graph. Every graph edge retains its source arc ID."""

    def __init__(
        self,
        nodes: list[XY],
        node_lonlat: list[tuple[float, float]],
        arcs: dict[int, Arc],
        projection: LocalProjection,
        config: DecoderConfig | None = None,
        invalid_arcs: list[str] | None = None,
    ):
        self.nodes = nodes
        self.node_lonlat = node_lonlat
        self.arcs = arcs
        self.projection = projection
        self.config = config or DecoderConfig()
        self.invalid_arcs = invalid_arcs or []
        self.geometry_warnings: list[str] = []
        self.adjacency: list[list[tuple[int, int, float]]] = [[] for _ in nodes]
        self.incoming: list[list[int]] = [[] for _ in nodes]
        for arc in arcs.values():
            self.adjacency[arc.tail].append((arc.head, arc.arc_id, arc.length_m))
            self.incoming[arc.head].append(arc.arc_id)
        self.spatial_index = ArcSpatialIndex(arcs, self.config.spatial_cell_m, self.config.max_cells_per_arc)
        self._distance_cache: OrderedDict[tuple[int, ...], float] = OrderedDict()
        self._source_cache: OrderedDict[int, dict[int, float]] = OrderedDict()

    def arc_endpoints(self, arc_id: int) -> tuple[int, int]:
        arc = self.arcs[arc_id]
        return arc.tail, arc.head

    def arc_length(self, arc_id: int) -> float:
        return self.arcs[arc_id].length_m

    def connected(self, arc_a: int, arc_b: int) -> bool:
        return self.arcs[arc_a].head == self.arcs[arc_b].tail

    def shortest_path(self, start_node: int, end_node: int, cutoff_m: float | None = None) -> tuple[float, list[int]] | None:
        cutoff = self.config.transition_cutoff_m if cutoff_m is None else cutoff_m
        if cutoff < 0:
            return None
        cache_key = (start_node, end_node, int(round(cutoff * 10)))
        if cache_key in self._distance_cache:
            self._distance_cache.move_to_end(cache_key)
            dist = self._distance_cache[cache_key]
            return (dist, self._recover_path(start_node, end_node, cutoff)) if math.isfinite(dist) else None
        distances, parents = self._dijkstra(start_node, cutoff)
        dist = distances.get(end_node, math.inf)
        self._remember(self._distance_cache, cache_key, dist, self.config.distance_cache_size)
        # Keep parent information only for the current query; distance cache is the reusable part.
        if math.isfinite(dist):
            return dist, self._backtrack(start_node, end_node, parents)
        return None

    def _dijkstra(self, start: int, cutoff: float) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
        if not 0 <= start < len(self.nodes):
            return {}, {}
        distances = {start: 0.0}
        parents: dict[int, tuple[int, int]] = {}
        heap = [(0.0, start)]
        while heap:
            distance, node = heappop(heap)
            if distance != distances.get(node):
                continue
            for neighbor, arc_id, weight in self.adjacency[node]:
                new_distance = distance + weight
                if new_distance <= cutoff and new_distance < distances.get(neighbor, math.inf):
                    distances[neighbor] = new_distance
                    parents[neighbor] = (node, arc_id)
                    heappush(heap, (new_distance, neighbor))
        return distances, parents

    def _backtrack(self, start: int, end: int, parents: dict[int, tuple[int, int]]) -> list[int]:
        path: list[int] = []
        node = end
        while node != start:
            if node not in parents:
                return []
            prev, arc_id = parents[node]
            path.append(arc_id)
            node = prev
        path.reverse()
        return path

    def _recover_path(self, start: int, end: int, cutoff: float) -> list[int]:
        _, parents = self._dijkstra(start, cutoff)
        return self._backtrack(start, end, parents)

    def network_distance(self, source: Candidate, target: Candidate, cutoff_m: float | None = None) -> float | None:
        """Directed distance between positions on arcs, including partial arc lengths."""
        cutoff = self.config.transition_cutoff_m if cutoff_m is None else cutoff_m
        key = (source.arc_id, round(source.fraction * 1_000_000), target.arc_id, round(target.fraction * 1_000_000), int(round(cutoff * 10)))
        if key in self._distance_cache:
            self._distance_cache.move_to_end(key)
            result = self._distance_cache[key]
            return result if math.isfinite(result) else None
        a, b = self.arcs.get(source.arc_id), self.arcs.get(target.arc_id)
        if a is None or b is None:
            return None
        direct = (target.fraction - source.fraction) * a.length_m if source.arc_id == target.arc_id else math.inf
        if math.isfinite(direct) and direct >= 0:
            self._remember(self._distance_cache, key, direct, self.config.distance_cache_size)
            return direct
        prefix_out = (1.0 - source.fraction) * a.length_m
        prefix_in = target.fraction * b.length_m
        remainder = cutoff - prefix_out - prefix_in
        if remainder < 0:
            return None
        if a.head == b.tail:
            middle = 0.0
        else:
            middle = self._node_distance(a.head, b.tail, remainder)
        result = math.inf if middle is None else prefix_out + middle + prefix_in
        if result > cutoff:
            result = math.inf
        self._remember(self._distance_cache, key, result, self.config.distance_cache_size)
        return result if math.isfinite(result) else None

    def network_distance_matrix(
        self,
        sources: Iterable[Candidate],
        targets: Iterable[Candidate],
        cutoff_m: float | None = None,
    ) -> list[list[float | None]]:
        """Batch candidate transitions, sharing each bounded search across destination arcs."""
        source_list, target_list = list(sources), list(targets)
        cutoff = self.config.transition_cutoff_m if cutoff_m is None else cutoff_m
        matrix: list[list[float | None]] = [[None] * len(target_list) for _ in source_list]
        for i, source in enumerate(source_list):
            a = self.arcs.get(source.arc_id)
            if a is None:
                continue
            unresolved: list[int] = []
            for j, target in enumerate(target_list):
                b = self.arcs.get(target.arc_id)
                if b is None:
                    continue
                direct = (target.fraction - source.fraction) * a.length_m if source.arc_id == target.arc_id else math.inf
                key = (source.arc_id, round(source.fraction * 1_000_000), target.arc_id,
                       round(target.fraction * 1_000_000), int(round(cutoff * 10)))
                cached = self._distance_cache.get(key)
                if cached is not None:
                    self._distance_cache.move_to_end(key)
                    matrix[i][j] = cached if math.isfinite(cached) else None
                elif math.isfinite(direct) and direct >= 0:
                    matrix[i][j] = direct
                    self._remember(self._distance_cache, key, direct, self.config.distance_cache_size)
                elif a.head == b.tail:
                    distance = (1.0 - source.fraction) * a.length_m + target.fraction * b.length_m
                    if distance <= cutoff:
                        matrix[i][j] = distance
                    self._remember(self._distance_cache, key, distance if distance <= cutoff else math.inf, self.config.distance_cache_size)
                else:
                    unresolved.append(j)
            if not unresolved:
                continue
            start_cost = (1.0 - source.fraction) * a.length_m
            allowed = cutoff - start_cost
            target_nodes = {self.arcs[target_list[j].arc_id].tail for j in unresolved
                            if target_list[j].arc_id in self.arcs and target_list[j].fraction * self.arcs[target_list[j].arc_id].length_m <= allowed}
            node_distances = self._dijkstra_targets(a.head, target_nodes, min(self.config.transition_cutoff_m, max(0.0, allowed)))
            for j in unresolved:
                b = self.arcs[target_list[j].arc_id]
                prefix = target_list[j].fraction * b.length_m
                middle = node_distances.get(b.tail, math.inf)
                distance = start_cost + middle + prefix
                key = (source.arc_id, round(source.fraction * 1_000_000), target_list[j].arc_id,
                       round(target_list[j].fraction * 1_000_000), int(round(cutoff * 10)))
                if middle < math.inf and distance <= cutoff and allowed >= 0:
                    matrix[i][j] = distance
                    self._remember(self._distance_cache, key, distance, self.config.distance_cache_size)
                else:
                    self._remember(self._distance_cache, key, math.inf, self.config.distance_cache_size)
        return matrix

    def _dijkstra_targets(self, start: int, targets: set[int], cutoff: float) -> dict[int, float]:
        if not targets or not 0 <= start < len(self.nodes):
            return {}
        remaining = set(targets)
        found: dict[int, float] = {}
        distances = {start: 0.0}
        heap = [(0.0, start)]
        while heap and remaining:
            distance, node = heappop(heap)
            if distance != distances.get(node):
                continue
            if node in remaining:
                found[node] = distance
                remaining.remove(node)
                if not remaining:
                    break
            for neighbor, _arc_id, weight in self.adjacency[node]:
                new_distance = distance + weight
                if new_distance <= cutoff and new_distance < distances.get(neighbor, math.inf):
                    distances[neighbor] = new_distance
                    heappush(heap, (new_distance, neighbor))
        return found

    def _node_distance(self, start: int, end: int, cutoff: float) -> float | None:
        if start == end:
            return 0.0
        cached = self._source_cache.get(start)
        if cached is not None:
            self._source_cache.move_to_end(start)
            dist = cached.get(end)
            return dist if dist is not None and dist <= cutoff else None
        dist = self._dijkstra_distance(start, end, self.config.transition_cutoff_m)
        # Cache only requested destinations: retaining full 88k-node search maps
        # per source wastes memory, while repeated HMM pairs hit this small cache.
        source_distances = self._source_cache.setdefault(start, {})
        source_distances[end] = dist
        self._source_cache.move_to_end(start)
        while len(self._source_cache) > self.config.source_cache_size:
            self._source_cache.popitem(last=False)
        return dist if math.isfinite(dist) and dist <= cutoff else None

    def _dijkstra_distance(self, start: int, target: int, cutoff: float) -> float:
        if not (0 <= start < len(self.nodes) and 0 <= target < len(self.nodes)):
            return math.inf
        distances = {start: 0.0}
        heap = [(0.0, start)]
        while heap:
            distance, node = heappop(heap)
            if distance != distances.get(node):
                continue
            if node == target:
                return distance
            for neighbor, _arc_id, weight in self.adjacency[node]:
                new_distance = distance + weight
                if new_distance <= cutoff and new_distance < distances.get(neighbor, math.inf):
                    distances[neighbor] = new_distance
                    heappush(heap, (new_distance, neighbor))
        return math.inf

    @staticmethod
    def _remember(cache: OrderedDict, key: object, value: object, limit: int) -> None:
        if limit <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def route_polyline(self, route_arc_ids: Iterable[int]) -> list[list[list[float]]]:
        """Build connected lon/lat line segments; disconnected route breaks stay separate."""
        lines: list[list[list[float]]] = []
        current: list[list[float]] = []
        previous: Arc | None = None
        for arc_id in route_arc_ids:
            arc = self.arcs.get(int(arc_id))
            if arc is None:
                continue
            start = list(self.projection.inverse(*arc.start))
            end = list(self.projection.inverse(*arc.end))
            if previous is None or previous.head != arc.tail:
                if current:
                    lines.append(current)
                current = [start, end]
            else:
                if current[-1] != start:
                    current.append(start)
                current.append(end)
            previous = arc
        if current:
            lines.append(current)
        return lines


def load_road_graph(
    trip_dir: str | Path,
    config: DecoderConfig | None = None,
    projection: LocalProjection | None = None,
) -> RoadGraph:
    """Load nodes and arcs from a trip directory. Arc row number is preserved as arc_id."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    node_path, arc_path = trip_dir / f"{trip_id}.nodes", trip_dir / f"{trip_id}.arcs"
    if not node_path.exists() or not arc_path.exists():
        raise FileNotFoundError(f"trip requires {node_path.name} and {arc_path.name}")
    raw_nodes: list[tuple[float, float]] = []
    for line_no, line in enumerate(node_path.read_text().splitlines(), 1):
        values = _numbers(line, 2)
        if values is None:
            raise ValueError(f"invalid node row {line_no} in {node_path}")
        lon, lat = values
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"invalid longitude/latitude in node row {line_no}: {values}")
        raw_nodes.append((lon, lat))
    projection = projection or LocalProjection.from_coordinates(raw_nodes)
    nodes = [projection.forward(*xy) for xy in raw_nodes]
    arcs: dict[int, Arc] = {}
    invalid: list[str] = []
    for arc_id, line in enumerate(arc_path.read_text().splitlines()):
        values = _numbers(line, 2)
        if values is None or any(not float(x).is_integer() for x in values):
            invalid.append(f"arc row {arc_id}: expected two integer endpoint indices")
            continue
        tail, head = map(int, values)
        if not (0 <= tail < len(nodes) and 0 <= head < len(nodes)):
            invalid.append(f"arc row {arc_id}: node reference out of range ({tail}, {head})")
            continue
        start, end = nodes[tail], nodes[head]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length == 0:
            invalid.append(f"arc row {arc_id}: zero-length geometry retained")
        arcs[arc_id] = Arc(arc_id, tail, head, start, end, length,
                           (min(start[0], end[0]), min(start[1], end[1]), max(start[0], end[0]), max(start[1], end[1])))
    return RoadGraph(nodes, raw_nodes, arcs, projection, config, invalid)


def _numbers(line: str, count: int) -> list[float] | None:
    try:
        fields = line.strip().split()
        if len(fields) != count:
            return None
        values = [float(value) for value in fields]
        return values if all(math.isfinite(value) for value in values) else None
    except ValueError:
        return None

