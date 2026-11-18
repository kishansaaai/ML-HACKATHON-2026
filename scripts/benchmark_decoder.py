#!/usr/bin/env python3
"""Measure decoder primitives on one training trip; this does not score predictions."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.graph import Candidate, load_road_graph
from pathsnap.decoder.hmm import viterbi_decode
from pathsnap.decoder.inference import _read_lonlat_nodes, _read_track
from pathsnap.decoder.projection import LocalProjection
from pathsnap.decoder.stitch import stitch_route


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trip_dir", type=Path, help="one training trip folder (for example Student folder/train/00000001)")
    parser.add_argument("--samples", type=int, default=100, help="maximum adjacent GPS samples for primitive microbenchmarks")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()
    trip_id = args.trip_dir.name
    nodes, node_read_s = timed(lambda: _read_lonlat_nodes(args.trip_dir / f"{trip_id}.nodes"))
    observations, track_read_s = timed(lambda: _read_track(args.trip_dir / f"{trip_id}.track"))
    gps = [(o.lon, o.lat) for o in observations if o.lon is not None and o.lat is not None]
    projection = LocalProjection.from_coordinates(nodes + gps)
    _, projection_s = timed(lambda: [projection.forward(*p) for p in nodes])
    config = DecoderConfig()
    graph, graph_s = timed(lambda: load_road_graph(args.trip_dir, config, projection))
    # The parsed observations keep raw coordinates; project a bounded, evenly spaced sample.
    sample = observations[:max(0, args.samples)]
    points = [projection.forward(o.lon, o.lat) if o.lon is not None and o.lat is not None else None for o in sample]

    def query_candidates():
        result = []
        for point in points:
            if point is None:
                result.append([])
            else:
                result.append(graph.spatial_index.query(point, 50.0)[:8])
        return result

    candidate_sets, lookup_s = timed(query_candidates)
    timestamps = [o.timestamp for o in sample]
    scores = [[1.0 / len(row)] * len(row) if row else [] for row in candidate_sets]
    xy = points

    def transition_work():
        calls = 0
        for index, (left, right) in enumerate(zip(candidate_sets, candidate_sets[1:]), 1):
            a, b = points[index - 1], points[index]
            gps_distance = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** .5 if a is not None and b is not None else 0.0
            cutoff = min(config.transition_cutoff_m, max(0.0, gps_distance + config.beta_m * abs(config.unreachable_log_penalty)))
            graph.network_distance_matrix(left, right, cutoff)
            calls += len(left) * len(right)
        return calls

    transition_calls, transition_s = timed(transition_work)
    graph._distance_cache.clear()
    graph._source_cache.clear()
    decoded, viterbi_s = timed(lambda: viterbi_decode(candidate_sets, scores, timestamps, xy, graph, config))
    stitched, stitch_s = timed(lambda: stitch_route(decoded, graph, timestamps, config.gap_threshold_s))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    peak_mb = peak / (1024 * 1024) if __import__("sys").platform == "darwin" else peak / 1024
    result = {
        "benchmark_type": "decoder_engine_microbenchmark_no_model_or_ground_truth",
        "trip_id": trip_id,
        "dataset_partition_requirement": "training only",
        "counts": {"nodes": len(nodes), "arcs": len(graph.arcs), "track_observations": len(observations),
                   "sampled_observations": len(sample), "transition_pairs": transition_calls,
                   "stitched_arc_count": len(stitched.arc_ids)},
        "stage_timings_s": {"read_nodes": node_read_s, "read_track": track_read_s,
                             "project_all_nodes": projection_s, "graph_build_and_spatial_index": graph_s,
                             "sample_spatial_lookup": lookup_s, "sample_candidate_transitions": transition_s,
                             "sample_viterbi": viterbi_s, "sample_route_stitching": stitch_s},
        "peak_process_memory_mb": peak_mb,
        "limitations": ["candidate microbenchmark caps each point at eight arcs and 50 m radius",
                        "uniform synthetic candidate scores are used only to time the decoder",
                        "no route labels are opened and no accuracy claim is produced"],
    }
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
