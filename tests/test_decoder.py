import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.graph import Arc, Candidate, RoadGraph, load_road_graph
from pathsnap.decoder.hmm import emission_log_scores, transition_log_score, viterbi_decode
from pathsnap.decoder.inference import InferenceComponents, predict_trip
from pathsnap.decoder.metrics import edit_distance, observation_classification_metrics, observation_top1, route_sequence_metrics
from pathsnap.decoder.projection import LocalProjection
from pathsnap.decoder.stitch import stitch_route
from pathsnap.decoder.tuning import grouped_folds, tune_decoder_group_cv


def graph_fixture():
    projection = LocalProjection(0.0, 0.0)
    xy = [(0., 0.), (10., 0.), (20., 0.), (20., 10.), (80., 0.)]
    raw = [projection.inverse(*point) for point in xy]
    endpoints = [(0, 1), (1, 2), (2, 3), (3, 1), (4, 4)]
    arcs = {}
    for arc_id, (tail, head) in enumerate(endpoints):
        start, end = xy[tail], xy[head]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        arcs[arc_id] = Arc(arc_id, tail, head, start, end, length,
                           (min(start[0], end[0]), min(start[1], end[1]), max(start[0], end[0]), max(start[1], end[1])))
    return RoadGraph(xy, raw, arcs, projection, DecoderConfig(beta_m=20, transition_cutoff_m=200))


class ProjectionTests(unittest.TestCase):
    def test_round_trip_and_metric_scale(self):
        projection = LocalProjection(8.6, 49.8)
        xy = projection.forward(8.601, 49.801)
        lonlat = projection.inverse(*xy)
        self.assertAlmostEqual(lonlat[0], 8.601, places=8)
        self.assertAlmostEqual(lonlat[1], 49.801, places=8)
        self.assertGreater(xy[0], 60)
        with self.assertRaises(ValueError):
            projection.forward(181, 0)


class GraphTests(unittest.TestCase):
    def test_spatial_index_and_original_ids(self):
        graph = graph_fixture()
        found = graph.spatial_index.nearest((8, 1), 2)
        self.assertEqual(found[0].arc_id, 0)
        self.assertEqual(graph.arcs[0].arc_id, 0)

    def test_directed_shortest_path_and_components(self):
        graph = graph_fixture()
        self.assertEqual(graph.shortest_path(0, 3), (30.0, [0, 1, 2]))
        self.assertIsNone(graph.shortest_path(4, 0))
        self.assertTrue(graph.connected(0, 1))
        self.assertFalse(graph.connected(1, 0))

    def test_fractional_network_distance(self):
        graph = graph_fixture()
        a = Candidate(0, .5, 0, (5, 0))
        b = Candidate(1, .25, 0, (12.5, 0))
        self.assertAlmostEqual(graph.network_distance(a, b), 7.5)
        reverse = Candidate(0, .25, 0, (2.5, 0))
        self.assertIsNone(graph.network_distance(a, reverse, cutoff_m=100))
        self.assertIsNotNone(graph.network_distance(reverse, a, cutoff_m=100))

    def test_batched_transitions_match_individual_shortest_paths(self):
        graph = graph_fixture()
        left = [Candidate(0, .2, 0, (2, 0)), Candidate(3, .7, 0, (20, 3))]
        right = [Candidate(1, .3, 0, (13, 0)), Candidate(4, .5, 0, (80, 0)), Candidate(2, .8, 0, (20, 8))]
        batched = graph.network_distance_matrix(left, right)
        individual = [[graph.network_distance(a, b) for b in right] for a in left]
        self.assertEqual(batched, individual)

    def test_loader_preserves_row_id_and_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "x"
            d.mkdir()
            (d / "x.nodes").write_text("8.6\t49.8\n8.601\t49.8\n")
            (d / "x.arcs").write_text("0\t1\n1\t0\n")
            graph = load_road_graph(d)
            self.assertEqual(set(graph.arcs), {0, 1})
            self.assertEqual(graph.arc_endpoints(0), (0, 1))


class ViterbiTests(unittest.TestCase):
    def test_viterbi_matches_brute_force_for_variable_state_counts(self):
        graph = graph_fixture()
        cfg = graph.config
        sets = [
            [Candidate(0, .2, 2, (2, 0)), Candidate(3, .5, 5, (20, 5))],
            [Candidate(1, .1, 1, (11, 0)), Candidate(3, .6, 2, (20, 6)), Candidate(2, .1, 3, (20, 1))],
            [Candidate(2, .8, 1, (20, 8))],
        ]
        probs = [[.7, .3], [.2, .5, .3], [1.0]]
        times = [0, 1, 2]
        coords = [(2, 0), (12, 0), (20, 8)]
        actual = viterbi_decode(sets, probs, times, coords, graph, cfg)
        best_score, best_path = -math.inf, None
        for path in itertools.product(*sets):
            score = sum(emission_log_scores(row, p, cfg)[row.index(state)] for row, p, state in zip(sets, probs, path))
            for t in range(1, len(path)):
                gps = math.hypot(coords[t][0] - coords[t - 1][0], coords[t][1] - coords[t - 1][1])
                score += transition_log_score(path[t - 1], path[t], gps, graph, cfg)
            if score > best_score:
                best_score, best_path = score, path
        self.assertEqual(actual, list(best_path))

    def test_gaps_and_empty_candidates_restart_or_return_none(self):
        graph = graph_fixture()
        sets = [[Candidate(0, .1, 1, (1, 0))], [], [Candidate(2, .5, 0, (20, 5))], [Candidate(3, .1, 0, (20, 1))]]
        scores = [[1], [], [1], [1]]
        decoded = viterbi_decode(sets, scores, [0, 1, 1000, 1001], [(0, 0), None, (20, 5), (20, 1)], graph)
        self.assertEqual([x.arc_id if x else None for x in decoded], [0, None, 2, 3])

    def test_route_stitching_adds_path_and_reports_breaks(self):
        graph = graph_fixture()
        states = [Candidate(0, .5, 0, (5, 0)), Candidate(2, .5, 0, (20, 5)), Candidate(4, .5, 0, (80, 0))]
        stitched = stitch_route(states, graph)
        self.assertEqual(stitched.arc_ids, [0, 1, 2, 4])
        self.assertEqual(stitched.breaks[0]["reason"], "disconnected")
        self.assertEqual(len(graph.route_polyline(stitched.arc_ids)), 2)
        gap = stitch_route(states[:2], graph, [0, 1000], 10)
        self.assertEqual(gap.breaks[0]["reason"], "time_gap")


class MetricTests(unittest.TestCase):
    def test_route_and_alignment_metrics(self):
        self.assertEqual(edit_distance([1, 2, 3], [1, 3]), 1)
        self.assertEqual(observation_top1([1, 2, None], [1, 3, None]), .5)
        metrics = route_sequence_metrics([1, 2, 2], [1, 2, 3])
        self.assertAlmostEqual(metrics["route_arc_set_jaccard"], .6666666667)
        obs = observation_classification_metrics([1, None, 3], [1, 2, 2])
        self.assertAlmostEqual(obs["accuracy"], 1 / 3)
        self.assertIn("missing", obs["confusion_matrix"]["labels"])

    def test_group_folds_are_reproducible_and_disjoint(self):
        folds = grouped_folds([f"trip{i}" for i in range(10)], 5, 17)
        self.assertEqual(folds, grouped_folds([f"trip{i}" for i in range(10)], 5, 17))
        for train_ids, valid_ids in folds:
            self.assertFalse(set(train_ids) & set(valid_ids))
            self.assertEqual(len(valid_ids), 2)

    def test_decoder_search_records_every_fold(self):
        trips = {f"trip{i}": i for i in range(6)}
        config = DecoderConfig(beta_m=20)
        result = tune_decoder_group_cv(
            trips, [config],
            lambda cfg, trip: {"hmm_top1_accuracy": .5 + trip / 100,
                               "hmm_route_continuity": {"route_continuity": .8},
                               "candidate_log_loss": 1.0},
            n_splits=3, seed=2,
        )
        self.assertEqual(len(result["trials"][0]["folds"]), 3)
        self.assertEqual(result["best_config"]["beta_m"], 20)


class PredictionLeakageTests(unittest.TestCase):
    def test_inference_succeeds_when_reference_file_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            trip = Path(tmp) / "tripA"
            trip.mkdir()
            (trip / "tripA.nodes").write_text("8.6 49.8\n8.601 49.8\n8.602 49.8\n")
            (trip / "tripA.arcs").write_text("0 1\n1 2\n")
            (trip / "tripA.track").write_text("8.6001 49.8 1\n8.6018 49.8 2\n")
            def candidates(points, graph):
                return [[graph.spatial_index.nearest(point, 1)[0]] if point else [] for point in points]
            components = InferenceComponents(candidates, lambda obs, cands, graph: cands,
                                             lambda cands: [[1.0] if row else [] for row in cands])
            result = predict_trip(trip, components)
            json.dumps(result)
            self.assertEqual(result["trip_id"], "tripA")
            self.assertEqual(len(result["observations"]), 2)
            self.assertEqual(result["hmm_arc_ids"], [0, 1])
            self.assertFalse((trip / "tripA.route").exists())


if __name__ == "__main__":
    unittest.main()
