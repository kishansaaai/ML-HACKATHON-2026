"""A + B integration: the real adapter, shared projection, metric agreement, no label access.

Uses the synthetic network from test_ml_pipeline, written to disk in the dataset's file format
(whitespace separated, no header), with NO `.route` file present.
"""
import numpy as np
import pytest

from pathsnap.cand_metrics import distinct_arc_continuity, route_continuity as a_continuity
from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.graph import load_road_graph
from pathsnap.decoder.inference import _get_graph, predict_trip
from pathsnap.decoder.metrics import observation_top1, route_continuity as b_continuity
from pathsnap.decoder.projection import LocalProjection
from pathsnap.geo import TripGraph, project_lonlat
from pathsnap.io import load_trip
from pathsnap.paths import ARTIFACT_DIR
from pathsnap.pipeline import make_components, predict
from test_ml_pipeline import make_trip

needs_model = pytest.mark.skipif(not (ARTIFACT_DIR / "model.joblib").exists(), reason="model not exported")


def write_trip(root, with_route=False):
    trip = make_trip(with_route=with_route)
    d = root / "synthetic"
    d.mkdir()
    np.savetxt(d / "synthetic.track", np.column_stack([trip.lon, trip.lat, trip.t]), fmt="%.8f")
    np.savetxt(d / "synthetic.nodes", trip.node_lonlat, fmt="%.8f")
    np.savetxt(d / "synthetic.arcs", trip.arcs, fmt="%d")
    return d, trip


@needs_model
def test_predict_without_any_route_file(tmp_path):
    d, trip = write_trip(tmp_path)
    assert not list(d.glob("*.route"))                       # leakage check: nothing to read
    out = predict(d)
    assert len(out["observations"]) == trip.n_obs
    mid = [a for a in out["hmm_arc_ids"][8:32] if a is not None]
    assert mid and set(mid) <= {0, 1, 2, 3}                  # eastbound main road, not the reverse arcs
    assert set(out["route_arc_ids"]) >= {0, 1, 2, 3}
    assert out["route_breaks"] == []


def test_decoder_and_classifier_share_one_projection(tmp_path):
    d, _ = write_trip(tmp_path)
    trip = load_trip(d)                                      # what the adapter sees (file precision)
    tgraph = TripGraph(trip)
    comps = make_components(d) if (ARTIFACT_DIR / "model.joblib").exists() else None
    proj = LocalProjection(tgraph.lon0, tgraph.lat0)
    if comps is not None:
        got = comps.projection_factory([], [])
        assert (got.lon0, got.lat0) == (tgraph.lon0, tgraph.lat0)
    pts = np.column_stack([trip.lon, trip.lat])
    ours = project_lonlat(pts[:, 0], pts[:, 1], tgraph.lon0, tgraph.lat0)
    theirs = np.array([proj.forward(lon, lat) for lon, lat in pts])
    assert np.allclose(ours, theirs, atol=1e-6)              # same formula, same origin -> same metres


def test_graph_cache_is_keyed_on_projection(tmp_path):
    d, _ = write_trip(tmp_path)
    cfg = DecoderConfig()
    g1 = _get_graph(d, cfg, LocalProjection(10.0, 50.0))
    g2 = _get_graph(d, cfg, LocalProjection(10.004, 50.0))
    assert g1 is not g2 and g1.nodes[0] != g2.nodes[0]


def test_official_continuity_matches_decoder_metric(tmp_path):
    d, trip = write_trip(tmp_path)
    graph = load_road_graph(d, DecoderConfig(), LocalProjection(10.0, 50.0))
    for seq in ([0, 0, 1, 2, 3, 3], [0, 1, 3, 3, 2], [3, 3, 3], [2, 0, 1]):
        frac, breaks = a_continuity(np.array(seq), trip.arcs)
        ref = b_continuity(seq, graph)
        assert breaks == ref["route_break_count"]
        assert frac == pytest.approx(ref["route_continuity"])
    # the old stricter variant is a different quantity and must never be mistaken for the official one
    assert distinct_arc_continuity(np.array([0, 0, 1, 3]), trip.arcs)[0] != a_continuity(np.array([0, 0, 1, 3]), trip.arcs)[0]


def test_top1_definition_matches():
    from pathsnap.cand_metrics import observation_metrics
    import pandas as pd
    # two points, two candidates each; the reference is the second point's lower-scored arc -> Top-1 = 0.5
    df = pd.DataFrame({"trip_id": "t", "obs_idx": [0, 0, 1, 1], "arc_id": [5, 6, 7, 8], "label": [1, 0, 0, 1]})
    obs = pd.DataFrame({"trip_id": "t", "obs_idx": [0, 1], "ref_arc": [5, 8]})
    ours = observation_metrics(df, np.array([0.9, 0.1, 0.8, 0.2]), obs)["top1"]
    assert ours == observation_top1([5, 7], [5, 8]) == 0.5


def test_shortest_path_early_exit_matches_full_search(tmp_path):
    d, _ = write_trip(tmp_path)
    g = load_road_graph(d, DecoderConfig(), LocalProjection(10.004, 50.0))
    for a in range(5):
        full_d, full_p = g._dijkstra(a, 10_000.0)
        for b in full_d:
            if b == a:
                continue
            dist, path = g.shortest_path(a, b, 10_000.0)
            assert dist == pytest.approx(full_d[b]) and path == g._backtrack(a, b, full_p)


def test_oof_scorer_covers_every_candidate_and_refuses_non_training_trips():
    from pathsnap.oof import OOF_PATH, oof_scorer
    if not OOF_PATH.exists():
        pytest.skip("run scripts/make_oof_scores.py first")
    from pathsnap.candidates import build_candidates
    from pathsnap.paths import DATA_ROOT
    d = DATA_ROOT / "train" / "00000001"
    trip = load_trip(d)
    table = build_candidates(trip, TripGraph(trip))
    p = oof_scorer("00000001")(table)
    assert p.shape == (len(table),) and ((p >= 0) & (p <= 1)).all()
    with pytest.raises(KeyError):
        oof_scorer("00000000")                                 # a student_test trip has no OOF scores
