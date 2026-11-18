"""Unit tests for Person A's pipeline on a tiny synthetic network (no dataset needed).

Network: a main road (nodes 0-4, two-way) and a parallel one-way street ~55 m north (nodes 5-9).
The vehicle drives east along the main road with a few metres of GPS noise.
"""
import numpy as np
import pytest

from pathsnap.cand_metrics import observation_metrics, route_continuity
from pathsnap.candidates import CandidateConfig, build_candidates
from pathsnap.features import FEATURE_ORDER
from pathsnap.geo import TripGraph, project_lonlat, unproject_xy
from pathsnap.io import Trip
from pathsnap.labels import derive_point_labels


def make_trip(with_route=True, seed=0):
    rng = np.random.default_rng(seed)
    lon0, lat0, dlon = 10.0, 50.0, 0.002
    main = np.array([[lon0 + i * dlon, lat0] for i in range(5)])
    par = np.array([[lon0 + i * dlon, lat0 + 0.0005] for i in range(5)])
    nodes = np.vstack([main, par])
    arcs = [(i, i + 1) for i in range(4)] + [(i + 1, i) for i in range(4)] + [(5 + i, 6 + i) for i in range(4)]
    arcs = np.array(arcs)                         # arc 0..3 main east, 4..7 main west, 8..11 parallel east
    # 1 Hz track along the main road at ~14 m/s
    total_deg = 4 * dlon
    n = 40
    lon = lon0 + np.linspace(0.0001, total_deg - 0.0001, n)
    lat = lat0 + rng.normal(0, 3e-5, n)           # ~3 m noise
    return Trip("synthetic", lon, lat, np.arange(1.0, n + 1), nodes, arcs,
                np.array([0, 1, 2, 3]) if with_route else None)


def test_projection_roundtrip():
    lon, lat = np.array([10.01, 10.02]), np.array([50.03, 50.04])
    xy = project_lonlat(lon, lat, 10.0, 50.0)
    back = unproject_xy(xy, 10.0, 50.0)
    assert np.allclose(back[:, 0], lon) and np.allclose(back[:, 1], lat)


def test_candidates_contain_true_arc_and_both_directions():
    trip = make_trip()
    g = TripGraph(trip)
    cand = build_candidates(trip, g, CandidateConfig(radius_m=80, k_min=3, max_cand=10))
    assert list(cand.columns[3:]) == FEATURE_ORDER
    assert not cand[FEATURE_ORDER].isna().any().any()
    mid = cand[cand.obs_idx == 20]
    assert {2, 6}.issubset(set(mid.arc_id))              # eastbound and westbound main-road arcs both offered
    assert mid.iloc[0].perp < 10                           # nearest candidate is on the main road
    assert (cand.groupby("obs_idx").size() <= 10).all()


def test_heading_feature_separates_directions():
    trip = make_trip()
    g = TripGraph(trip)
    cand = build_candidates(trip, g, CandidateConfig(radius_m=80, k_min=3, max_cand=10))
    c = cand[(cand.obs_idx == 20)].set_index("arc_id")
    assert c.loc[2, "head_absdiff"] < 20 and c.loc[6, "head_absdiff"] > 160


def test_labels_recover_route_monotonically():
    trip = make_trip()
    g = TripGraph(trip)
    ref = derive_point_labels(trip, g)
    assert (ref >= 0).all()
    assert (np.diff(ref) >= 0).all()                      # monotone along the route
    assert set(ref) <= {0, 1, 2, 3}
    assert ref[0] == 0 and ref[-1] == 3


def test_labels_need_route():
    trip = make_trip(with_route=False)
    with pytest.raises(ValueError):
        derive_point_labels(trip, TripGraph(trip))


def test_far_points_stay_unlabelled():
    trip = make_trip()
    trip.lat = trip.lat + 0.01                            # ~1.1 km north of every road
    ref = derive_point_labels(trip, TripGraph(trip))
    assert (ref == -1).all()


def test_metrics_top1_and_continuity():
    trip = make_trip()
    g = TripGraph(trip)
    ref = derive_point_labels(trip, g)
    cand = build_candidates(trip, g, CandidateConfig(radius_m=80, k_min=3, max_cand=10))
    cand["label"] = (cand["arc_id"].to_numpy() == ref[cand["obs_idx"].to_numpy()]).astype("int8")
    import pandas as pd
    obs = pd.DataFrame({"trip_id": trip.trip_id, "obs_idx": np.arange(trip.n_obs), "ref_arc": ref})
    perfect = observation_metrics(cand, cand["label"].to_numpy().astype(float), obs)
    assert perfect["top1"] == 1.0 and perfect["candidate_miss_rate"] == 0.0
    assert route_continuity(np.array([0, 0, 1, 2, 3]), trip.arcs) == (1.0, 0)
    cont, breaks = route_continuity(np.array([0, 2, 3]), trip.arcs)   # 0 -> 2 is not connected
    assert breaks == 1 and cont == 0.5


def test_score_candidates_uses_saved_artifacts():
    from pathsnap.clf import normalise_per_obs, score_candidates
    from pathsnap.paths import ARTIFACT_DIR
    if not (ARTIFACT_DIR / "model.joblib").exists():
        pytest.skip("model artifacts not exported yet")
    trip = make_trip(with_route=False)
    cand = build_candidates(trip, TripGraph(trip), CandidateConfig(radius_m=80, k_min=3, max_cand=10))
    p = score_candidates(cand)
    assert p.shape == (len(cand),) and ((p >= 0) & (p <= 1)).all()
    q = normalise_per_obs(cand, p)
    assert np.allclose(q.sum() , cand.obs_idx.nunique())  # each point's probabilities sum to 1
    # on this easy network the eastbound main-road arc should usually win at mid-trip points
    top = cand.assign(p=p).sort_values("p").groupby("obs_idx").tail(1)
    assert (top[top.obs_idx.between(10, 30)].arc_id.isin([0, 1, 2, 3])).mean() > 0.8
