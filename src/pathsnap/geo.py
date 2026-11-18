"""Per-trip projection, arc geometry and spatial index.

Interim implementation of the `TripGraph` surface that Person A needs (projection, arc
geometry, nearest-arc query, point-on-arc projection). Person B owns this file long term
and adds the network-distance / shortest-path machinery on top; keep the names below stable.

Design choices
--------------
* Every trip has its own road network in its own region, so one graph per trip.
* Projection: local equirectangular around the track centroid, in metres. Over a trip's
  extent (tens of km at most) the distortion is far below GPS noise (< 0.1 %), it needs no
  zone selection, and it is trivially invertible. Everything is in this frame.
* Only arcs that can matter (bounding box of the track plus a margin) are indexed; some
  trips ship a network much larger than the track.
* Spatial index: a KD-tree over points sampled every `sample_spacing_m` along each arc.
  Any arc passing within R of a query point has a sample within R + spacing/2 of it, so a
  ball query with that radius followed by an exact point-to-segment distance is exact.
* Angles use the mathematical convention (radians from the +x/east axis, counter-clockwise)
  everywhere, for both arc bearings and vehicle headings, so differences are consistent.
"""
from __future__ import annotations

from itertools import chain

import numpy as np
from scipy.spatial import cKDTree

from .io import Trip

R_EARTH_M = 6_371_008.8


def project_lonlat(lon, lat, lon0: float, lat0: float) -> np.ndarray:
    kx = np.deg2rad(1.0) * R_EARTH_M * np.cos(np.deg2rad(lat0))
    ky = np.deg2rad(1.0) * R_EARTH_M
    return np.column_stack([(np.asarray(lon) - lon0) * kx, (np.asarray(lat) - lat0) * ky])


def unproject_xy(xy: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    kx = np.deg2rad(1.0) * R_EARTH_M * np.cos(np.deg2rad(lat0))
    ky = np.deg2rad(1.0) * R_EARTH_M
    return np.column_stack([xy[:, 0] / kx + lon0, xy[:, 1] / ky + lat0])  # lon, lat


def segment_project(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Project points p (n,2) onto segments a->b (n,2). Returns (dist, frac in [0,1], proj)."""
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    safe = np.where(denom > 0, denom, 1.0)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / safe, 0.0, 1.0)
    t = np.where(denom > 0, t, 0.0)
    proj = a + t[:, None] * ab
    d = np.hypot(*(p - proj).T)
    return d, t, proj


class TripGraph:
    def __init__(self, trip: Trip, margin_m: float = 600.0, sample_spacing_m: float = 10.0):
        self.trip_id = trip.trip_id
        self.lon0 = float(np.median(trip.lon))
        self.lat0 = float(np.median(trip.lat))
        self.obs_xy = project_lonlat(trip.lon, trip.lat, self.lon0, self.lat0)
        self.t = trip.t
        self.node_xy = project_lonlat(trip.node_lonlat[:, 0], trip.node_lonlat[:, 1], self.lon0, self.lat0)
        self.arcs = trip.arcs
        self.n_nodes = len(self.node_xy)
        self.n_arcs = len(self.arcs)

        self.arc_a = self.node_xy[self.arcs[:, 0]]
        self.arc_b = self.node_xy[self.arcs[:, 1]]
        d = self.arc_b - self.arc_a
        self.arc_len = np.hypot(d[:, 0], d[:, 1])
        self.arc_bearing = np.arctan2(d[:, 1], d[:, 0])

        # node degrees (used as features) and "has a reverse arc" (two-way road) flag
        self.out_deg = np.bincount(self.arcs[:, 0], minlength=self.n_nodes)
        self.in_deg = np.bincount(self.arcs[:, 1], minlength=self.n_nodes)
        n = np.int64(self.n_nodes)
        key = self.arcs[:, 0].astype(np.int64) * n + self.arcs[:, 1]
        rev_key = self.arcs[:, 1].astype(np.int64) * n + self.arcs[:, 0]
        self.two_way = np.isin(rev_key, key)

        self.spacing = float(sample_spacing_m)
        self._build_index(margin_m)

    # ------------------------------------------------------------------ index
    def _build_index(self, margin_m: float) -> None:
        lo = self.obs_xy.min(axis=0) - margin_m
        hi = self.obs_xy.max(axis=0) + margin_m
        amin = np.minimum(self.arc_a, self.arc_b)
        amax = np.maximum(self.arc_a, self.arc_b)
        active = np.flatnonzero((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) &
                                (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))
        self.active_arcs = active
        if len(active) == 0:
            self._tree = None
            self._sample_arc = np.zeros(0, np.int64)
            return
        n_samp = np.maximum(np.ceil(self.arc_len[active] / self.spacing).astype(np.int64) + 1, 2)
        total = int(n_samp.sum())
        arc_rep = np.repeat(active, n_samp)
        starts = np.cumsum(n_samp) - n_samp
        pos = np.arange(total) - np.repeat(starts, n_samp)
        t = pos / np.repeat(n_samp - 1, n_samp)
        a = self.arc_a[arc_rep]
        b = self.arc_b[arc_rep]
        pts = a + t[:, None] * (b - a)
        self._tree = cKDTree(pts)
        self._sample_arc = arc_rep

    # ------------------------------------------------------------- projection
    def project_on_arcs(self, xy: np.ndarray, arc_ids: np.ndarray):
        """Vectorised (perp_dist_m, frac_along 0..1, projected_xy) of points xy[i] on arc_ids[i]."""
        return segment_project(xy, self.arc_a[arc_ids], self.arc_b[arc_ids])

    def project_on_arc(self, xy, arc_id: int) -> tuple[float, float]:
        d, t, _ = self.project_on_arcs(np.asarray(xy, float)[None, :], np.array([arc_id]))
        return float(d[0]), float(t[0])

    # ------------------------------------------------------------- candidates
    def _ball_pairs(self, xy: np.ndarray, obs_ids: np.ndarray, radius_m: float):
        """Exact (obs, arc, perp, frac) pairs with perp <= radius for the given query points."""
        if self._tree is None or len(xy) == 0:
            e = np.zeros(0)
            return e.astype(np.int64), e.astype(np.int64), e, e
        lists = self._tree.query_ball_point(xy, r=radius_m + self.spacing / 2.0, workers=-1)
        counts = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
        if counts.sum() == 0:
            e = np.zeros(0)
            return e.astype(np.int64), e.astype(np.int64), e, e
        flat = np.fromiter(chain.from_iterable(lists), dtype=np.int64, count=int(counts.sum()))
        local = np.repeat(np.arange(len(xy)), counts)
        arc = self._sample_arc[flat]
        key = local.astype(np.int64) * np.int64(self.n_arcs) + arc
        key = np.unique(key)                      # one row per (point, arc)
        local = key // self.n_arcs
        arc = key % self.n_arcs
        perp, frac, _ = self.project_on_arcs(xy[local], arc)
        keep = perp <= radius_m
        return obs_ids[local[keep]], arc[keep], perp[keep], frac[keep]

    def query_candidates(self, xy: np.ndarray, radius_m: float, k_min: int, max_cand: int,
                         fallback_max_m: float = 400.0):
        """Candidate arcs for each query point.

        All arcs within `radius_m`; if a point has fewer than `k_min`, the radius is doubled
        (up to `fallback_max_m`) until `k_min` nearest arcs are found; each point keeps at most
        `max_cand` nearest arcs. Returns flat arrays (obs, arc, perp, frac) sorted by (obs, perp).
        """
        K = len(xy)
        obs, arc, perp, frac = self._ball_pairs(xy, np.arange(K), radius_m)
        counts = np.bincount(obs, minlength=K)
        r = radius_m
        while True:
            need = np.flatnonzero(counts < k_min)
            if len(need) == 0 or r >= fallback_max_m:
                break
            r = min(r * 2.0, fallback_max_m)
            o2, a2, p2, f2 = self._ball_pairs(xy[need], need, r)
            # these points are re-queried at the larger radius; keep the nearest k_min of them
            drop = np.isin(obs, need)
            obs, arc, perp, frac = obs[~drop], arc[~drop], perp[~drop], frac[~drop]
            order = np.lexsort((p2, o2))
            o2, a2, p2, f2 = o2[order], a2[order], p2[order], f2[order]
            rank = _rank_in_sorted_groups(o2)
            keep = rank < k_min
            obs = np.concatenate([obs, o2[keep]])
            arc = np.concatenate([arc, a2[keep]])
            perp = np.concatenate([perp, p2[keep]])
            frac = np.concatenate([frac, f2[keep]])
            counts = np.bincount(obs, minlength=K)
        order = np.lexsort((perp, obs))
        obs, arc, perp, frac = obs[order], arc[order], perp[order], frac[order]
        keep = _rank_in_sorted_groups(obs) < max_cand
        return obs[keep], arc[keep], perp[keep], frac[keep]

    def nearest_arcs(self, xy, radius_m: float, k_min: int, max_cand: int = 40) -> list[np.ndarray]:
        """Contract method: arc ids per query point (nearest first)."""
        xy = np.atleast_2d(np.asarray(xy, float))
        obs, arc, _, _ = self.query_candidates(xy, radius_m, k_min, max_cand)
        splits = np.cumsum(np.bincount(obs, minlength=len(xy)))[:-1]
        return np.split(arc, splits)

    def out_arcs(self, node_id: int) -> np.ndarray:
        return np.flatnonzero(self.arcs[:, 0] == node_id)


def _rank_in_sorted_groups(group: np.ndarray) -> np.ndarray:
    """0-based rank of each row inside its (contiguous, sorted) group."""
    if len(group) == 0:
        return np.zeros(0, np.int64)
    new = np.r_[True, group[1:] != group[:-1]]
    start = np.flatnonzero(new)
    sizes = np.diff(np.r_[start, len(group)])
    return np.arange(len(group)) - np.repeat(start, sizes)
