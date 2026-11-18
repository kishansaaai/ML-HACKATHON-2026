"""Derive a per-GPS-point reference arc from the trip's `.route`.

TRAINING / EVALUATION ONLY. Nothing on the prediction path may import this module.

The route is the chronological list of arcs the vehicle drove, with far fewer arcs than GPS
points, so there is no per-point label. We recover one with a monotone alignment:

    point i  ->  route arc j(i),   j(i) non-decreasing in i

minimising the sum over points of  min(perp_dist(point_i, route_arc_j), cap).
Dynamic programming over (point, route index). Moving forward along the route is allowed by
any number of arcs (a fast vehicle can cross several short arcs between two 1 s samples),
at a tiny per-arc cost `lam` that only breaks ties in favour of fewer jumps.

The prefix-minimum trick makes each DP step O(L) numpy work:
    dp_i[j] = cost_i[j] + min_{j' <= j} ( dp_{i-1}[j'] + lam * (j - j') )
            = cost_i[j] + lam*j + cummin_j( dp_{i-1}[j'] - lam*j' )
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import TripGraph, segment_project
from .io import Trip


@dataclass
class LabelStats:
    n_obs: int
    n_labelled: int
    frac_labelled: float
    dist_median_m: float
    dist_p90_m: float
    dist_p99_m: float
    frac_dist_gt_30m: float
    frac_ambiguous: float       # another, non-adjacent route arc is about as close as the assigned one
    route_len: int
    route_consecutive_frac: float


def _point_to_route_cost(p: np.ndarray, ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Distance from point p (2,) to every route segment (L,2)->(L,2)."""
    ab = rb - ra
    denom = np.einsum("ij,ij->i", ab, ab)
    safe = np.where(denom > 0, denom, 1.0)
    t = np.clip(((p - ra) * ab).sum(1) / safe, 0.0, 1.0)
    t = np.where(denom > 0, t, 0.0)
    q = ra + t[:, None] * ab
    return np.hypot(q[:, 0] - p[0], q[:, 1] - p[1])


def derive_point_labels(trip: Trip, graph: TripGraph, max_label_dist_m: float = 100.0,
                        cost_cap_m: float = 150.0, lam: float = 1e-3,
                        return_stats: bool = False):
    """Reference arc id per observation (shape (K,)), -1 where undefined.

    A point is left undefined (-1) when even its best aligned route arc is farther than
    `max_label_dist_m` (GPS dropout / the reference route not passing near the point).
    """
    if trip.route is None or len(trip.route) == 0:
        raise ValueError("derive_point_labels needs a trip loaded with with_labels=True")
    route = trip.route.astype(np.int64)
    L = len(route)
    K = graph.obs_xy.shape[0]
    ra, rb = graph.arc_a[route], graph.arc_b[route]
    jj = np.arange(L, dtype=np.float64)
    back_dtype = np.int16 if L < 32000 else np.int32
    back = np.empty((K, L), dtype=back_dtype)

    dp = np.minimum(_point_to_route_cost(graph.obs_xy[0], ra, rb), cost_cap_m)
    back[0] = jj.astype(back_dtype)
    for i in range(1, K):
        g = dp - lam * jj
        run = np.minimum.accumulate(g)
        idx = np.where(g == run, np.arange(L), 0)
        back[i] = np.maximum.accumulate(idx)
        cost = np.minimum(_point_to_route_cost(graph.obs_xy[i], ra, rb), cost_cap_m)
        dp = cost + run + lam * jj

    j = np.empty(K, dtype=np.int64)
    j[K - 1] = int(np.argmin(dp))
    for i in range(K - 1, 0, -1):
        j[i - 1] = back[i][j[i]]
    # (back[i][j] is the predecessor index of state j at step i)

    labels = route[j]
    # exact distance from each point to its assigned arc
    d_assigned, _, _ = segment_project(graph.obs_xy, graph.arc_a[labels], graph.arc_b[labels])
    labels = np.where(d_assigned <= max_label_dist_m, labels, -1)

    if not return_stats:
        return labels

    # ambiguity: does a route arc far away in route order lie about as close as the assigned one?
    amb = np.zeros(K, bool)
    for i in range(K):
        c = _point_to_route_cost(graph.obs_xy[i], ra, rb)
        far = np.abs(np.arange(L) - j[i]) > 5
        if far.any():
            amb[i] = bool((c[far] <= d_assigned[i] + 3.0).any())
    lab = labels >= 0
    cons = float(np.mean(graph.arcs[route[1:], 0] == graph.arcs[route[:-1], 1])) if L > 1 else 1.0
    stats = LabelStats(
        n_obs=K, n_labelled=int(lab.sum()), frac_labelled=float(lab.mean()),
        dist_median_m=float(np.median(d_assigned)), dist_p90_m=float(np.percentile(d_assigned, 90)),
        dist_p99_m=float(np.percentile(d_assigned, 99)), frac_dist_gt_30m=float((d_assigned > 30).mean()),
        frac_ambiguous=float(amb.mean()), route_len=L, route_consecutive_frac=cons)
    return labels, d_assigned, stats

