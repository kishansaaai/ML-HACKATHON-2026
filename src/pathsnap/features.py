"""Feature engineering per (observation, candidate arc) pair.

No labels, trip ids or evaluator information are used here; the same code path builds the
features for training, evaluation and live inference. Distances are metres, angles degrees.

Feature groups (kept for the ablation study):
  geometry   : perp, frac, along_m, rem_m, arc_len
  relative   : dist_min, dist_gap, rank_dist, n_cand
  topology   : two_way, deg_out_end, deg_in_start
  motion     : speed, moving, head_absdiff, head_cos, dt_prev, dt_next
  neighbours : perp_prev1/next1/prev3/next3, perp_win, win_gap, rank_win, along_prev1/next1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import TripGraph, _rank_in_sorted_groups

FEATURE_GROUPS = {
    "geometry": ["perp", "frac", "along_m", "rem_m", "arc_len"],
    "relative": ["dist_min", "dist_gap", "rank_dist", "n_cand"],
    "topology": ["two_way", "deg_out_end", "deg_in_start"],
    "motion": ["speed", "moving", "head_absdiff", "head_cos", "dt_prev", "dt_next"],
    "neighbours": ["perp_prev1", "perp_next1", "perp_prev3", "perp_next3", "perp_win",
                   "win_gap", "rank_win", "along_prev1", "along_next1"],
}
FEATURE_ORDER = [f for g in FEATURE_GROUPS.values() for f in g]

PERP_SENTINEL = 100.0     # perp distance used when a neighbour point is unavailable
MAX_NBR_GAP_S = 10.0      # neighbours further than this in time are treated as unavailable
STILL_SPEED = 1.0         # m/s below which heading is considered undefined


def _motion(graph: TripGraph):
    """Per-point speed (m/s) and heading (radians, math convention); heading NaN if undefined."""
    xy, t = graph.obs_xy, graph.t
    K = len(xy)
    idx = np.arange(K)
    speed = np.zeros(K)
    head = np.full(K, np.nan)
    for w in (2, 1):                                   # prefer a 5-point window, fall back to 3
        a, b = np.clip(idx - w, 0, K - 1), np.clip(idx + w, 0, K - 1)
        dt = t[b] - t[a]
        ok = (dt > 0) & (dt <= 2 * w * 4.0) & np.isnan(head)
        v = xy[b] - xy[a]
        sp = np.where(dt > 0, np.hypot(v[:, 0], v[:, 1]) / np.where(dt > 0, dt, 1), 0.0)
        speed = np.where(ok, sp, speed)
        h = np.arctan2(v[:, 1], v[:, 0])
        head = np.where(ok & (sp >= STILL_SPEED), h, head)
    return speed, head


def _neighbour(graph: TripGraph, obs, arc, off, own_along):
    """Perp distance and signed along-arc offset of point (obs+off) projected onto candidate arc."""
    K = len(graph.obs_xy)
    nb = np.clip(obs + off, 0, K - 1)
    dt = np.abs(graph.t[nb] - graph.t[obs])
    valid = (nb != obs) & (dt <= MAX_NBR_GAP_S * abs(off))
    perp, frac, _ = graph.project_on_arcs(graph.obs_xy[nb], arc)
    along = frac * graph.arc_len[arc] - own_along
    return (np.where(valid, perp, PERP_SENTINEL),
            np.where(valid, np.clip(along, -100.0, 100.0), 0.0))


def compute_features(graph: TripGraph, obs, arc, perp, frac) -> pd.DataFrame:
    """Feature table for candidate pairs sorted by (obs, perp). Returns float32 columns."""
    n = len(obs)
    K = len(graph.obs_xy)
    new = np.r_[True, obs[1:] != obs[:-1]] if n else np.zeros(0, bool)
    starts = np.flatnonzero(new)
    sizes = np.diff(np.r_[starts, n])

    arc_len = graph.arc_len[arc]
    along = frac * arc_len
    f = {"perp": perp, "frac": frac, "along_m": along, "rem_m": (1.0 - frac) * arc_len, "arc_len": arc_len}

    dist_min = np.repeat(np.minimum.reduceat(perp, starts), sizes)
    f["dist_min"] = dist_min
    f["dist_gap"] = perp - dist_min
    f["rank_dist"] = _rank_in_sorted_groups(obs).astype(float)
    f["n_cand"] = np.repeat(sizes, sizes).astype(float)

    f["two_way"] = graph.two_way[arc].astype(float)
    f["deg_out_end"] = graph.out_deg[graph.arcs[arc, 1]].astype(float)
    f["deg_in_start"] = graph.in_deg[graph.arcs[arc, 0]].astype(float)

    speed, head = _motion(graph)
    sp, hd = speed[obs], head[obs]
    moving = ~np.isnan(hd)
    diff = np.where(moving, hd - graph.arc_bearing[arc], 0.0)
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    f["speed"] = np.clip(sp, 0, 60)
    f["moving"] = moving.astype(float)
    f["head_absdiff"] = np.where(moving, np.degrees(np.abs(diff)), 90.0)   # neutral when undefined
    f["head_cos"] = np.where(moving, np.cos(diff), 0.0)
    t = graph.t
    f["dt_prev"] = np.clip(np.where(obs > 0, t[obs] - t[np.maximum(obs - 1, 0)], 0.0), 0, 60)
    f["dt_next"] = np.clip(np.where(obs < K - 1, t[np.minimum(obs + 1, K - 1)] - t[obs], 0.0), 0, 60)

    pp1, ap1 = _neighbour(graph, obs, arc, -1, along)
    pn1, an1 = _neighbour(graph, obs, arc, +1, along)
    pp3, _ = _neighbour(graph, obs, arc, -3, along)
    pn3, _ = _neighbour(graph, obs, arc, +3, along)
    pp2, _ = _neighbour(graph, obs, arc, -2, along)
    pn2, _ = _neighbour(graph, obs, arc, +2, along)
    f["perp_prev1"], f["perp_next1"], f["perp_prev3"], f["perp_next3"] = pp1, pn1, pp3, pn3
    win = (perp + pp1 + pn1 + pp2 + pn2 + pp3 + pn3) / 7.0
    f["perp_win"] = win
    f["win_gap"] = win - np.repeat(np.minimum.reduceat(win, starts), sizes)
    order = np.lexsort((win, obs))                      # rank by neighbourhood support within obs
    rk = np.empty(n)
    rk[order] = _rank_in_sorted_groups(obs[order])
    f["rank_win"] = rk
    f["along_prev1"], f["along_next1"] = ap1, an1

    return pd.DataFrame({k: np.asarray(f[k], dtype=np.float32) for k in FEATURE_ORDER})

