"""Raw trip loading.

Interim implementation of Person C's `load_trip` contract so the ML pipeline can run;
C may replace the internals as long as the `Trip` fields and signature stay the same.
`.route` (ground truth) is read only when `with_labels=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray                 # seconds, file order
    node_lonlat: np.ndarray       # (N, 2) lon, lat; row index = node id
    arcs: np.ndarray              # (M, 2) int from_node, to_node; row index = arc_id
    route: np.ndarray | None      # ground-truth arc_id sequence, only when with_labels=True
    trip_dir: str = ""

    @property
    def n_obs(self) -> int:
        return len(self.lon)


def _read(path: Path, dtype) -> np.ndarray:
    # Files are whitespace separated, no header. pandas' C parser is much faster than loadtxt.
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype)
    return df.to_numpy()


def load_trip(trip_dir: str | Path, with_labels: bool = False) -> Trip:
    d = Path(trip_dir)
    tid = d.name
    track = _read(d / f"{tid}.track", np.float64)
    nodes = _read(d / f"{tid}.nodes", np.float64)
    arcs = _read(d / f"{tid}.arcs", np.int64)
    route = None
    if with_labels:
        rpath = d / f"{tid}.route"
        route = _read(rpath, np.int64)[:, 0] if rpath.stat().st_size > 0 else np.zeros(0, np.int64)
    return Trip(trip_id=tid, lon=track[:, 0], lat=track[:, 1], t=track[:, 2],
                node_lonlat=nodes, arcs=arcs, route=route, trip_dir=str(d))
