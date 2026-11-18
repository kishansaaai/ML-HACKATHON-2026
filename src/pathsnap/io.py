"""Raw-trip loader implementing Person A's shared ``Trip`` contract.

``.route`` is read only when ``with_labels=True``. Prediction callers must use
the default so the label file is never opened on the inference path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trip:
    trip_id: str
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray
    node_lonlat: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None
    trip_dir: str = ""

    @property
    def n_obs(self) -> int:
        return len(self.lon)


def _read(path: Path, dtype, columns: int | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        values = np.loadtxt(path, dtype=dtype, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"could not parse whitespace-delimited data in {path}") from exc
    if values.size == 0:
        return np.empty((0, columns or 0), dtype=dtype)
    if values.ndim != 2 or (columns is not None and values.shape[1] != columns):
        raise ValueError(f"{path}: expected {columns} columns, found shape {values.shape}")
    return values


def load_trip(trip_dir: str | Path, with_labels: bool = False) -> Trip:
    """Load raw arrays for exactly one trip; route labels are opt-in."""
    directory = Path(trip_dir)
    trip_id = directory.name
    track = _read(directory / f"{trip_id}.track", np.float64, 3)
    nodes = _read(directory / f"{trip_id}.nodes", np.float64, 2)
    arcs = _read(directory / f"{trip_id}.arcs", np.int64, 2)
    if len(arcs) and ((arcs < 0).any() or (arcs >= len(nodes)).any()):
        raise ValueError(f"{directory}: an arc endpoint is outside the node index range")

    route = None
    if with_labels:
        route_rows = _read(directory / f"{trip_id}.route", np.int64, 1)
        route = route_rows[:, 0]
        if len(route) and ((route < 0).any() or (route >= len(arcs)).any()):
            raise ValueError(f"{directory}: route contains an invalid arc_id")

    return Trip(
        trip_id=trip_id,
        lon=track[:, 0],
        lat=track[:, 1],
        t=track[:, 2],
        node_lonlat=nodes,
        arcs=arcs,
        route=route,
        trip_dir=str(directory),
    )
