"""Trip-local metric projection with longitude/latitude input and output."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from typing import Iterable

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True, slots=True)
class LocalProjection:
    """Locally centred equirectangular projection, suitable for one city-scale trip."""

    lon0: float
    lat0: float

    @classmethod
    def from_coordinates(cls, coordinates: Iterable[tuple[float, float]]) -> "LocalProjection":
        pts = [(float(lon), float(lat)) for lon, lat in coordinates if _valid_lonlat(lon, lat)]
        if not pts:
            raise ValueError("cannot create a projection without valid longitude/latitude coordinates")
        return cls(sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def forward(self, lon: float, lat: float) -> tuple[float, float]:
        if not _valid_lonlat(lon, lat):
            raise ValueError(f"invalid longitude/latitude: {(lon, lat)!r}")
        x = EARTH_RADIUS_M * radians(lon - self.lon0) * cos(radians(self.lat0))
        y = EARTH_RADIUS_M * radians(lat - self.lat0)
        return x, y

    def inverse(self, x: float, y: float) -> tuple[float, float]:
        c = cos(radians(self.lat0))
        if abs(c) < 1e-12:
            raise ValueError("local equirectangular inverse is unstable near the poles")
        lon = self.lon0 + (x / (EARTH_RADIUS_M * c)) * 180.0 / 3.141592653589793
        lat = self.lat0 + (y / EARTH_RADIUS_M) * 180.0 / 3.141592653589793
        return lon, lat


def _valid_lonlat(lon: float, lat: float) -> bool:
    try:
        return -180 <= float(lon) <= 180 and -90 <= float(lat) <= 90
    except (TypeError, ValueError):
        return False


