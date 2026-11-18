"""Folium map builder for visualizing PathSnap API responses.

Renders only what the API returns:
- Raw GPS observations
- HMM / Viterbi predicted route drawn as separate polylines per connected
  segment (`hmm_route_segments`, falling back to legacy `hmm_route_polyline`
  only if `hmm_route_segments` is absent)
- Classifier-only predictions (optional toggle layer)
- Start / End markers, legend, and interactive LayerControl

All incoming coordinates must be `[lat, lon]` (WGS-84 degrees). Any point with
non-finite values, `lat` outside [-90, 90], or `lon` outside [-180, 180] is
skipped and counted so the UI can warn about coordinate-order issues.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import folium

from config import MAX_CLF_SEGMENTS_DISPLAY, MAX_GPS_MARKERS_DISPLAY


@dataclass
class MapRenderResult:
    fmap: folium.Map
    skipped_points: int
    hmm_segment_count: int


def _normalize_and_validate_point(pt: Any) -> tuple[float, float] | None:
    """Parse [lat, lon] or {'lat': ..., 'lon': ...} and validate WGS-84 bounds."""
    try:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            lat = float(pt[0])
            lon = float(pt[1])
        elif isinstance(pt, dict) and "lat" in pt and "lon" in pt:
            lat = float(pt["lat"])
            lon = float(pt["lon"])
        else:
            return None
    except (TypeError, ValueError):
        return None

    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return (lat, lon)


def _extract_points(items: list[Any] | None) -> tuple[list[tuple[float, float]], int]:
    """Return (valid_points, skipped_count) for a sequence of coordinates."""
    if not items:
        return [], 0
    valid: list[tuple[float, float]] = []
    skipped = 0
    for item in items:
        coord = _normalize_and_validate_point(item)
        if coord is None:
            skipped += 1
        else:
            valid.append(coord)
    return valid, skipped


def _extract_segments(raw_segments: list[Any] | None) -> tuple[list[list[tuple[float, float]]], int]:
    """Parse a list of polylines into validated segments without joining them."""
    if not raw_segments:
        return [], 0
    segments: list[list[tuple[float, float]]] = []
    skipped_total = 0
    for seg in raw_segments:
        if not isinstance(seg, (list, tuple)):
            skipped_total += 1
            continue
        pts, skipped = _extract_points(list(seg))
        skipped_total += skipped
        if len(pts) >= 2:
            segments.append(pts)
    return segments, skipped_total


def _subsample_indices(n: int, max_items: int) -> list[int]:
    if n <= max_items:
        return list(range(n))
    step = n / max_items
    indices = [int(i * step) for i in range(max_items)]
    if indices[-1] != n - 1:
        indices[-1] = n - 1
    return indices


def _legend_html(has_clf_layer: bool) -> str:
    clf_row = (
        """
        <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
          <span style="display:inline-block;width:18px;height:0;border-top:3px dashed #f97316;"></span>
          <span>Classifier-Only Route</span>
        </div>
        """
        if has_clf_layer
        else ""
    )
    return f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 14px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 12px;
        color: #0f172a;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.15);
    ">
      <div style="font-weight:700;margin-bottom:6px;font-size:13px;">PathSnap Map Legend</div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#2563eb;border:1px solid #1d4ed8;"></span>
        <span>Raw GPS Observations</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
        <span style="display:inline-block;width:18px;height:4px;background:#16a34a;border-radius:2px;"></span>
        <span>HMM / Viterbi Predicted Route</span>
      </div>
      {clf_row}
      <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#059669;"></span>
        <span>Start &nbsp;/&nbsp;</span>
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#dc2626;"></span>
        <span>End</span>
      </div>
    </div>
    """


def build_trip_map(prediction: dict[str, Any]) -> MapRenderResult:
    """Create a Folium map from a POST /predict response dictionary.

    Draws each segment in `hmm_route_segments` as its own separate PolyLine
    (never joining disconnected segments). Falls back to `hmm_route_polyline`
    only when `hmm_route_segments` is absent.
    """
    skipped_total = 0

    raw_gps, skipped = _extract_points(prediction.get("raw_gps"))
    skipped_total += skipped
    if not raw_gps and isinstance(prediction.get("observations"), list):
        raw_gps, skipped = _extract_points(prediction["observations"])
        skipped_total += skipped

    # Prefer hmm_route_segments; fall back to legacy hmm_route_polyline only if absent
    hmm_segments: list[list[tuple[float, float]]] = []
    if prediction.get("hmm_route_segments") is not None:
        hmm_segments, skipped = _extract_segments(prediction.get("hmm_route_segments"))
        skipped_total += skipped
    elif prediction.get("hmm_route_polyline") is not None:
        legacy_poly, skipped = _extract_points(prediction.get("hmm_route_polyline"))
        skipped_total += skipped
        if len(legacy_poly) >= 2:
            hmm_segments = [legacy_poly]

    clf_segments, skipped = _extract_segments(prediction.get("classifier_route_segments"))
    skipped_total += skipped

    hmm_flat_coords = [pt for seg in hmm_segments for pt in seg]
    all_coords = raw_gps + hmm_flat_coords
    if not all_coords:
        center = (0.0, 0.0)
        zoom = 2
    else:
        avg_lat = sum(c[0] for c in all_coords) / len(all_coords)
        avg_lon = sum(c[1] for c in all_coords) / len(all_coords)
        center = (avg_lat, avg_lon)
        zoom = 14

    fmap = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        prefer_canvas=True,
        control_scale=True,
    )

    # Layer 1: Raw GPS Observations
    if raw_gps:
        gps_group = folium.FeatureGroup(name="Raw GPS Observations", show=True)
        idxs = _subsample_indices(len(raw_gps), MAX_GPS_MARKERS_DISPLAY)
        for i in idxs:
            lat, lon = raw_gps[i]
            folium.CircleMarker(
                location=(lat, lon),
                radius=3,
                color="#2563eb",
                fill=True,
                fill_color="#3b82f6",
                fill_opacity=0.75,
                weight=1,
                tooltip=f"GPS obs #{i} ({lat:.5f}, {lon:.5f})",
            ).add_to(gps_group)
        gps_group.add_to(fmap)

    # Layer 2: Classifier-Only Predictions (optional toggle, off by default)
    has_clf_layer = False
    if clf_segments:
        clf_group = folium.FeatureGroup(name="Classifier-Only Predictions", show=False)
        seg_idxs = _subsample_indices(len(clf_segments), MAX_CLF_SEGMENTS_DISPLAY)
        for i in seg_idxs:
            seg = clf_segments[i]
            if len(seg) >= 2:
                has_clf_layer = True
                folium.PolyLine(
                    locations=seg,
                    color="#f97316",
                    weight=3,
                    opacity=0.75,
                    dash_array="6 6",
                ).add_to(clf_group)
        if has_clf_layer:
            clf_group.add_to(fmap)

    # Layer 3: HMM / Viterbi Predicted Route — draw EACH segment as its own PolyLine
    if hmm_segments:
        hmm_group = folium.FeatureGroup(name="HMM / Viterbi Predicted Route", show=True)
        trip_id = prediction.get("trip_id", "")
        for seg_idx, seg in enumerate(hmm_segments, start=1):
            folium.PolyLine(
                locations=seg,
                color="#16a34a",
                weight=5,
                opacity=0.9,
                tooltip=f"Predicted HMM Route ({trip_id}) — Segment {seg_idx}/{len(hmm_segments)}",
            ).add_to(hmm_group)
        hmm_group.add_to(fmap)

    # Layer 4: Start & End Markers
    start_pt = hmm_segments[0][0] if hmm_segments else (raw_gps[0] if len(raw_gps) >= 2 else None)
    end_pt = hmm_segments[-1][-1] if hmm_segments else (raw_gps[-1] if len(raw_gps) >= 2 else None)
    if start_pt is not None and end_pt is not None:
        markers_group = folium.FeatureGroup(name="Start & End Markers", show=True)
        folium.CircleMarker(
            location=start_pt,
            radius=7,
            color="#ffffff",
            fill=True,
            fill_color="#059669",
            fill_opacity=1.0,
            weight=2,
            tooltip="Trip Start",
        ).add_to(markers_group)
        folium.CircleMarker(
            location=end_pt,
            radius=7,
            color="#ffffff",
            fill=True,
            fill_color="#dc2626",
            fill_opacity=1.0,
            weight=2,
            tooltip="Trip End",
        ).add_to(markers_group)
        markers_group.add_to(fmap)

    # Auto-fit map bounds to the full trip extent
    if all_coords:
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(24, 24))

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_legend_html(has_clf_layer)))
    return MapRenderResult(
        fmap=fmap,
        skipped_points=skipped_total,
        hmm_segment_count=len(hmm_segments),
    )
