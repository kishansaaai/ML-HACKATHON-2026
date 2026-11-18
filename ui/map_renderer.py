"""Folium map builder for visualizing PathSnap API responses.

Renders only what the API returns:
- Raw GPS observations
- HMM / Viterbi predicted route (primary layer)
- Classifier-only predictions (optional comparison layer)
- Start / End markers, legend, and interactive LayerControl
"""

from __future__ import annotations

from typing import Any

import folium

from config import MAX_CLF_SEGMENTS_DISPLAY, MAX_GPS_MARKERS_DISPLAY


def _normalize_point(pt: Any) -> tuple[float, float] | None:
    """Convert [lat, lon] or {'lat': ..., 'lon': ...} into (lat, lon)."""
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        return (float(pt[0]), float(pt[1]))
    if isinstance(pt, dict) and "lat" in pt and "lon" in pt:
        return (float(pt["lat"]), float(pt["lon"]))
    return None


def _extract_points(items: list[Any] | None) -> list[tuple[float, float]]:
    if not items:
        return []
    out: list[tuple[float, float]] = []
    for item in items:
        coord = _normalize_point(item)
        if coord is not None:
            out.append(coord)
    return out


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


def build_trip_map(prediction: dict[str, Any]) -> folium.Map:
    """Create a Folium map from a POST /predict response dictionary."""
    raw_gps = _extract_points(prediction.get("raw_gps"))
    if not raw_gps and isinstance(prediction.get("observations"), list):
        raw_gps = _extract_points(prediction["observations"])

    hmm_polyline = _extract_points(prediction.get("hmm_route_polyline"))
    clf_segments_raw = prediction.get("classifier_route_segments") or []

    all_coords = raw_gps + hmm_polyline
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

    # Layer 2: Classifier-Only Predictions (optional toggle, off by default for clarity)
    has_clf_layer = False
    if clf_segments_raw:
        clf_group = folium.FeatureGroup(name="Classifier-Only Predictions", show=False)
        seg_idxs = _subsample_indices(len(clf_segments_raw), MAX_CLF_SEGMENTS_DISPLAY)
        for i in seg_idxs:
            seg = _extract_points(clf_segments_raw[i])
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

    # Layer 3: HMM / Viterbi Predicted Route (primary layer)
    if len(hmm_polyline) >= 2:
        hmm_group = folium.FeatureGroup(name="HMM / Viterbi Predicted Route", show=True)
        folium.PolyLine(
            locations=hmm_polyline,
            color="#16a34a",
            weight=5,
            opacity=0.9,
            tooltip=f"Predicted HMM Route ({prediction.get('trip_id', '')})",
        ).add_to(hmm_group)
        hmm_group.add_to(fmap)

    # Layer 4: Start & End Markers
    endpoint_source = hmm_polyline if len(hmm_polyline) >= 2 else raw_gps
    if len(endpoint_source) >= 2:
        markers_group = folium.FeatureGroup(name="Start & End Markers", show=True)
        folium.CircleMarker(
            location=endpoint_source[0],
            radius=7,
            color="#ffffff",
            fill=True,
            fill_color="#059669",
            fill_opacity=1.0,
            weight=2,
            tooltip="Trip Start",
        ).add_to(markers_group)
        folium.CircleMarker(
            location=endpoint_source[-1],
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
    return fmap
