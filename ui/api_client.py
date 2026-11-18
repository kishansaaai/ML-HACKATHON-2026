"""HTTP client for communicating with the PathSnap FastAPI backend on Laptop 1.

This module performs zero ML inference and never accesses local dataset or
ground-truth `.route` files. It only calls `/health`, `/trips`, and `/predict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import requests

from config import HEALTH_TIMEOUT_SEC, PREDICT_TIMEOUT_SEC, TRIPS_TIMEOUT_SEC


class APIError(Exception):
    """Readable error surfaced to the evaluator UI when the API fails."""


@dataclass
class TripSummary:
    trip_id: str
    num_observations: int | None = None
    num_nodes: int | None = None
    num_arcs: int | None = None


def _clean_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if not url:
        raise APIError("API Base URL is empty. Enter Laptop 1's URL (e.g. http://192.168.1.10:8000).")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url


def _extract_error_detail(resp: requests.Response) -> str:
    try:
        payload = resp.json()
        if isinstance(payload, dict) and "detail" in payload:
            return str(payload["detail"])
    except ValueError:
        pass
    text = resp.text.strip()
    return text[:300] if text else f"HTTP {resp.status_code}"


def check_health(base_url: str) -> dict[str, Any]:
    """Call GET /health and return parsed status payload with measured latency."""
    url = f"{_clean_base_url(base_url)}/health"
    t0 = time.perf_counter()
    try:
        resp = requests.get(url, timeout=HEALTH_TIMEOUT_SEC)
    except requests.exceptions.ConnectionError as exc:
        raise APIError(
            f"Cannot reach API at {url}. Check that Laptop 1's FastAPI server is running, "
            "bound to 0.0.0.0, and allowed through the firewall."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise APIError(f"Health check timed out after {HEALTH_TIMEOUT_SEC:.0f}s ({url}).") from exc
    except requests.exceptions.RequestException as exc:
        raise APIError(f"Health check failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000.0
    if not resp.ok:
        raise APIError(f"API health check returned HTTP {resp.status_code}: {_extract_error_detail(resp)}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise APIError(f"API returned invalid JSON from /health.") from exc

    if not isinstance(data, dict):
        raise APIError("Unexpected /health payload format (expected a JSON object).")

    data["_latency_ms"] = round(latency_ms, 1)
    return data


def fetch_trips(base_url: str) -> tuple[str, list[TripSummary]]:
    """Call GET /trips and return (partition_name, list[TripSummary])."""
    url = f"{_clean_base_url(base_url)}/trips"
    try:
        resp = requests.get(url, timeout=TRIPS_TIMEOUT_SEC)
    except requests.exceptions.ConnectionError as exc:
        raise APIError(
            f"Cannot connect to {url} to load evaluation trips. Verify Laptop 1 IP and server status."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise APIError(f"Loading trip list timed out after {TRIPS_TIMEOUT_SEC:.0f}s.") from exc
    except requests.exceptions.RequestException as exc:
        raise APIError(f"Failed to load trips: {exc}") from exc

    if not resp.ok:
        raise APIError(f"GET /trips failed (HTTP {resp.status_code}): {_extract_error_detail(resp)}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise APIError("API returned invalid JSON from /trips.") from exc

    partition = "evaluation"
    raw_trips: list[Any]
    if isinstance(data, list):
        raw_trips = data
    elif isinstance(data, dict):
        partition = str(data.get("partition", "evaluation"))
        raw_trips = data.get("trips", [])
    else:
        raise APIError("Unexpected /trips response structure.")

    trips: list[TripSummary] = []
    for item in raw_trips:
        if isinstance(item, str):
            trips.append(TripSummary(trip_id=item))
        elif isinstance(item, dict) and "trip_id" in item:
            trips.append(
                TripSummary(
                    trip_id=str(item["trip_id"]),
                    num_observations=item.get("num_observations"),
                    num_nodes=item.get("num_nodes"),
                    num_arcs=item.get("num_arcs"),
                )
            )
    return partition, trips


def run_prediction(base_url: str, trip_id: str) -> dict[str, Any]:
    """Send POST /predict with {'trip_id': trip_id} and return the API JSON response."""
    url = f"{_clean_base_url(base_url)}/predict"
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            url,
            json={"trip_id": str(trip_id)},
            timeout=PREDICT_TIMEOUT_SEC,
        )
    except requests.exceptions.ConnectionError as exc:
        raise APIError(
            f"Lost connection to {url} while running trip {trip_id}. "
            "Check Laptop 1 terminal logs."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise APIError(
            f"Prediction for trip {trip_id} timed out after {PREDICT_TIMEOUT_SEC:.0f}s."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise APIError(f"Prediction request failed: {exc}") from exc

    roundtrip_sec = time.perf_counter() - t0
    if not resp.ok:
        raise APIError(
            f"POST /predict failed for trip {trip_id} (HTTP {resp.status_code}): "
            f"{_extract_error_detail(resp)}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise APIError("API returned invalid JSON from /predict.") from exc

    if not isinstance(data, dict):
        raise APIError("Unexpected /predict response format (expected a JSON object).")

    data["_client_roundtrip_sec"] = round(roundtrip_sec, 3)
    return data
