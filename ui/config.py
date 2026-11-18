"""Single-source configuration for the PathSnap Evaluator UI (Laptop 2)."""

import os

# Change this IP (or set the PATHSNAP_API_URL environment variable) to point
# Laptop 2 at Laptop 1 running the FastAPI backend on port 8000.
DEFAULT_API_BASE_URL: str = os.getenv(
    "PATHSNAP_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

# HTTP request timeouts (seconds)
HEALTH_TIMEOUT_SEC: float = float(os.getenv("PATHSNAP_HEALTH_TIMEOUT", "5"))
TRIPS_TIMEOUT_SEC: float = float(os.getenv("PATHSNAP_TRIPS_TIMEOUT", "10"))
PREDICT_TIMEOUT_SEC: float = float(os.getenv("PATHSNAP_PREDICT_TIMEOUT", "120"))

# Visual rendering limits to keep Leaflet smooth on large trips
MAX_GPS_MARKERS_DISPLAY: int = int(os.getenv("PATHSNAP_MAX_GPS_MARKERS", "2000"))
MAX_CLF_SEGMENTS_DISPLAY: int = int(os.getenv("PATHSNAP_MAX_CLF_SEGMENTS", "1500"))
