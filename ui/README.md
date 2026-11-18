# PathSnap Evaluator UI (Laptop 2 — Person D)

Lightweight **Streamlit + Folium** evaluator console for the two-laptop PathSnap deployment.

## Architecture Guarantee

- Sends **only** `trip_id` to Laptop 1's FastAPI backend (`GET /health`, `GET /trips`, `POST /predict`).
- Performs **zero** local ML inference or route decoding.
- Never reads `.track`, `.nodes`, `.arcs`, or `.route` files; ground-truth labels remain strictly server-side on Laptop 1.

## Setup (Laptop 2)

```bash
cd ui
pip install -r requirements.txt
```

## Run Against Laptop 1 (Evaluation / Production)

Pass Laptop 1's LAN IP via argument or `PATHSNAP_API_URL` (also editable live in the sidebar):

- **Windows:**
  ```bat
  run_ui.bat http://<LAPTOP_1_IP>:8000
  ```
- **macOS / Linux:**
  ```bash
  ./run_ui.sh http://<LAPTOP_1_IP>:8000
  ```

Open `http://localhost:8501` in the browser.

## Local Development Testing (Before Laptop 1 API Is Ready)

> **Dev only:** `mock_api.py` is a nearest-arc geometry stub for testing the UI against `Student folder/student_test` before Person C's real FastAPI backend is deployed. Do **not** use `mock_api.py` for final evaluation.

```bash
pip install -r requirements-dev.txt
```
1. Terminal 1: `run_mock_api.bat` (or `uvicorn mock_api:app --host 0.0.0.0 --port 8000`)
2. Terminal 2: `run_ui.bat http://127.0.0.1:8000`

## Related Docs

- [API Contract](../docs/API_CONTRACT.md)
- [Two-Laptop Demo Checklist](../docs/DEMO_CHECKLIST.md)
- [Report Section 7 Draft](../docs/REPORT_SECTION_7.md)
- [Person D Viva Prep](../docs/VIVA_PREP_PERSON_D.md)
