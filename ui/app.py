"""PathSnap Evaluator UI (Laptop 2 — Person D).

A lightweight Streamlit + Folium console that communicates strictly with the
PathSnap FastAPI backend on Laptop 1:
  - GET  /health   : verify backend readiness, model_name, and partition
  - GET  /trips    : fetch the evaluation trip IDs
  - POST /predict  : send {"trip_id": ...} and render returned route + metrics

This application performs zero ML inference and never accesses `.route` files.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from api_client import APIError, TripSummary, check_health, fetch_trips, run_prediction
from config import DEFAULT_API_BASE_URL
from map_renderer import build_trip_map


EVALUATOR_PARTITIONS = {"evaluator", "evaluator_test", "evaluator_testing", "eval"}

st.set_page_config(
    page_title="PathSnap — Evaluator Console",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --muted: rgba(128,128,128,0.95); --line: rgba(128,128,128,0.28); --soft: rgba(128,128,128,0.12); --brand:#0f9d8a; }
    [data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }
    .block-container { max-width: 1180px; padding-top: 3.6rem; padding-bottom: 3rem; }
    [data-testid="stHeader"] { background: transparent; }
    html, body, [class*="css"] { font-family: "Inter", "Segoe UI", system-ui, sans-serif; }
    h1 { font-size: 1.7rem !important; font-weight: 700 !important; letter-spacing: -0.02em; margin: 0 !important; padding: 0 !important; }
    h2, h3 { font-weight: 650 !important; letter-spacing: -0.01em; }
    [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h2 { font-size: 1rem !important; }
    .app-sub { color: var(--muted); font-size: 0.9rem; margin: 2px 0 12px 0; }
    .status-pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.78rem; font-weight:600;
        background: rgba(34,197,94,0.18); color: #22c55e; }
    .status-err { background: rgba(239,68,68,0.18); color: #ef4444; }
    .chip { display:inline-block; background:var(--soft); border:1px solid var(--line); color:inherit;
        padding:3px 11px; border-radius:999px; font-size:0.8rem; margin:0 8px 6px 0; }
    .chip b { font-weight:600; }
    .chip-warn { border-color: rgba(245,158,11,0.6); background: rgba(245,158,11,0.14); }
    .backend-down { padding:10px 14px; border-radius:8px; margin-bottom:14px; font-size:0.9rem;
        background: rgba(239,68,68,0.16); border:1px solid rgba(239,68,68,0.5); }
    .section-label { color:var(--muted); font-size:0.75rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; margin:18px 0 6px 0; }
    [data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.8rem; }
    [data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 700; }
    .stButton > button[kind="primary"] { background: var(--brand); border-color: var(--brand); color:#fff; font-weight: 600; }
    .stButton > button[kind="primary"]:hover { background:#0b7d6e; border-color:#0b7d6e; color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fmt_pct(val: Any) -> str:
    if val is None:
        return "N/A"
    return f"{float(val) * 100:.2f}%"


def _fmt_delta_pct(hmm_val: Any, clf_val: Any) -> str | None:
    if hmm_val is None or clf_val is None:
        return None
    diff = (float(hmm_val) - float(clf_val)) * 100.0
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.2f}% vs clf"


def _fmt_float(val: Any, digits: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{float(val):.{digits}f}"


def _fmt_int(val: Any) -> str:
    if val is None:
        return "N/A"
    return f"{int(val):,}"


def _is_mock_or_non_evaluator(health: dict[str, Any], fallback_partition: str | None) -> bool:
    model_name = str(health.get("model_name") or "")
    partition = str(health.get("partition") or fallback_partition or "").strip().lower()
    if "mock" in model_name.lower():
        return True
    if partition not in EVALUATOR_PARTITIONS:
        return True
    return False


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "api_base_url": DEFAULT_API_BASE_URL,
        "health_info": None,
        "health_error": None,
        "partition": None,
        "trips": [],
        "trips_error": None,
        "selected_trip_id": None,
        "last_prediction": None,
        "last_error": None,
        "session_scores": {},
        "bootstrapped": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _refresh_backend(api_url: str) -> None:
    st.session_state["health_error"] = None
    st.session_state["trips_error"] = None
    try:
        st.session_state["health_info"] = check_health(api_url)
    except APIError as exc:
        st.session_state["health_info"] = None
        st.session_state["health_error"] = str(exc)

    try:
        partition, trips = fetch_trips(api_url)
        st.session_state["partition"] = partition
        st.session_state["trips"] = trips
        if trips and st.session_state["selected_trip_id"] not in {t.trip_id for t in trips}:
            st.session_state["selected_trip_id"] = trips[0].trip_id
    except APIError as exc:
        st.session_state["trips"] = []
        st.session_state["trips_error"] = str(exc)


def _render_backend_header_banner() -> None:
    """Model and partition are always visible as small neutral labels; a red box only if the backend is down."""
    health = st.session_state["health_info"]
    health_err = st.session_state["health_error"]
    fallback_partition = st.session_state.get("partition")

    if health is None:
        if health_err:
            st.markdown(
                '<div class="backend-down"><strong>Backend unreachable.</strong> '
                "Cannot verify model or partition via <code>GET /health</code>.</div>",
                unsafe_allow_html=True,
            )
        return

    model = html.escape(str(health.get("model_name") or "unknown"))
    st.markdown(f'<span class="chip">Model <b>{model}</b></span>', unsafe_allow_html=True)


def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## Connection")
        api_url = st.text_input(
            "API address",
            value=st.session_state["api_base_url"],
            help="Laptop 1's LAN address during the evaluation, e.g. http://192.168.1.25:8000",
        ).strip()
        if api_url != st.session_state["api_base_url"]:
            st.session_state["api_base_url"] = api_url

        if st.button("Check connection", use_container_width=True):
            _refresh_backend(api_url)

        health = st.session_state["health_info"]
        health_err = st.session_state["health_error"]
        if health:
            st.markdown('<span class="status-pill status-ok">● Connected</span>', unsafe_allow_html=True)
            st.caption(
                f"{health.get('_latency_ms', '?')} ms · v{health.get('version', 'n/a')}  \n"
                f"Partition: **{health.get('partition', st.session_state.get('partition', 'unknown'))}**"
            )
            if not health.get("model_loaded", True):
                st.warning("API is reachable, but the model is not loaded.")
        elif health_err:
            st.markdown('<span class="status-pill status-err">● Unreachable</span>', unsafe_allow_html=True)
            st.error(health_err)
        else:
            st.caption("Press **Check connection** to contact Laptop 1.")

        with st.expander("How this works"):
            st.caption(
                "**Laptop 2 (this screen)** sends a trip id and draws the answer. "
                "**Laptop 1 (API)** runs the classifier and HMM, and scores the result against the "
                "reference route, which never leaves Laptop 1."
            )
    return api_url


def _trip_label(trip: TripSummary) -> str:
    extra = f" · {trip.num_observations:,} GPS points" if trip.num_observations is not None else ""
    return f"{trip.trip_id}{extra}"


def _render_trip_picker(trips: list[TripSummary]) -> bool:
    """Trip dropdown, detail chips and the run button. Returns True when Run was pressed."""
    ids = [t.trip_id for t in trips]
    by_id = {t.trip_id: t for t in trips}
    if st.session_state["selected_trip_id"] not in by_id:
        st.session_state["selected_trip_id"] = ids[0]

    st.markdown('<div class="section-label">Trip</div>', unsafe_allow_html=True)
    pick_col, run_col = st.columns([3, 1.2], vertical_alignment="bottom")
    with pick_col:
        selected = st.selectbox(
            "Trip",
            ids,
            index=ids.index(st.session_state["selected_trip_id"]),
            format_func=lambda trip_id: _trip_label(by_id[trip_id]),
            label_visibility="collapsed",
        )
        st.session_state["selected_trip_id"] = selected
    with run_col:
        run_clicked = st.button("Match route", type="primary", use_container_width=True)

    trip = by_id[selected]
    chips = []
    if trip.num_arcs is not None:
        chips.append(f"{trip.num_arcs:,} road arcs")
    prev = st.session_state["session_scores"].get(selected)
    if prev is not None:
        chips.append(f"last HMM Top-1 {_fmt_pct(prev)}")
    if chips:
        st.markdown("".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips), unsafe_allow_html=True)
    return run_clicked


def _render_kpis(prediction: dict[str, Any]) -> None:
    metrics = prediction.get("metrics") or {}
    clf = metrics.get("classifier_only") or {}
    hmm = metrics.get("hmm") or {}
    timings = prediction.get("timings_sec") or {}

    with st.container(border=True):
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Top-1 accuracy (HMM)", _fmt_pct(hmm.get("top1_accuracy")),
                  delta=_fmt_delta_pct(hmm.get("top1_accuracy"), clf.get("top1_accuracy")))
        breaks = hmm.get("continuity_breaks")
        k2.metric("Route continuity", _fmt_pct(hmm.get("route_continuity")),
                  delta=f"{breaks} breaks" if breaks is not None else None, delta_color="off")
        seq_key = "sequence_f1" if hmm.get("sequence_f1") is not None else "macro_f1"
        k3.metric("Sequence F1" if seq_key == "sequence_f1" else "Macro F1", _fmt_pct(hmm.get(seq_key)),
                  delta=_fmt_delta_pct(hmm.get(seq_key), clf.get(seq_key)))
        total_srv = timings.get("total")
        roundtrip = prediction.get("_client_roundtrip_sec")
        shown = total_srv if total_srv is not None else roundtrip
        k4.metric("Server time", f"{float(shown):.2f} s" if shown is not None else "N/A",
                  delta=f"round trip {roundtrip:.2f} s" if (total_srv is not None and roundtrip is not None) else None,
                  delta_color="off")
    st.caption("Deltas compare the HMM result with the classifier alone.")


def _render_comparison_tab(prediction: dict[str, Any]) -> None:
    metrics = prediction.get("metrics") or {}
    clf = metrics.get("classifier_only") or {}
    hmm = metrics.get("hmm") or {}
    rows = [
        ("Top-1 accuracy", _fmt_pct(clf.get("top1_accuracy")), _fmt_pct(hmm.get("top1_accuracy"))),
        ("Macro F1", _fmt_pct(clf.get("macro_f1")), _fmt_pct(hmm.get("macro_f1"))),
        ("Precision", _fmt_pct(clf.get("precision")), _fmt_pct(hmm.get("precision"))),
        ("Recall", _fmt_pct(clf.get("recall")), _fmt_pct(hmm.get("recall"))),
        ("Log loss", _fmt_float(clf.get("log_loss")), _fmt_float(hmm.get("log_loss"))),
        ("Route continuity", _fmt_pct(clf.get("route_continuity")), _fmt_pct(hmm.get("route_continuity"))),
        ("Continuity breaks", _fmt_int(clf.get("continuity_breaks")), _fmt_int(hmm.get("continuity_breaks"))),
        ("Sequence F1", _fmt_pct(clf.get("sequence_f1")), _fmt_pct(hmm.get("sequence_f1"))),
        ("Candidate miss rate", _fmt_pct(clf.get("candidate_miss_rate")), _fmt_pct(hmm.get("candidate_miss_rate"))),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Classifier only", "Classifier + HMM"]),
                 hide_index=True, use_container_width=True)


def _render_timings_tab(prediction: dict[str, Any]) -> None:
    timings = prediction.get("timings_sec") or {}
    if not timings:
        st.info("The API returned no per-stage timings.")
        return
    labels = {
        "load_and_index": "Load and spatial index", "candidates_and_features": "Candidates and features",
        "classifier_inference": "Classifier inference", "hmm_viterbi": "HMM / Viterbi decoding",
        "server_scoring": "Server-side scoring", "total": "Total",
    }
    rows = [(labels.get(k, k), f"{float(v):.3f} s") for k, v in timings.items() if v is not None]
    st.dataframe(pd.DataFrame(rows, columns=["Stage", "Duration"]), hide_index=True, use_container_width=True)
    st.caption(
        f"Trip {prediction.get('trip_id', 'N/A')} · {_fmt_int(prediction.get('num_observations'))} observations · "
        f"server time {prediction.get('server_timestamp', 'N/A')}"
    )


def _render_observations_tab(prediction: dict[str, Any]) -> None:
    obs = prediction.get("observations") or []
    if not obs:
        st.info("No per-observation records in this response.")
        return
    st.caption(f"First {min(len(obs), 100)} of {len(obs):,} observations returned by the API.")
    st.dataframe(pd.DataFrame(obs[:100]), hide_index=True, use_container_width=True)


def _render_results(prediction: dict[str, Any]) -> None:
    trip_id = prediction.get("trip_id", "unknown")
    hmm_metrics = (prediction.get("metrics") or {}).get("hmm") or {}

    st.markdown(f'<div class="section-label">Result · trip {html.escape(str(trip_id))}</div>', unsafe_allow_html=True)
    _render_kpis(prediction)

    render_res = build_trip_map(prediction)
    st.markdown('<div class="section-label">Route map</div>', unsafe_allow_html=True)
    if render_res.skipped_points > 0:
        st.warning(
            f"Skipped {render_res.skipped_points} point(s) outside valid latitude/longitude range. "
            "Check that the backend sends [lat, lon]."
        )
    st_folium(render_res.fmap, width=None, height=560, returned_objects=[])
    st.caption(
        f"{_fmt_int(render_res.hmm_segment_count)} route segment(s) drawn · "
        f"{_fmt_int(hmm_metrics.get('continuity_breaks'))} continuity breaks. "
        "Disconnected segments are drawn separately, never joined by a false line. "
        "Use the layer control (top right) to toggle layers."
    )

    st.markdown('<div class="section-label">Details</div>', unsafe_allow_html=True)
    tab_cmp, tab_time, tab_obs, tab_json = st.tabs(["Classifier vs HMM", "Timings", "Observations", "API response"])
    with tab_cmp:
        _render_comparison_tab(prediction)
    with tab_time:
        _render_timings_tab(prediction)
    with tab_obs:
        _render_observations_tab(prediction)
    with tab_json:
        st.json({k: (f"<list of {len(v)} items>" if isinstance(v, list) else v) for k, v in prediction.items()})

    st.download_button(
        label="Download map (HTML)",
        data=render_res.fmap.get_root().render(),
        file_name=f"pathsnap_trip_{trip_id}.html",
        mime="text/html",
    )


def main() -> None:
    _init_state()
    if not st.session_state["bootstrapped"]:
        st.session_state["bootstrapped"] = True
        _refresh_backend(st.session_state["api_base_url"])
    api_url = _render_sidebar()

    st.title("PathSnap")
    st.markdown(
        '<div class="app-sub">Match noisy GPS traces to the road network · PES University ML Hackathon 2026</div>',
        unsafe_allow_html=True,
    )
    _render_backend_header_banner()

    trips: list[TripSummary] = st.session_state["trips"]
    trips_err = st.session_state["trips_error"]

    if trips_err:
        st.error(f"Unable to load trips from Laptop 1: {trips_err}")
        st.info(
            "1. Check that the API is running on Laptop 1 (`uvicorn ... --host 0.0.0.0`).\n"
            "2. Check that both laptops are on the same network.\n"
            "3. Enter Laptop 1's address in the sidebar and press **Check connection**."
        )
        return
    if not trips:
        st.warning("Connected, but the API returned no trips.")
        return

    run_clicked = _render_trip_picker(trips)
    selected_id = st.session_state["selected_trip_id"]

    if run_clicked and selected_id:
        st.session_state["last_error"] = None
        with st.spinner(f"Running the pipeline on Laptop 1 for trip {selected_id}..."):
            try:
                result = run_prediction(api_url, selected_id)
                st.session_state["last_prediction"] = result
                hmm_acc = (result.get("metrics") or {}).get("hmm", {}).get("top1_accuracy")
                if hmm_acc is not None:
                    st.session_state["session_scores"][selected_id] = float(hmm_acc)
            except APIError as exc:
                st.session_state["last_error"] = str(exc)

    if st.session_state["last_error"]:
        st.error(f"Prediction failed: {st.session_state['last_error']}")
    if st.session_state["last_prediction"]:
        _render_results(st.session_state["last_prediction"])


if __name__ == "__main__":
    main()
