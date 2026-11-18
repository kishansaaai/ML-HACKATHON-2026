"""PathSnap Evaluator UI (Laptop 2 — Person D).

A lightweight Streamlit + Folium console that communicates strictly with the
PathSnap FastAPI backend on Laptop 1:
  - GET  /health   : verify backend and model readiness
  - GET  /trips    : fetch the 10 evaluation trip IDs
  - POST /predict  : send {"trip_id": ...} and render returned route + metrics

This application performs zero ML inference and never accesses `.route` files.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from api_client import APIError, TripSummary, check_health, fetch_trips, run_prediction
from config import DEFAULT_API_BASE_URL
from map_renderer import build_trip_map


st.set_page_config(
    page_title="PathSnap — Evaluator Console",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle styling for crisp trip cards and readable metrics
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .status-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .status-ok { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .status-err { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .trip-meta {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: -6px;
        margin-bottom: 6px;
    }
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
        "session_scores": {},  # trip_id -> HMM top-1 accuracy from this live session
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


def _render_sidebar() -> str:
    with st.sidebar:
        st.header("Laptop 1 API Config")
        api_url = st.text_input(
            "API Base URL",
            value=st.session_state["api_base_url"],
            help="Set to Laptop 1's LAN URL during evaluation, e.g. http://192.168.1.25:8000",
        ).strip()
        if api_url != st.session_state["api_base_url"]:
            st.session_state["api_base_url"] = api_url

        if st.button("Check Connection & Reload Trips", use_container_width=True):
            _refresh_backend(api_url)

        st.divider()
        st.subheader("Backend Status")
        health = st.session_state["health_info"]
        health_err = st.session_state["health_error"]

        if health:
            st.markdown('<span class="status-pill status-ok">● Connected</span>', unsafe_allow_html=True)
            st.caption(f"Latency: {health.get('_latency_ms', '?')} ms")
            st.write(f"**Model:** {health.get('model_name', 'Unknown')}")
            st.write(f"**Version:** `{health.get('version', 'N/A')}`")
            st.write(f"**Partition:** `{health.get('partition', st.session_state.get('partition', 'N/A'))}`")
            if not health.get("model_loaded", True):
                st.warning("API is reachable, but `model_loaded` is False.")
        elif health_err:
            st.markdown('<span class="status-pill status-err">● Unreachable</span>', unsafe_allow_html=True)
            st.error(health_err)
        else:
            st.info("Click **Check Connection** to ping Laptop 1.")

        st.divider()
        st.caption(
            "**Two-Laptop Architecture**\n\n"
            "- **Laptop 2 (this UI):** Displays trips, sends `trip_id`, renders map & metrics.\n"
            "- **Laptop 1 (FastAPI):** Runs classifier + HMM/Viterbi and server-side scoring. "
            "Ground truth never leaves Laptop 1."
        )
    return api_url


def _render_trip_cards(trips: list[TripSummary]) -> None:
    partition = st.session_state.get("partition") or "evaluation"
    st.subheader(f"Evaluation Trips ({len(trips)} available — `{partition}`)")
    st.caption("Select any trip below, then click **Run Map-Matching** to send a live request to Laptop 1.")

    cols_per_row = 5
    for row_start in range(0, len(trips), cols_per_row):
        row_trips = trips[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, trip in zip(cols, row_trips):
            is_selected = st.session_state["selected_trip_id"] == trip.trip_id
            prev_score = st.session_state["session_scores"].get(trip.trip_id)

            with col:
                with st.container(border=True):
                    label = f"Trip {trip.trip_id}"
                    if st.button(
                        f"✓ {label}" if is_selected else label,
                        key=f"select_trip_{trip.trip_id}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state["selected_trip_id"] = trip.trip_id
                        st.rerun()

                    meta_parts = []
                    if trip.num_observations is not None:
                        meta_parts.append(f"{trip.num_observations:,} GPS pts")
                    if trip.num_arcs is not None:
                        meta_parts.append(f"{trip.num_arcs:,} arcs")
                    if meta_parts:
                        st.markdown(f'<div class="trip-meta">{" · ".join(meta_parts)}</div>', unsafe_allow_html=True)

                    if prev_score is not None:
                        st.caption(f"Last HMM Top-1: **{_fmt_pct(prev_score)}**")


def _render_metrics_comparison(prediction: dict[str, Any]) -> None:
    metrics = prediction.get("metrics") or {}
    clf = metrics.get("classifier_only") or {}
    hmm = metrics.get("hmm") or {}
    timings = prediction.get("timings_sec") or {}

    # Top KPI strip
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="HMM Observation Top-1 Accuracy",
            value=_fmt_pct(hmm.get("top1_accuracy")),
            delta=_fmt_delta_pct(hmm.get("top1_accuracy"), clf.get("top1_accuracy")),
        )
    with k2:
        breaks = hmm.get("continuity_breaks")
        breaks_suffix = f" ({breaks} breaks)" if breaks is not None else ""
        st.metric(
            label=f"HMM Route Continuity{breaks_suffix}",
            value=_fmt_pct(hmm.get("route_continuity")),
            delta=_fmt_delta_pct(hmm.get("route_continuity"), clf.get("route_continuity")),
        )
    with k3:
        seq_or_f1_label = "HMM Sequence F1" if hmm.get("sequence_f1") is not None else "HMM Macro F1"
        seq_key = "sequence_f1" if hmm.get("sequence_f1") is not None else "macro_f1"
        st.metric(
            label=seq_or_f1_label,
            value=_fmt_pct(hmm.get(seq_key)),
            delta=_fmt_delta_pct(hmm.get(seq_key), clf.get(seq_key)),
        )
    with k4:
        total_srv = timings.get("total")
        roundtrip = prediction.get("_client_roundtrip_sec")
        val_str = f"{float(total_srv):.2f}s" if total_srv is not None else (
            f"{float(roundtrip):.2f}s" if roundtrip is not None else "N/A"
        )
        delta_str = f"round-trip {roundtrip:.2f}s" if (total_srv is not None and roundtrip is not None) else None
        st.metric(
            label="Server Inference Time",
            value=val_str,
            delta=delta_str,
            delta_color="off",
        )

    # Detailed side-by-side table + stage timings
    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.markdown("#### Classifier-Only vs. Classifier + HMM")
        rows = [
            ("Observation Top-1 Accuracy", _fmt_pct(clf.get("top1_accuracy")), _fmt_pct(hmm.get("top1_accuracy"))),
            ("Macro F1-score", _fmt_pct(clf.get("macro_f1")), _fmt_pct(hmm.get("macro_f1"))),
            ("Precision", _fmt_pct(clf.get("precision")), _fmt_pct(hmm.get("precision"))),
            ("Recall", _fmt_pct(clf.get("recall")), _fmt_pct(hmm.get("recall"))),
            ("Log Loss", _fmt_float(clf.get("log_loss")), _fmt_float(hmm.get("log_loss"))),
            ("Route Continuity", _fmt_pct(clf.get("route_continuity")), _fmt_pct(hmm.get("route_continuity"))),
            ("Continuity Breaks", _fmt_int(clf.get("continuity_breaks")), _fmt_int(hmm.get("continuity_breaks"))),
            ("Sequence-Level Score (F1)", _fmt_pct(clf.get("sequence_f1")), _fmt_pct(hmm.get("sequence_f1"))),
            ("Candidate Miss Rate", _fmt_pct(clf.get("candidate_miss_rate")), _fmt_pct(hmm.get("candidate_miss_rate"))),
        ]
        df_metrics = pd.DataFrame(rows, columns=["Metric", "Classifier-Only", "Classifier + HMM"])
        st.dataframe(df_metrics, hide_index=True, use_container_width=True)

    with right_col:
        st.markdown("#### Pipeline Timings (Laptop 1)")
        if timings:
            stage_labels = {
                "load_and_index": "Load & Spatial Index",
                "candidates_and_features": "Candidates & Features",
                "classifier_inference": "Classifier Inference",
                "hmm_viterbi": "HMM / Viterbi Decoding",
                "server_scoring": "Server-Side Scoring",
                "total": "Total Server Time",
            }
            timing_rows = [
                (stage_labels.get(k, k), f"{float(v):.3f} s")
                for k, v in timings.items()
                if v is not None
            ]
            df_timings = pd.DataFrame(timing_rows, columns=["Stage", "Duration"])
            st.dataframe(df_timings, hide_index=True, use_container_width=True)
        else:
            st.info("No per-stage timings returned by the API.")

        st.caption(
            f"**Trip ID:** `{prediction.get('trip_id', 'N/A')}`  \n"
            f"**Model:** {prediction.get('model_name', 'N/A')}  \n"
            f"**Observations:** {_fmt_int(prediction.get('num_observations'))}  \n"
            f"**Server Timestamp:** `{prediction.get('server_timestamp', 'N/A')}`"
        )


def _render_results(prediction: dict[str, Any]) -> None:
    trip_id = prediction.get("trip_id", "unknown")
    st.divider()
    st.subheader(f"Route Reconstruction Results — Trip `{trip_id}`")

    _render_metrics_comparison(prediction)

    st.markdown("#### Interactive Route Map (Folium)")
    st.caption(
        "Use the top-right layer selector to toggle **Raw GPS Observations**, "
        "**HMM / Viterbi Predicted Route**, and **Classifier-Only Predictions**."
    )
    fmap = build_trip_map(prediction)
    # returned_objects=[] prevents unnecessary Streamlit reruns when panning or zooming
    st_folium(fmap, width=None, height=540, returned_objects=[])

    # Exportable standalone HTML map (useful for offline inspection or report)
    map_html = fmap.get_root().render()
    st.download_button(
        label=f"Download Interactive Folium HTML (trip_{trip_id}.html)",
        data=map_html,
        file_name=f"pathsnap_trip_{trip_id}.html",
        mime="text/html",
    )

    with st.expander("Inspect Returned Observations & API JSON (Verification)"):
        tab_obs, tab_json = st.tabs(["Sample Observations", "API Response Summary"])
        with tab_obs:
            obs = prediction.get("observations") or []
            if obs:
                st.caption(f"Showing first {min(len(obs), 100)} of {len(obs):,} observations returned by the API.")
                st.dataframe(pd.DataFrame(obs[:100]), hide_index=True, use_container_width=True)
            else:
                st.info("No per-observation records included in this response.")
        with tab_json:
            # Show compact summary so large coordinate arrays don't freeze the browser
            summary_json = {
                k: (f"<list of {len(v)} items>" if isinstance(v, list) else v)
                for k, v in prediction.items()
            }
            st.json(summary_json)


def main() -> None:
    _init_state()
    api_url = _render_sidebar()

    # Auto-connect on initial page load
    if not st.session_state["bootstrapped"]:
        st.session_state["bootstrapped"] = True
        _refresh_backend(api_url)

    st.title("PathSnap — Distributed Map-Matching Evaluator UI")
    st.caption(
        "PES University · UE24CS352A Machine Learning Hackathon 2026 · "
        "Laptop 2 Evaluator Console"
    )

    trips: list[TripSummary] = st.session_state["trips"]
    trips_err = st.session_state["trips_error"]

    if trips_err:
        st.error(f"**Unable to load trips from Laptop 1:** {trips_err}")
        st.info(
            "**Quick Troubleshooting:**\n"
            "1. Confirm FastAPI is running on Laptop 1 (`uvicorn ... --host 0.0.0.0 --port 8000`).\n"
            "2. Confirm both laptops are on the same Wi-Fi / hotspot.\n"
            "3. Enter Laptop 1's IP address in the sidebar **API Base URL** and click **Check Connection & Reload Trips**."
        )
        return

    if not trips:
        st.warning("Connected to API, but `/trips` returned an empty trip list.")
        return

    _render_trip_cards(trips)

    # Action bar to run live prediction for the selected trip
    selected_id = st.session_state["selected_trip_id"]
    run_col, info_col = st.columns([1.4, 3])
    with run_col:
        run_clicked = st.button(
            f"Run Map-Matching for Trip {selected_id}",
            type="primary",
            use_container_width=True,
            disabled=not selected_id,
        )
    with info_col:
        st.caption(
            f"Sends `POST {api_url}/predict` with `{{\"trip_id\": \"{selected_id}\"}}`. "
            "Watch Laptop 1's terminal console for live request logs."
        )

    if run_clicked and selected_id:
        st.session_state["last_error"] = None
        with st.spinner(f"Running PathSnap pipeline on Laptop 1 for Trip {selected_id}..."):
            try:
                result = run_prediction(api_url, selected_id)
                st.session_state["last_prediction"] = result
                hmm_acc = (result.get("metrics") or {}).get("hmm", {}).get("top1_accuracy")
                if hmm_acc is not None:
                    st.session_state["session_scores"][selected_id] = float(hmm_acc)
            except APIError as exc:
                st.session_state["last_error"] = str(exc)

    if st.session_state["last_error"]:
        st.error(f"**Prediction Failed:** {st.session_state['last_error']}")

    if st.session_state["last_prediction"]:
        _render_results(st.session_state["last_prediction"])


if __name__ == "__main__":
    main()
