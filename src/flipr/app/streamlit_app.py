"""Streamlit dashboard for FLIPR session QC and inspection.

Run with::

    uv run streamlit run src/flipr/app/streamlit_app.py

Or via the installed console script::

    uv run flipr-app
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from flipr.io.session_csv import SessionData, list_sessions, load_session
from flipr.io.tidy_csv import (
    TidyData,
    list_tidy_files,
    load_tidy,
    match_tidy_to_session,
)
from flipr.preprocess.lifetime import DoubleExpFit, fit_double_exp
from flipr.viz.tcspc import fit_params_table, tcspc_decay_figure
from flipr.viz.traces import session_traces_figure

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FLIPR analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3].parent / "FLIPR data"


# -----------------------------------------------------------------------------
# Cached loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading session CSVs…")
def cached_load_session(session_path: str) -> SessionData:
    return load_session(session_path)


@st.cache_data(show_spinner="Loading tidy TCSPC data…")
def cached_load_tidy(data_csv: str) -> TidyData:
    return load_tidy(data_csv)


@st.cache_data(show_spinner="Fitting double-exponential…")
def cached_fit(
    tcspc: np.ndarray,
    bins: np.ndarray,
    irf_sigma: float,
    t0: float,
    fit_start: float,
    fit_stop: float,
) -> DoubleExpFit:
    return fit_double_exp(
        tcspc,
        bins,
        irf_sigma_ns=irf_sigma,
        t0_ns=t0,
        fit_start_ns=fit_start,
        fit_stop_ns=fit_stop,
    )


# -----------------------------------------------------------------------------
# Sidebar: data root + session picker
# -----------------------------------------------------------------------------
def sidebar() -> tuple[Path | None, SessionData | None, TidyData | None]:
    st.sidebar.title("FLIPR analysis")
    st.sidebar.caption("v0.0.1")

    default = str(DEFAULT_DATA_ROOT) if DEFAULT_DATA_ROOT.is_dir() else ""
    data_root_str = st.sidebar.text_input(
        "Data root",
        value=default,
        help="Directory containing tidy/ and sessions/ subfolders",
    )
    if not data_root_str:
        st.sidebar.warning("Enter a FLIPR data root to begin.")
        return None, None, None
    data_root = Path(data_root_str).expanduser()
    if not data_root.is_dir():
        st.sidebar.error(f"Not a directory: {data_root}")
        return None, None, None

    sessions = list_sessions(data_root)
    if not sessions:
        st.sidebar.warning(f"No sessions/ folder found under {data_root}.")
        return data_root, None, None

    session_names = [p.name for p in sessions]
    picked_name = st.sidebar.selectbox("Session", session_names, index=len(sessions) - 1)
    picked_path = next(p for p in sessions if p.name == picked_name)

    try:
        session = cached_load_session(str(picked_path))
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Failed to load session: {exc}")
        return data_root, None, None

    # Match to tidy file for TCSPC data (for the interval inspector)
    tidy_files = list_tidy_files(data_root)
    matched = match_tidy_to_session(tidy_files, session.blockname)
    tidy: TidyData | None = None
    if matched is None:
        st.sidebar.warning("No matching tidy/ file for this session — the Interval Inspector will be disabled.")
    else:
        st.sidebar.caption(f"Tidy file: `{matched.name}`")
        try:
            tidy = cached_load_tidy(str(matched))
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Failed to load tidy file: {exc}")

    # Session metadata block
    with st.sidebar.expander("Session metadata", expanded=True):
        st.write(
            {
                "subject": session.subject,
                "procedure": session.procedure,
                "fs (Hz)": session.fs,
                "blockname": session.blockname,
                "duration (s)": float(session.streams["time"].max()),
                "n samples": int(len(session.streams)),
                "event types": session.event_types,
            }
        )
        note = session.meta.get("note_general", "")
        if note and note != "NA":
            st.caption(f"Note: {note}")

    return data_root, session, tidy


# -----------------------------------------------------------------------------
# Tab 1: Session overview
# -----------------------------------------------------------------------------
def render_overview(session: SessionData, highlight_range: tuple[float, float] | None = None) -> None:
    st.subheader("Session overview")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        log_intensity = st.toggle("log intensity", value=False)
    with col2:
        max_points = st.select_slider(
            "render points",
            options=[5_000, 10_000, 20_000, 50_000, 100_000],
            value=20_000,
            help="Downsample the stream plot to keep zoom snappy.",
        )
    with col3:
        ev_types = st.multiselect(
            "events to overlay",
            options=session.event_types,
            default=session.event_types,
        )

    fig = session_traces_figure(
        session,
        max_points=max_points,
        log_intensity=log_intensity,
        event_types=ev_types,
        highlight_range=highlight_range,
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True)

    # QC summary row
    mc1, mc2, mc3, mc4 = st.columns(4)
    streams = session.streams
    mc1.metric("mean intensity", f"{streams['intensity'].mean():,.0f}")
    mc2.metric("intensity CV", f"{streams['intensity'].std() / max(streams['intensity'].mean(), 1):.3f}")
    mc3.metric("mean τ (ns)", f"{streams['lifetime'].mean():.3f}")
    mc4.metric("τ std (ns)", f"{streams['lifetime'].std():.3f}")

    # Per-event count summary
    with st.expander("event counts"):
        ev_df = (
            session.events.groupby("event_id_char")
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={"event_id_char": "event"})
        )
        st.dataframe(ev_df, hide_index=True, use_container_width=False)


# -----------------------------------------------------------------------------
# Tab 2: Interval inspector
# -----------------------------------------------------------------------------
def render_interval(session: SessionData, tidy: TidyData) -> None:
    st.subheader("Interval inspector")
    st.caption(
        "Sum TCSPC histograms across a user-selected time window and re-fit "
        "the double exponential. Useful for comparing baseline vs event-locked "
        "lifetime, or flagging drift over the session."
    )

    col_slider, col_step = st.columns([3, 1])
    with col_step:
        window_nudge = st.number_input(
            "nudge step (s)", min_value=0.05, max_value=60.0, value=0.5, step=0.05
        )
    with col_slider:
        t_end = float(session.streams["time"].max())
        t_range = st.slider(
            "time window (s)",
            min_value=0.0,
            max_value=t_end,
            key="interval_range",  # session-state-backed; both tabs read this
            step=float(window_nudge),
        )

    t0_window, t1_window = t_range
    if t1_window <= t0_window:
        st.error("window end must exceed window start")
        return

    hist = tidy.integrate_histogram(t0_window, t1_window).astype(np.float64)
    if hist.sum() == 0:
        st.error("no photons in the selected window")
        return t_range

    # Fit params default to the session's meta
    default_irf = float(session.meta.get("IRF", "0.212"))
    default_t0 = float(session.meta.get("t0", "1.012"))
    default_start = float(tidy.params.get("header_state_spcrangelow", "0.4"))
    default_stop = float(tidy.params.get("header_state_spcrangehigh", "12.3"))

    with st.expander("fit settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        irf = c1.number_input("IRF σ (ns)", value=default_irf, min_value=0.05, max_value=2.0, step=0.01)
        t0 = c2.number_input("t₀ (ns)", value=default_t0, min_value=0.0, max_value=5.0, step=0.01)
        fit_lo = c3.number_input("fit start (ns)", value=default_start, min_value=0.0, max_value=10.0, step=0.1)
        fit_hi = c4.number_input("fit stop (ns)", value=default_stop, min_value=1.0, max_value=12.5, step=0.1)

    fit = cached_fit(hist, tidy.tcspc_bins_ns, irf, t0, fit_lo, fit_hi)

    title = f"window: t ∈ [{t0_window:.2f}, {t1_window:.2f}] s   ·   {int(hist.sum()):,} photons"
    fig = tcspc_decay_figure(hist, tidy.tcspc_bins_ns, fit=fit, log_y=True, title=title, height=560)
    st.plotly_chart(fig, use_container_width=True)

    # Fit parameters table + header comparison
    pc1, pc2 = st.columns([2, 1])
    with pc1:
        st.markdown("**Fit parameters**")
        st.dataframe(
            pd.DataFrame(fit_params_table(fit)),
            hide_index=True,
            use_container_width=True,
        )
    with pc2:
        st.markdown("**Instrument header (reference)**")
        header_rows = [
            ("τ₁ (ns)", tidy.params.get("header_state_tau1", "—")),
            ("τ₂ (ns)", tidy.params.get("header_state_tau2", "—")),
            ("τ̄ avg (ns)", tidy.params.get("header_state_avgtau", "—")),
            ("pop₁ %", tidy.params.get("header_state_pop1pct", "—")),
            ("pop₂ %", tidy.params.get("header_state_pop2pct", "—")),
            ("IRF β (ns)", tidy.params.get("header_state_beta6", "—")),
        ]
        st.dataframe(
            pd.DataFrame(header_rows, columns=["param", "value"]),
            hide_index=True,
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    data_root, session, tidy = sidebar()
    if session is None:
        st.info("Pick a FLIPR data root and session in the sidebar to begin.")
        return

    st.title(f"{session.subject} · {session.blockname}")

    # Seed the interval_range session state the first time the session loads,
    # so both the overview (for its shaded highlight) and the interval
    # inspector's slider start from the same default and stay in sync.
    t_end = float(session.streams["time"].max())
    default_range = (0.0, min(30.0, t_end))
    if "interval_range" not in st.session_state:
        st.session_state["interval_range"] = default_range
    # Reset if we've switched sessions (different blockname → previous range
    # may exceed the new session length)
    if st.session_state.get("_range_session") != session.blockname:
        st.session_state["interval_range"] = default_range
        st.session_state["_range_session"] = session.blockname

    tab_overview, tab_interval = st.tabs(["Session overview", "Interval inspector"])

    with tab_overview:
        render_overview(session, highlight_range=st.session_state["interval_range"])

    with tab_interval:
        if tidy is None:
            st.warning(
                "Interval inspector requires a tidy/\\*_data.csv file matching "
                "this session (for per-frame TCSPC histograms)."
            )
        else:
            render_interval(session, tidy)


if __name__ == "__main__":
    main()
