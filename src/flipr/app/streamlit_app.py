"""Streamlit dashboard for FLIPR session QC and inspection.

Run with::

    uv run streamlit run src/flipr/app/streamlit_app.py

Or via the installed console script::

    uv run flipr-app
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from flipr.align.peth import build_peth
from flipr.io.iflip2 import (
    IFlip2File,
    list_iflip2_files,
    load_iflip2,
    match_iflip2_to_session,
)
from flipr.io.session_csv import SessionData, list_sessions, load_session
from flipr.io.tidy_csv import (
    TidyData,
    list_tidy_files,
    load_tidy,
    match_tidy_to_session,
)
from flipr.preprocess.filters import FILTER_MODES, filtered_session
from flipr.preprocess.lifetime import DoubleExpFit, fit_double_exp
from flipr.preprocess.motion import QCResult, compute_qc
from flipr.preprocess.phasor import phasor_from_histogram, phasor_series_from_tcspc
from flipr.viz.peth import peth_figure
from flipr.viz.phasor import phasor_plot_figure
from flipr.viz.tcspc import fit_params_table, tcspc_decay_figure
from flipr.viz.traces import session_traces_figure

LOGO_PATH = Path(__file__).parent / "assets" / "FLIPRlogo.png"

# Union type for objects that expose the TCSPC-source interface used by
# the interval inspector and phasor tabs. Both TidyData (from tidy CSV
# exports) and IFlip2File (from raw binary files) satisfy it.
TCSPCSource = TidyData | IFlip2File

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


@st.cache_data(show_spinner="Loading raw .iFLiP2 file…")
def cached_load_iflip2(path: str) -> IFlip2File:
    return load_iflip2(path)


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


@st.cache_data(show_spinner="Computing phasor trajectory…")
def cached_phasor_series(
    tcspc: np.ndarray,
    bins: np.ndarray,
    period_ns: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    return phasor_series_from_tcspc(tcspc, bins, period_ns=period_ns)


@st.cache_data(show_spinner="Computing QC metrics…")
def cached_qc(
    session_path: str,
    filter_mode: str,
    window_s: float,
    polyorder: int,
    motion_window_s: float,
    motion_corr_threshold: float,
) -> QCResult:
    session = cached_load_session(session_path)
    filt = filtered_session(
        session, mode=filter_mode, window_s=window_s, polyorder=polyorder  # type: ignore[arg-type]
    )
    return compute_qc(
        filt,
        motion_window_s=motion_window_s,
        motion_corr_threshold=motion_corr_threshold,
    )


# -----------------------------------------------------------------------------
# Sidebar: data root + session picker
# -----------------------------------------------------------------------------
@dataclass
class SidebarState:
    """Everything the sidebar exposes to the tabs."""

    data_root: Path | None
    session_raw: SessionData | None
    session: SessionData | None  # == session_raw if filter_mode == "none"
    session_path: Path | None
    tcspc_source: TCSPCSource | None  # TidyData or IFlip2File, whichever the user picked
    tcspc_source_label: str  # e.g. "raw .iFLiP2" or "tidy CSV" for UI display
    filter_mode: str
    filter_window_s: float
    filter_polyorder: int
    qc_motion_window_s: float
    qc_motion_corr_threshold: float


def sidebar() -> SidebarState:
    if LOGO_PATH.is_file():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.sidebar.title("FLIPR analysis")
    st.sidebar.caption("v0.0.1")

    empty_state = SidebarState(
        data_root=None, session_raw=None, session=None, session_path=None,
        tcspc_source=None, tcspc_source_label="",
        filter_mode="none", filter_window_s=0.25,
        filter_polyorder=2, qc_motion_window_s=1.0,
        qc_motion_corr_threshold=0.85,
    )

    default = str(DEFAULT_DATA_ROOT) if DEFAULT_DATA_ROOT.is_dir() else ""
    data_root_str = st.sidebar.text_input(
        "Data root",
        value=default,
        help="Directory containing tidy/ and sessions/ subfolders",
    )
    if not data_root_str:
        st.sidebar.warning("Enter a FLIPR data root to begin.")
        return empty_state
    data_root = Path(data_root_str).expanduser()
    if not data_root.is_dir():
        st.sidebar.error(f"Not a directory: {data_root}")
        return empty_state

    sessions = list_sessions(data_root)
    if not sessions:
        st.sidebar.warning(f"No sessions/ folder found under {data_root}.")
        return replace(empty_state, data_root=data_root)

    session_names = [p.name for p in sessions]
    picked_name = st.sidebar.selectbox("Session", session_names, index=len(sessions) - 1)
    picked_path = next(p for p in sessions if p.name == picked_name)

    try:
        session_raw = cached_load_session(str(picked_path))
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Failed to load session: {exc}")
        return replace(empty_state, data_root=data_root)

    # Find TCSPC sources: the raw .iFLiP2 (if present) is byte-exact,
    # the tidy CSV is the exported form. Prefer iFLiP2 when both exist
    # but let the user override in the UI.
    iflip2_files = list_iflip2_files(data_root)
    tidy_files = list_tidy_files(data_root)
    iflip2_match = match_iflip2_to_session(iflip2_files, session_raw.blockname)
    tidy_match = match_tidy_to_session(tidy_files, session_raw.blockname)

    # Decide which options are available
    source_options: list[str] = []
    if iflip2_match is not None:
        source_options.append("raw .iFLiP2")
    if tidy_match is not None:
        source_options.append("tidy CSV")

    tcspc_source: TCSPCSource | None = None
    tcspc_source_label: str = ""
    if not source_options:
        st.sidebar.warning(
            "No matching raw/ or tidy/ file for this session — Interval "
            "Inspector, Phasor, and QC are disabled."
        )
    else:
        # Default to iFLiP2 if present (source of truth)
        default_idx = 0
        chosen = st.sidebar.radio(
            "TCSPC source",
            options=source_options,
            index=default_idx,
            help="The raw .iFLiP2 file is the source of truth produced "
                 "directly by the acquisition rig; the tidy CSV is an "
                 "export of the same data. Use raw when available.",
        )
        try:
            if chosen == "raw .iFLiP2" and iflip2_match is not None:
                tcspc_source = cached_load_iflip2(str(iflip2_match))
                tcspc_source_label = f"raw · {iflip2_match.name}"
                st.sidebar.caption(f"Raw file: `{iflip2_match.name}`")
            elif chosen == "tidy CSV" and tidy_match is not None:
                tcspc_source = cached_load_tidy(str(tidy_match))
                tcspc_source_label = f"tidy · {tidy_match.name}"
                st.sidebar.caption(f"Tidy file: `{tidy_match.name}`")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Failed to load {chosen}: {exc}")
            tcspc_source = None

    # --- Filter controls (global) ---
    st.sidebar.divider()
    st.sidebar.markdown("**Stream filter**")
    st.sidebar.caption(
        "Smoothing applied to intensity + lifetime. Propagates to the session "
        "overview and PETH tabs. Interval inspector and phasor re-compute from "
        "the raw TCSPC histograms and are unaffected."
    )
    filter_mode = st.sidebar.selectbox(
        "filter mode",
        options=list(FILTER_MODES),
        index=0,
        key="filter_mode",
    )
    if filter_mode == "none":
        window_s = 0.25
        polyorder = 2
    else:
        window_s = st.sidebar.number_input(
            "window (s)",
            min_value=0.05, max_value=10.0, value=0.25, step=0.05,
            key="filter_window",
        )
        if filter_mode == "savgol":
            polyorder = int(
                st.sidebar.number_input(
                    "polyorder", min_value=1, max_value=5, value=2, step=1,
                    key="filter_polyorder",
                )
            )
        else:
            polyorder = 2

    session = filtered_session(
        session_raw, mode=filter_mode,  # type: ignore[arg-type]
        window_s=float(window_s), polyorder=int(polyorder),
    )

    # --- QC correlation-diagnostic controls ---
    with st.sidebar.expander("QC · correlation diagnostic", expanded=False):
        st.caption(
            "Rolling r(intensity, lifetime) is informational only — "
            "see the Overview tab. Not used for flagging."
        )
        qc_motion_window_s = st.number_input(
            "rolling window (s)", min_value=0.1, max_value=30.0,
            value=1.0, step=0.1, key="qc_motion_window",
        )
        qc_motion_corr_threshold = st.slider(
            "highlight |r| above", min_value=0.3, max_value=0.99,
            value=0.85, step=0.01, key="qc_motion_threshold",
        )

    # Session metadata block
    with st.sidebar.expander("Session metadata", expanded=False):
        st.write(
            {
                "subject": session_raw.subject,
                "procedure": session_raw.procedure,
                "fs (Hz)": session_raw.fs,
                "blockname": session_raw.blockname,
                "duration (s)": float(session_raw.streams["time"].max()),
                "n samples": int(len(session_raw.streams)),
                "event types": session_raw.event_types,
            }
        )
        note = session_raw.meta.get("note_general", "")
        if note and note != "NA":
            st.caption(f"Note: {note}")

    return SidebarState(
        data_root=data_root,
        session_raw=session_raw,
        session=session,
        session_path=picked_path,
        tcspc_source=tcspc_source,
        tcspc_source_label=tcspc_source_label,
        filter_mode=str(filter_mode),
        filter_window_s=float(window_s),
        filter_polyorder=int(polyorder),
        qc_motion_window_s=float(qc_motion_window_s),
        qc_motion_corr_threshold=float(qc_motion_corr_threshold),
    )


# -----------------------------------------------------------------------------
# Tab 1: Session overview
# -----------------------------------------------------------------------------
def render_overview(
    session: SessionData,
    *,
    qc: QCResult | None = None,
    filter_label: str = "raw",
    highlight_range: tuple[float, float] | None = None,
) -> None:
    # Logo banner at the top of the first page
    if LOGO_PATH.is_file():
        col_logo, col_title = st.columns([1, 4])
        with col_logo:
            st.image(str(LOGO_PATH), width=140)
        with col_title:
            st.subheader("Session overview")
            st.caption(
                f"Subject **{session.subject}** · procedure **{session.procedure}** · "
                f"filter: *{filter_label}*"
            )
    else:
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

    # QC flags + diagnostic correlation
    if qc is not None:
        st.divider()
        st.markdown("**QC · artifact flags**")
        st.caption(
            "Photon-starvation and intensity-jump flags are the only "
            "true artifact detectors for FLIPR data. A stable head-fixed "
            "recording should show ~0% flagged. The rolling intensity/"
            "lifetime correlation shown below is a *diagnostic only* — "
            "for FLIM biosensors it naturally tracks biology (ligand "
            "binding changes both channels) and is not a motion "
            "rejection criterion."
        )
        summary = qc.summary()
        qc_cols = st.columns(4)
        qc_cols[0].metric("flagged (any)", f"{summary['any']*100:.2f}%")
        qc_cols[1].metric("low photons", f"{summary['intensity']*100:.2f}%")
        qc_cols[2].metric("intensity jumps", f"{summary['jump']*100:.2f}%")
        qc_cols[3].metric(
            "|rolling r| above threshold (diagnostic)",
            f"{summary['corr_above_threshold']*100:.2f}%",
            help="Informational only — NOT counted in the flagged total.",
        )

        with st.expander("rolling intensity/lifetime correlation (diagnostic)"):
            import plotly.graph_objects as go

            corr_fig = go.Figure()
            corr_fig.add_trace(
                go.Scattergl(
                    x=qc.time,
                    y=qc.rolling_corr,
                    mode="lines",
                    line=dict(color="#888", width=1),
                    name="rolling r(I, τ)",
                )
            )
            above = qc.corr_above_threshold
            if above.any():
                corr_fig.add_trace(
                    go.Scattergl(
                        x=qc.time[above],
                        y=qc.rolling_corr[above],
                        mode="markers",
                        marker=dict(size=4, color="#1f77b4"),
                        name="above threshold (diagnostic)",
                    )
                )
            # Genuine artifact flags highlighted in red
            real = qc.any_flag
            if real.any():
                corr_fig.add_trace(
                    go.Scattergl(
                        x=qc.time[real],
                        y=qc.rolling_corr[real],
                        mode="markers",
                        marker=dict(size=6, color="#d62728"),
                        name="real flag (any_flag)",
                    )
                )
            corr_fig.add_hline(y=0, line_dash="dot", line_color="#bbb", line_width=1)
            corr_fig.update_layout(
                height=240,
                margin=dict(l=60, r=20, t=10, b=40),
                xaxis_title="time (s)",
                yaxis_title="rolling r",
                yaxis_range=[-1, 1],
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
            )
            st.plotly_chart(corr_fig, use_container_width=True)

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
def render_interval(session: SessionData, tidy: TCSPCSource) -> None:
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
        # Stringify explicitly — tidy CSV returns str values, iFLiP2 returns
        # float/int, and mixing them in a "value" column gives pandas an
        # object dtype that Streamlit's Arrow serialiser can't convert.
        def _fmt(key: str) -> str:
            v = tidy.params.get(key)
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.5g}"
            return str(v)

        header_rows = [
            ("τ₁ (ns)", _fmt("header_state_tau1")),
            ("τ₂ (ns)", _fmt("header_state_tau2")),
            ("τ̄ avg (ns)", _fmt("header_state_avgtau")),
            ("pop₁ %", _fmt("header_state_pop1pct")),
            ("pop₂ %", _fmt("header_state_pop2pct")),
            ("IRF β (ns)", _fmt("header_state_beta6")),
        ]
        st.dataframe(
            pd.DataFrame(header_rows, columns=["param", "value"]),
            hide_index=True,
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Tab 3: Phasor explorer
# -----------------------------------------------------------------------------
def render_phasor(
    session: SessionData,
    tidy: TCSPCSource,
    interval_range: tuple[float, float] | None = None,
) -> None:
    st.subheader("Phasor explorer")
    st.caption(
        "Each frame's TCSPC histogram is mapped to a point in phasor space "
        "``(G, S)`` at the laser fundamental (80 MHz). Single-exponential "
        "decays sit on the universal semicircle; mixtures land inside it. "
        "Cloud drift across the session highlights sensor state changes "
        "without any fitting."
    )
    st.info(
        "**Note:** points are **uncalibrated** — the instrument IRF rotates "
        "the entire cloud by a fixed angle, so the raw coordinates may sit "
        "slightly above or below the semicircle. Relative drift and cloud "
        "shape are still meaningful. A reference-dye calibration step is a "
        "planned v2 addition.",
        icon="ℹ️",
    )

    # Compute period from the bin axis (last bin + bin_step); in practice 12.5 ns
    bin_step = float(tidy.tcspc_bins_ns[1] - tidy.tcspc_bins_ns[0])
    period_ns = round(bin_step * (int(12.5 / bin_step)), 4)  # → 12.5 for 0.1 ns bins

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        color_mode = st.radio(
            "point colour",
            options=["time", "trial", "none"],
            horizontal=True,
            index=0,
            help="Colour the phasor cloud by session time, trial index, or single colour.",
        )
    with col_b:
        show_ref = st.toggle("show header τ reference points", value=True)
    with col_c:
        show_window = st.toggle("mark selected window", value=True, disabled=interval_range is None)

    # Full-session phasor trajectory
    real, imag, mean, freq = cached_phasor_series(
        tidy.tcspc.astype(np.float64), tidy.tcspc_bins_ns, period_ns
    )

    # Colour array + hover
    t_axis = tidy.streams["time"].to_numpy()
    hover = [f"t={t:.2f}s" for t in t_axis]
    if color_mode == "time":
        color_arr = t_axis
    elif color_mode == "trial":
        # Assign each frame its trial number by forward-filling from
        # solution_onset events
        events = session.events
        onsets = events[events["event_id_char"] == "solution_onset"]["time"].to_numpy()
        if onsets.size == 0:
            color_arr = None
        else:
            color_arr = np.searchsorted(onsets, t_axis, side="right").astype(np.float64)
    else:
        color_arr = None

    reference_taus: list[float] | None = None
    if show_ref:
        try:
            tau1 = float(tidy.params["header_state_tau1"])
            tau2 = float(tidy.params["header_state_tau2"])
            reference_taus = [tau1, tau2]
        except (KeyError, ValueError):
            reference_taus = None

    highlight_point: tuple[float, float] | None = None
    if show_window and interval_range is not None:
        window_hist = tidy.integrate_histogram(*interval_range).astype(np.float64)
        win = phasor_from_histogram(window_hist, tidy.tcspc_bins_ns, period_ns=period_ns)
        if np.isfinite(win.real) and np.isfinite(win.imag):
            highlight_point = (win.real, win.imag)

    # Keep only finite points for plotting
    finite = np.isfinite(real) & np.isfinite(imag)
    real_plot = real[finite]
    imag_plot = imag[finite]
    hover_plot = [hover[i] for i in np.where(finite)[0]]
    color_plot = color_arr[finite] if color_arr is not None else None

    title = f"{freq:.1f} MHz · {finite.sum():,} frames"
    fig = phasor_plot_figure(
        real=real_plot,
        imag=imag_plot,
        point_color=color_plot,
        point_label="session frames",
        point_hover=np.array(hover_plot),
        reference_taus=reference_taus,
        highlight=highlight_point,
        period_ns=period_ns,
        title=title,
        height=620,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats row
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("mean G", f"{np.nanmean(real):.4f}")
    mc2.metric("mean S", f"{np.nanmean(imag):.4f}")
    mc3.metric("G std", f"{np.nanstd(real):.4f}")
    mc4.metric("S std", f"{np.nanstd(imag):.4f}")

    if highlight_point is not None and interval_range is not None:
        st.caption(
            f"Selected window t ∈ [{interval_range[0]:.2f}, {interval_range[1]:.2f}] s → "
            f"G = {highlight_point[0]:.4f}, S = {highlight_point[1]:.4f}"
        )


# -----------------------------------------------------------------------------
# Tab 4: Event-aligned PETH
# -----------------------------------------------------------------------------
def render_peth(session: SessionData, *, filter_label: str = "raw") -> None:
    st.subheader("Event-aligned PETH")
    st.caption(
        "Per-trial matrix + mean ± SEM trace for any continuous signal "
        "around any event type. Rebuilt from the session streams + event "
        "list, so you can pick arbitrary windows and normalisation modes. "
        f"**Signal source: {filter_label}** (change via sidebar filter)."
    )

    event_types = session.peth_event_types() or session.event_types
    if not event_types:
        st.warning("this session has no events")
        return

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        event_type = st.selectbox(
            "event type",
            options=event_types,
            index=0,
        )
    with col_b:
        signal = st.selectbox(
            "signal",
            options=["lifetime", "intensity"],
            index=0,
        )
    with col_c:
        norm_mode = st.selectbox(
            "normalisation",
            options=["raw", "baseline-corrected", "z-scored"],
            index=1,
            help="baseline-corrected = subtract pre-event baseline; "
                 "z-scored = (x - baseline_mean) / baseline_std",
        )

    col_pre, col_post, col_bl_lo, col_bl_hi = st.columns(4)
    with col_pre:
        pre_window = st.number_input(
            "pre (s)", min_value=-60.0, max_value=-0.1, value=-3.0, step=0.1
        )
    with col_post:
        post_window = st.number_input(
            "post (s)", min_value=0.1, max_value=60.0, value=5.0, step=0.1
        )
    with col_bl_lo:
        bl_lo = st.number_input(
            "baseline start (s)", min_value=-60.0, max_value=0.0,
            value=max(float(pre_window), -2.5), step=0.1,
            disabled=(norm_mode == "raw"),
        )
    with col_bl_hi:
        bl_hi = st.number_input(
            "baseline end (s)", min_value=-60.0, max_value=5.0,
            value=-0.1, step=0.1,
            disabled=(norm_mode == "raw"),
        )

    try:
        peth = build_peth(
            session,
            event_type=event_type,
            signal=signal,
            pre_window=float(pre_window),
            post_window=float(post_window),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    baseline_window = (float(bl_lo), float(bl_hi))
    if norm_mode == "baseline-corrected":
        try:
            peth_display = peth.baseline_corrected(window=baseline_window)
        except ValueError as exc:
            st.error(f"baseline correction failed: {exc}")
            return
    elif norm_mode == "z-scored":
        try:
            peth_display = peth.zscored(window=baseline_window)
        except ValueError as exc:
            st.error(f"z-score failed: {exc}")
            return
    else:
        peth_display = peth

    fig = peth_figure(
        peth_display,
        title=f"{event_type} · {signal} · {norm_mode}",
        height=620,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    mean = peth_display.mean()
    peak_idx = int(np.nanargmax(np.abs(mean)))
    peak_time = float(peth_display.time_rel[peak_idx])
    peak_val = float(mean[peak_idx])

    try:
        auc = peth_display.trial_auc(window=(0.0, float(post_window)))
        auc_mean = float(np.nanmean(auc))
        auc_sem = float(np.nanstd(auc, ddof=1) / np.sqrt(len(auc)))
    except ValueError:
        auc_mean = float("nan")
        auc_sem = float("nan")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("n trials", f"{peth_display.n_trials}")
    mc2.metric("peak |mean|", f"{peak_val:+.4f}")
    mc3.metric("peak time (s)", f"{peak_time:+.2f}")
    mc4.metric("post-event AUC ± SEM", f"{auc_mean:.3f} ± {auc_sem:.3f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    state = sidebar()
    if state.session is None or state.session_raw is None:
        st.info("Pick a FLIPR data root and session in the sidebar to begin.")
        return

    session = state.session  # filtered (or raw if filter_mode == 'none')
    session_raw = state.session_raw
    tcspc_source = state.tcspc_source

    st.title(f"{session_raw.subject} · {session_raw.blockname}")

    # Seed the interval_range session state the first time the session loads.
    t_end = float(session_raw.streams["time"].max())
    default_range = (0.0, min(30.0, t_end))
    if "interval_range" not in st.session_state:
        st.session_state["interval_range"] = default_range
    if st.session_state.get("_range_session") != session_raw.blockname:
        st.session_state["interval_range"] = default_range
        st.session_state["_range_session"] = session_raw.blockname

    # QC is computed from the filtered session (user choice propagates)
    try:
        qc = cached_qc(
            str(state.session_path),
            state.filter_mode,
            state.filter_window_s,
            state.filter_polyorder,
            state.qc_motion_window_s,
            state.qc_motion_corr_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"QC computation failed: {exc}")
        qc = None

    filter_label = (
        "raw"
        if state.filter_mode == "none"
        else f"{state.filter_mode} · {state.filter_window_s:.2f}s"
        + (f" · order {state.filter_polyorder}" if state.filter_mode == "savgol" else "")
    )

    tab_overview, tab_interval, tab_phasor, tab_peth = st.tabs(
        [
            "Session overview",
            "Interval inspector",
            "Phasor explorer",
            "Event PETH",
        ]
    )

    with tab_overview:
        render_overview(
            session,
            qc=qc,
            filter_label=filter_label,
            highlight_range=st.session_state["interval_range"],
        )

    with tab_interval:
        if tcspc_source is None:
            st.warning(
                "Interval inspector needs either a raw/\\*.iFLiP2 file or a "
                "tidy/\\*_data.csv file matching this session (for per-frame "
                "TCSPC histograms)."
            )
        else:
            st.caption(f"TCSPC source: **{state.tcspc_source_label}**")
            # Interval inspector uses the raw session (stream filter is
            # irrelevant; it re-fits TCSPC histograms directly).
            render_interval(session_raw, tcspc_source)

    with tab_phasor:
        if tcspc_source is None:
            st.warning(
                "Phasor explorer needs either a raw/\\*.iFLiP2 file or a "
                "tidy/\\*_data.csv file matching this session (for per-frame "
                "TCSPC histograms)."
            )
        else:
            st.caption(f"TCSPC source: **{state.tcspc_source_label}**")
            render_phasor(
                session_raw, tcspc_source,
                interval_range=st.session_state["interval_range"],
            )

    with tab_peth:
        # PETH uses the FILTERED session so smoothing propagates
        render_peth(session, filter_label=filter_label)


if __name__ == "__main__":
    main()
