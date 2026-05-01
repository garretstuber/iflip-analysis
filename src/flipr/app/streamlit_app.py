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
    load_iflip2,
)
from flipr.io.session_csv import (
    AcquisitionEntry,
    SessionData,
    discover_acquisitions,
    load_session,
    session_from_tcspc_source,
)
from flipr.io.tidy_csv import (
    TidyData,
    load_tidy,
)
from flipr.preprocess.background import (
    BackgroundEstimate,
    compute_background,
    subtract_background,
)
from flipr.preprocess.filters import FILTER_MODES, filtered_session
from flipr.preprocess.lifetime import (
    DoubleExpFit,
    LifetimeFit,
    SingleExpFit,
    fit_double_exp,
    fit_information_criteria,
    fit_single_exp,
)
from flipr.preprocess.motion import QCResult, compute_qc
from flipr.preprocess.phasor import phasor_from_histogram, phasor_series_from_tcspc
from flipr.preprocess.sliding_tau import (
    SlidingTauResult,
    sliding_tau,
)
from flipr.viz.peth import peth_figure
from flipr.viz.phasor import phasor_plot_figure
from flipr.viz.sliding_tau import sliding_tau_figure
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
def cached_fit_double(
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


@st.cache_data(show_spinner="Fitting single-exponential…")
def cached_fit_single(
    tcspc: np.ndarray,
    bins: np.ndarray,
    irf_sigma: float,
    t0: float,
    fit_start: float,
    fit_stop: float,
) -> SingleExpFit:
    return fit_single_exp(
        tcspc,
        bins,
        irf_sigma_ns=irf_sigma,
        t0_ns=t0,
        fit_start_ns=fit_start,
        fit_stop_ns=fit_stop,
    )


@st.cache_data(show_spinner="Computing background…")
def cached_compute_background(
    tcspc: np.ndarray,
    bins_ns: np.ndarray,
    *,
    time_index: np.ndarray | None = None,
    time_window: tuple[float, float] | None = None,
    source_label: str = "background",
) -> BackgroundEstimate:
    return compute_background(
        tcspc, bins_ns,
        time_index=time_index,
        time_window=time_window,
        source_label=source_label,
    )


@st.cache_data(show_spinner="Sliding-window τ…")
def cached_sliding_tau(
    tcspc: np.ndarray,
    bins_ns: np.ndarray,
    fs: float,
    *,
    method: str,
    window_s: float,
    step_s: float,
    period_ns: float,
    bg_per_frame: np.ndarray | None,
    bg_scale: float,
    irf_sigma_ns: float,
    t0_ns: float,
    fit_start_ns: float,
    fit_stop_ns: float,
) -> SlidingTauResult:
    bg: BackgroundEstimate | None = None
    if bg_per_frame is not None:
        bg = BackgroundEstimate(
            per_frame=bg_per_frame,
            bins_ns=bins_ns,
            n_frames=1,  # informational only
            source_label="cached",
        )
    fit_kwargs = {
        "irf_sigma_ns": irf_sigma_ns,
        "t0_ns": t0_ns,
        "fit_start_ns": fit_start_ns,
        "fit_stop_ns": fit_stop_ns,
    }
    return sliding_tau(
        tcspc, bins_ns, fs,
        method=method,  # type: ignore[arg-type]
        window_s=window_s,
        step_s=step_s,
        period_ns=period_ns,
        background=bg,
        bg_scale=bg_scale,
        fit_kwargs=fit_kwargs,
    )


def _cached_fit(
    model: str,
    tcspc: np.ndarray,
    bins: np.ndarray,
    irf_sigma: float,
    t0: float,
    fit_start: float,
    fit_stop: float,
) -> LifetimeFit:
    if model == "single":
        return cached_fit_single(tcspc, bins, irf_sigma, t0, fit_start, fit_stop)
    return cached_fit_double(tcspc, bins, irf_sigma, t0, fit_start, fit_stop)


@st.cache_data(show_spinner="Computing phasor trajectory…")
def cached_phasor_series(
    tcspc: np.ndarray,
    bins: np.ndarray,
    period_ns: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    return phasor_series_from_tcspc(tcspc, bins, period_ns=period_ns)


def _compute_qc_for_session(
    session: SessionData,
    filter_mode: str,
    window_s: float,
    polyorder: int,
    motion_window_s: float,
    motion_corr_threshold: float,
) -> QCResult:
    """Compute QC metrics from a (possibly synthetic) SessionData."""
    filt = filtered_session(
        session,
        mode=filter_mode,
        window_s=window_s,
        polyorder=polyorder,  # type: ignore[arg-type]
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
    background: BackgroundEstimate | None = None
    bg_scale: float = 1.0


def _background_picker(
    container,
    acquisitions: list[AcquisitionEntry],
    current: AcquisitionEntry,
    tcspc_source: TCSPCSource | None,
) -> tuple[BackgroundEstimate | None, float]:
    """Sidebar UI for picking a background source.

    Returns ``(background, scale)`` where ``background`` is None when
    the feature is off. The scale lets the user compensate for
    differences in laser power/coupling between bg and test recordings.
    """
    with container.expander("Background subtraction", expanded=False):
        st.caption(
            "Subtract autofluorescence + dark counts from TCSPC histograms "
            "before fitting / phasor / sliding-τ. Pick a bg acquisition or "
            "use a quiet pre-stimulus window of the current recording."
        )
        mode = st.radio(
            "bg source",
            options=["off", "another acquisition", "current acquisition window"],
            index=0,
            key="bg_mode",
            help="**off**: no subtraction. "
            "**another acquisition**: load a separate (typically `bg*`) "
            "recording and subtract its mean per-frame histogram. "
            "**current acquisition window**: derive the bg from a "
            "user-defined time window of the loaded session.",
        )
        bg: BackgroundEstimate | None = None

        if mode == "another acquisition":
            # Prefer entries with "bg" in the name, then everything else
            candidates = [a for a in acquisitions if a.blockname != current.blockname
                          and (a.iflip2_path is not None or a.tidy_data_path is not None)]
            if not candidates:
                st.warning("No other acquisitions available.")
            else:
                # Sort: bg* first, then alphabetical
                bg_first = sorted(
                    candidates,
                    key=lambda a: (0 if "bg" in a.blockname.lower() else 1, a.blockname),
                )
                names = [a.blockname for a in bg_first]
                idx = st.selectbox(
                    "bg acquisition",
                    options=range(len(bg_first)),
                    format_func=lambda i: names[i] + (
                        " (bg)" if "bg" in names[i].lower() else ""
                    ),
                    index=0,
                )
                chosen = bg_first[idx]
                try:
                    if chosen.iflip2_path is not None:
                        bg_src: TCSPCSource = cached_load_iflip2(str(chosen.iflip2_path))
                    elif chosen.tidy_data_path is not None:
                        bg_src = cached_load_tidy(str(chosen.tidy_data_path))
                    else:
                        bg_src = None  # type: ignore[assignment]
                    if bg_src is not None:
                        bg = cached_compute_background(
                            bg_src.tcspc.astype(np.float64),
                            bg_src.tcspc_bins_ns,
                            source_label=chosen.blockname,
                        )
                        st.caption(
                            f"Loaded `{chosen.blockname}` "
                            f"({bg.n_frames} frames, "
                            f"{bg.total_per_frame:.1f} photons/frame avg)."
                        )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to load bg: {exc}")

        elif mode == "current acquisition window":
            if tcspc_source is None:
                st.warning("Need a TCSPC source loaded first.")
            else:
                t_max = float(tcspc_source.streams["time"].max())
                bg_window = st.slider(
                    "bg time window (s)",
                    min_value=0.0,
                    max_value=t_max,
                    value=(0.0, min(5.0, t_max)),
                    step=0.5,
                    key="bg_window",
                    help="Use a quiet pre-stimulus stretch of the current "
                    "recording as the background reference.",
                )
                try:
                    bg = cached_compute_background(
                        tcspc_source.tcspc.astype(np.float64),
                        tcspc_source.tcspc_bins_ns,
                        time_index=tcspc_source.streams["time"].to_numpy(),
                        time_window=tuple(bg_window),
                        source_label=f"{current.blockname} t∈[{bg_window[0]:.1f},{bg_window[1]:.1f}]s",
                    )
                    st.caption(f"{bg.n_frames} frames averaged.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to compute bg: {exc}")

        scale = 1.0
        if bg is not None:
            scale = float(
                st.number_input(
                    "scale",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.05,
                    key="bg_scale",
                    help="Multiplier on the subtracted bg. Use 1.0 if the "
                    "bg recording was at matched laser power and fiber "
                    "coupling.",
                )
            )

    return bg, scale


def _load_custom_file(path_str: str, empty_state: SidebarState) -> SidebarState:
    """Load a single tidy CSV or .iFLiP2 file by absolute path and build a
    minimal SessionData around it. Bypasses the data root / discovery flow.

    Returns a fully-populated :class:`SidebarState` ready to feed the tabs,
    or ``empty_state`` (with an error in the sidebar) on failure.
    """
    path = Path(path_str).expanduser()
    if not path.is_file():
        st.sidebar.error(f"File not found: {path}")
        return empty_state

    name = path.name.lower()
    tcspc_source: TCSPCSource | None = None
    tcspc_source_label = ""
    try:
        if name.endswith(".iflip2"):
            tcspc_source = cached_load_iflip2(str(path))
            tcspc_source_label = f"raw · {path.name}"
        elif name.endswith("_data.csv"):
            tcspc_source = cached_load_tidy(str(path))
            tcspc_source_label = f"tidy · {path.name}"
        elif name.endswith(".csv"):
            # Try as a tidy data CSV anyway (will error if it's not)
            tcspc_source = cached_load_tidy(str(path))
            tcspc_source_label = f"tidy · {path.name}"
        else:
            st.sidebar.error(
                f"Unsupported file type: {path.name}. "
                "Expected .iFLiP2 or *_data.csv."
            )
            return empty_state
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Failed to load `{path.name}`: {exc}")
        return empty_state

    # Build a minimal SessionData around the source
    try:
        # Strip filename → blockname guess
        from flipr.io.session_csv import _blockname_from_iflip2, _blockname_from_tidy

        bn_guess = _blockname_from_tidy(path.name) or _blockname_from_iflip2(path.name)
        session_raw = session_from_tcspc_source(tcspc_source, blockname=bn_guess)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Failed to build session: {exc}")
        return empty_state

    st.sidebar.success(f"Loaded custom file: `{path.name}`")
    st.sidebar.caption(f"Blockname: `{session_raw.blockname}`")

    # Filter controls (same UI as the discovery flow)
    st.sidebar.divider()
    st.sidebar.markdown("**Stream filter**")
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
            min_value=0.05,
            max_value=10.0,
            value=0.25,
            step=0.05,
            key="filter_window",
        )
        if filter_mode == "savgol":
            polyorder = int(
                st.sidebar.number_input(
                    "polyorder",
                    min_value=1,
                    max_value=5,
                    value=2,
                    step=1,
                    key="filter_polyorder",
                )
            )
        else:
            polyorder = 2

    session = filtered_session(
        session_raw,
        mode=filter_mode,  # type: ignore[arg-type]
        window_s=float(window_s),
        polyorder=int(polyorder),
    )

    return SidebarState(
        data_root=path.parent,
        session_raw=session_raw,
        session=session,
        session_path=None,
        tcspc_source=tcspc_source,
        tcspc_source_label=tcspc_source_label,
        filter_mode=str(filter_mode),
        filter_window_s=float(window_s),
        filter_polyorder=int(polyorder),
        qc_motion_window_s=1.0,
        qc_motion_corr_threshold=0.85,
    )


def sidebar() -> SidebarState:
    if LOGO_PATH.is_file():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.sidebar.title("FLIPR analysis")
    st.sidebar.caption("v0.0.1")

    empty_state = SidebarState(
        data_root=None,
        session_raw=None,
        session=None,
        session_path=None,
        tcspc_source=None,
        tcspc_source_label="",
        filter_mode="none",
        filter_window_s=0.25,
        filter_polyorder=2,
        qc_motion_window_s=1.0,
        qc_motion_corr_threshold=0.85,
    )

    default = str(DEFAULT_DATA_ROOT) if DEFAULT_DATA_ROOT.is_dir() else ""
    data_root_str = st.sidebar.text_input(
        "Data root",
        value=default,
        help="Directory containing tidy/, raw/, and/or sessions/ subfolders",
    )

    # --- Custom file picker (load any file by absolute path) ---
    with st.sidebar.expander("Load file directly", expanded=False):
        st.caption(
            "Load a single tidy `*_data.csv` or `.iFLiP2` file by absolute "
            "path (bypasses the data root and dropdown). Useful for "
            "one-off files outside your usual data folder."
        )
        custom_path_str = st.text_input(
            "File path",
            value="",
            placeholder="/path/to/file.iFLiP2 or /path/to/file_data.csv",
            key="custom_file_path",
        )
        # File uploader as alternate UX (limited to small files)
        uploaded = st.file_uploader(
            "…or drop a file",
            type=["iFLiP2", "iflip2", "csv"],
            help=(
                "Drop a `.iFLiP2` file or a tidy `_data.csv` file. "
                "For tidy CSVs, the matching `_param.csv` must live "
                "in the same folder as the data file (so prefer the "
                "path text box above for tidy CSVs)."
            ),
            key="custom_uploader",
        )
        if uploaded is not None and not custom_path_str:
            # Persist the upload to a temp path so loaders can read it
            import tempfile

            tmp = Path(tempfile.gettempdir()) / uploaded.name
            tmp.write_bytes(uploaded.getvalue())
            custom_path_str = str(tmp)
            st.caption(f"Saved upload to `{tmp}` (will be reloaded if you re-run).")

    if custom_path_str:
        return _load_custom_file(custom_path_str, empty_state)

    if not data_root_str:
        st.sidebar.warning("Enter a FLIPR data root or load a file directly.")
        return empty_state
    data_root = Path(data_root_str).expanduser()
    if not data_root.is_dir():
        st.sidebar.error(f"Not a directory: {data_root}")
        return empty_state

    # Unified discovery across sessions/, tidy/, and raw/
    acquisitions = discover_acquisitions(data_root)
    if not acquisitions:
        st.sidebar.warning(
            f"No acquisitions found under {data_root}. "
            "Expected tidy/*_data.csv, raw/*.iFLiP2, or sessions/<block>/ folders."
        )
        return replace(empty_state, data_root=data_root)

    # Show a label hint for each entry
    acq_labels = []
    for a in acquisitions:
        parts = []
        if a.has_session:
            parts.append("session")
        if a.iflip2_path is not None:
            parts.append("raw")
        if a.tidy_data_path is not None:
            parts.append("tidy")
        acq_labels.append(f"{a.blockname}  ({', '.join(parts)})")

    picked_idx = st.sidebar.selectbox(
        "Acquisition",
        options=range(len(acquisitions)),
        format_func=lambda i: acq_labels[i],
        index=len(acquisitions) - 1,
    )
    picked: AcquisitionEntry = acquisitions[picked_idx]

    # --- Load or build the SessionData ---
    session_raw: SessionData | None = None
    session_path: Path | None = None
    tcspc_source: TCSPCSource | None = None
    tcspc_source_label: str = ""

    # Determine available TCSPC sources
    source_options: list[str] = []
    if picked.iflip2_path is not None:
        source_options.append("raw .iFLiP2")
    if picked.tidy_data_path is not None:
        source_options.append("tidy CSV")

    # Choose TCSPC source
    if not source_options:
        st.sidebar.warning(
            "No matching raw/ or tidy/ file for this acquisition — "
            "Interval Inspector, Phasor, and QC are disabled."
        )
    else:
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
            if chosen == "raw .iFLiP2" and picked.iflip2_path is not None:
                tcspc_source = cached_load_iflip2(str(picked.iflip2_path))
                tcspc_source_label = f"raw · {picked.iflip2_path.name}"
                st.sidebar.caption(f"Raw file: `{picked.iflip2_path.name}`")
            elif chosen == "tidy CSV" and picked.tidy_data_path is not None:
                tcspc_source = cached_load_tidy(str(picked.tidy_data_path))
                tcspc_source_label = f"tidy · {picked.tidy_data_path.name}"
                st.sidebar.caption(f"Tidy file: `{picked.tidy_data_path.name}`")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Failed to load {chosen}: {exc}")
            tcspc_source = None

    # Load or build the SessionData
    if picked.has_session and picked.session_path is not None:
        # Full session folder available
        session_path = picked.session_path
        try:
            session_raw = cached_load_session(str(session_path))
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Failed to load session: {exc}")
            return replace(empty_state, data_root=data_root)
    elif tcspc_source is not None:
        # No session folder — build a minimal SessionData from the TCSPC source
        try:
            session_raw = session_from_tcspc_source(tcspc_source, blockname=picked.blockname)
            session_path = None
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Failed to build session from TCSPC source: {exc}")
            return replace(empty_state, data_root=data_root)
    else:
        st.sidebar.error("No loadable data for this acquisition.")
        return replace(empty_state, data_root=data_root)

    # --- Background subtraction (optional) ---
    st.sidebar.divider()
    background, bg_scale = _background_picker(
        st.sidebar, acquisitions, picked, tcspc_source
    )

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
            min_value=0.05,
            max_value=10.0,
            value=0.25,
            step=0.05,
            key="filter_window",
        )
        if filter_mode == "savgol":
            polyorder = int(
                st.sidebar.number_input(
                    "polyorder",
                    min_value=1,
                    max_value=5,
                    value=2,
                    step=1,
                    key="filter_polyorder",
                )
            )
        else:
            polyorder = 2

    session = filtered_session(
        session_raw,
        mode=filter_mode,  # type: ignore[arg-type]
        window_s=float(window_s),
        polyorder=int(polyorder),
    )

    # --- QC correlation-diagnostic controls ---
    with st.sidebar.expander("QC · correlation diagnostic", expanded=False):
        st.caption(
            "Rolling r(intensity, lifetime) is informational only — "
            "see the Overview tab. Not used for flagging."
        )
        qc_motion_window_s = st.number_input(
            "rolling window (s)",
            min_value=0.1,
            max_value=30.0,
            value=1.0,
            step=0.1,
            key="qc_motion_window",
        )
        qc_motion_corr_threshold = st.slider(
            "highlight |r| above",
            min_value=0.3,
            max_value=0.99,
            value=0.85,
            step=0.01,
            key="qc_motion_threshold",
        )

    # Session metadata block
    with st.sidebar.expander("Session metadata", expanded=False):
        meta_dict: dict[str, object] = {
            "subject": session_raw.subject,
            "procedure": session_raw.procedure,
            "fs (Hz)": session_raw.fs,
            "blockname": session_raw.blockname,
            "duration (s)": float(session_raw.streams["time"].max()),
            "n samples": int(len(session_raw.streams)),
        }
        if session_raw.event_types:
            meta_dict["event types"] = session_raw.event_types
        if not picked.has_session:
            meta_dict["source"] = "tidy/raw only (no session folder)"
        st.write(meta_dict)
        note = session_raw.meta.get("note_general", "")
        if note and note != "NA":
            st.caption(f"Note: {note}")

    return SidebarState(
        data_root=data_root,
        session_raw=session_raw,
        session=session,
        session_path=session_path,
        tcspc_source=tcspc_source,
        tcspc_source_label=tcspc_source_label,
        filter_mode=str(filter_mode),
        filter_window_s=float(window_s),
        filter_polyorder=int(polyorder),
        qc_motion_window_s=float(qc_motion_window_s),
        qc_motion_corr_threshold=float(qc_motion_corr_threshold),
        background=background,
        bg_scale=float(bg_scale),
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
    mc2.metric(
        "intensity CV", f"{streams['intensity'].std() / max(streams['intensity'].mean(), 1):.3f}"
    )
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
        qc_cols[0].metric("flagged (any)", f"{summary['any'] * 100:.2f}%")
        qc_cols[1].metric("low photons", f"{summary['intensity'] * 100:.2f}%")
        qc_cols[2].metric("intensity jumps", f"{summary['jump'] * 100:.2f}%")
        qc_cols[3].metric(
            "|rolling r| above threshold (diagnostic)",
            f"{summary['corr_above_threshold'] * 100:.2f}%",
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
    if session.event_types:
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
def render_interval(
    session: SessionData,
    tidy: TCSPCSource,
    *,
    background: BackgroundEstimate | None = None,
    bg_scale: float = 1.0,
) -> None:
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

    hist_raw = tidy.integrate_histogram(t0_window, t1_window).astype(np.float64)
    if hist_raw.sum() == 0:
        st.error("no photons in the selected window")
        return t_range

    # Apply background subtraction (if enabled)
    if background is not None:
        # n_frames in the window = duration * fs
        t_axis = tidy.streams["time"].to_numpy()
        n_frames_window = int(((t_axis >= t0_window) & (t_axis <= t1_window)).sum())
        hist = subtract_background(
            hist_raw, background, n_frames=n_frames_window, scale=bg_scale
        )
        bg_subtracted_total = float(
            bg_scale * n_frames_window * background.per_frame.sum()
        )
        st.caption(
            f"Background subtraction: source `{background.source_label}` × "
            f"{n_frames_window} frames × {bg_scale:.2f} = "
            f"{bg_subtracted_total:,.0f} photons removed"
        )
    else:
        hist = hist_raw

    # Fit params default to the session's meta
    default_irf = float(session.meta.get("IRF", "0.212"))
    default_t0 = float(session.meta.get("t0", "1.012"))
    default_start = float(tidy.params.get("header_state_spcrangelow", "0.4"))
    default_stop = float(tidy.params.get("header_state_spcrangehigh", "12.3"))

    fit_col1, fit_col2 = st.columns([1, 3])
    with fit_col1:
        fit_model = st.radio(
            "fit model",
            options=["double", "single"],
            index=0,
            horizontal=True,
            help=(
                "**double** = sum of two EMG components (slow + fast). "
                "Standard for FLIM-DA0.5 and most FLIM biosensors. "
                "**single** = one EMG component. Use for reference dyes "
                "or as a null model for AIC/BIC comparison."
            ),
            key="fit_model",
        )
    with fit_col2:
        compare_models = st.checkbox(
            "compare to alternative model (AIC/BIC)",
            value=False,
            help="Fit both single and double exp and report AIC/BIC and ΔAIC. "
            "ΔAIC > ~10 in favour of one model is decisive.",
        )

    with st.expander("fit settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        irf = c1.number_input(
            "IRF σ (ns)", value=default_irf, min_value=0.05, max_value=2.0, step=0.01
        )
        t0 = c2.number_input("t₀ (ns)", value=default_t0, min_value=0.0, max_value=5.0, step=0.01)
        fit_lo = c3.number_input(
            "fit start (ns)", value=default_start, min_value=0.0, max_value=10.0, step=0.1
        )
        fit_hi = c4.number_input(
            "fit stop (ns)", value=default_stop, min_value=1.0, max_value=12.5, step=0.1
        )

    fit = _cached_fit(fit_model, hist, tidy.tcspc_bins_ns, irf, t0, fit_lo, fit_hi)

    title = (
        f"window: t ∈ [{t0_window:.2f}, {t1_window:.2f}] s   ·   "
        f"{int(hist.sum()):,} photons   ·   {fit_model}-exp fit"
    )
    fig = tcspc_decay_figure(hist, tidy.tcspc_bins_ns, fit=fit, log_y=True, title=title, height=560)
    st.plotly_chart(fig, use_container_width=True)

    # Optional AIC/BIC comparison row
    if compare_models:
        alt_model = "single" if fit_model == "double" else "double"
        alt_fit = _cached_fit(alt_model, hist, tidy.tcspc_bins_ns, irf, t0, fit_lo, fit_hi)
        # n_params: single has 3 free params (α, τ, bg) when t0/σ fixed,
        # double has 5 (α1, τ1, α2, τ2, bg).
        n_params_main = 3 if fit_model == "single" else 5
        n_params_alt = 3 if alt_model == "single" else 5
        ic_main = fit_information_criteria(fit, n_params_main)
        ic_alt = fit_information_criteria(alt_fit, n_params_alt)
        delta_aic = ic_main["aic"] - ic_alt["aic"]
        delta_bic = ic_main["bic"] - ic_alt["bic"]
        st.markdown("**Model comparison**")
        comp_cols = st.columns(5)
        comp_cols[0].metric(f"AIC ({fit_model})", f"{ic_main['aic']:.1f}")
        comp_cols[1].metric(f"AIC ({alt_model})", f"{ic_alt['aic']:.1f}")
        comp_cols[2].metric(
            "ΔAIC",
            f"{delta_aic:+.1f}",
            help=f"AIC({fit_model}) − AIC({alt_model}). Negative → {fit_model} preferred.",
        )
        comp_cols[3].metric(f"χ² ({fit_model})", f"{fit.chi2_reduced:.2f}")
        comp_cols[4].metric(f"χ² ({alt_model})", f"{alt_fit.chi2_reduced:.2f}")
        if abs(delta_aic) >= 10:
            preferred = fit_model if delta_aic < 0 else alt_model
            st.success(
                f"AIC strongly prefers the **{preferred}-exponential** model "
                f"(|ΔAIC| = {abs(delta_aic):.1f} ≥ 10)."
            )
        elif abs(delta_aic) >= 2:
            preferred = fit_model if delta_aic < 0 else alt_model
            st.info(
                f"AIC mildly favours the **{preferred}-exponential** model "
                f"(|ΔAIC| = {abs(delta_aic):.1f}). Inspect residuals before deciding."
            )
        else:
            st.info(
                f"AIC is inconclusive between models (|ΔAIC| = {abs(delta_aic):.1f} < 2). "
                "Both models fit the data comparably well; prefer the simpler one."
            )
        st.caption(f"ΔBIC = {delta_bic:+.1f} (lower magnitude needed to swap models than AIC).")

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
    *,
    background: BackgroundEstimate | None = None,
    bg_scale: float = 1.0,
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

    # Full-session phasor trajectory. If a background is set, subtract it
    # from each frame before computing per-frame phasor coordinates.
    tcspc_for_phasor = tidy.tcspc.astype(np.float64)
    if background is not None:
        from flipr.preprocess.background import subtract_background_per_frame

        tcspc_for_phasor = subtract_background_per_frame(
            tcspc_for_phasor, background, scale=bg_scale
        )
        st.caption(
            f"Per-frame bg subtracted (`{background.source_label}` × {bg_scale:.2f})."
        )
    real, imag, mean, freq = cached_phasor_series(
        tcspc_for_phasor, tidy.tcspc_bins_ns, period_ns
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
        if background is not None:
            t_axis_full = tidy.streams["time"].to_numpy()
            n_frames_window = int(
                ((t_axis_full >= interval_range[0]) & (t_axis_full <= interval_range[1])).sum()
            )
            window_hist = subtract_background(
                window_hist, background, n_frames=n_frames_window, scale=bg_scale
            )
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
            "baseline start (s)",
            min_value=-60.0,
            max_value=0.0,
            value=max(float(pre_window), -2.5),
            step=0.1,
            disabled=(norm_mode == "raw"),
        )
    with col_bl_hi:
        bl_hi = st.number_input(
            "baseline end (s)",
            min_value=-60.0,
            max_value=5.0,
            value=-0.1,
            step=0.1,
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
# Tab 5: Sliding-window τ
# -----------------------------------------------------------------------------
def render_sliding_tau(
    session: SessionData,
    tidy: TCSPCSource,
    *,
    background: BackgroundEstimate | None = None,
    bg_scale: float = 1.0,
) -> None:
    st.subheader("Sliding-window τ")
    st.caption(
        "Sweep a fixed-width window across the session and compute one "
        "lifetime per step. Phasor methods are fast (live recompute on "
        "parameter change). Fit methods are accurate but slow — pick a "
        "coarse step (≥1 s) when fitting."
    )

    method_label_map = {
        "phasor_phase": "phasor · τ_phase (fast)",
        "phasor_mod": "phasor · τ_modulation (fast)",
        "fit_double": "fit · double-exp τ_amp (slow)",
        "fit_single": "fit · single-exp τ (slow)",
    }
    cmd_method, cmd_window, cmd_step = st.columns([2, 1, 1])
    with cmd_method:
        method = st.selectbox(
            "method",
            options=list(method_label_map.keys()),
            format_func=lambda k: method_label_map[k],
            index=0,
            key="sliding_method",
        )
    with cmd_window:
        window_s = st.number_input(
            "window (s)", min_value=0.1, max_value=60.0, value=1.0, step=0.1,
            key="sliding_window",
        )
    with cmd_step:
        # Fits are expensive; default step matches window for those.
        default_step = float(window_s) if str(method).startswith("fit") else max(0.5, float(window_s) / 2)
        step_s = st.number_input(
            "step (s)", min_value=0.05, max_value=60.0, value=default_step, step=0.05,
            key="sliding_step",
        )

    fit_lo_default = float(tidy.params.get("header_state_spcrangelow", "0.4"))
    fit_hi_default = float(tidy.params.get("header_state_spcrangehigh", "12.3"))
    irf_default = float(session.meta.get("IRF", "0.212"))
    t0_default = float(session.meta.get("t0", "1.012"))

    if str(method).startswith("fit"):
        with st.expander("fit settings (slower)", expanded=False):
            fc1, fc2, fc3, fc4 = st.columns(4)
            irf_default = fc1.number_input(
                "IRF σ (ns)", value=irf_default, min_value=0.05, max_value=2.0, step=0.01,
                key="sliding_irf",
            )
            t0_default = fc2.number_input(
                "t₀ (ns)", value=t0_default, min_value=0.0, max_value=5.0, step=0.01,
                key="sliding_t0",
            )
            fit_lo_default = fc3.number_input(
                "fit start (ns)", value=fit_lo_default, min_value=0.0, max_value=10.0, step=0.1,
                key="sliding_fit_lo",
            )
            fit_hi_default = fc4.number_input(
                "fit stop (ns)", value=fit_hi_default, min_value=1.0, max_value=12.5, step=0.1,
                key="sliding_fit_hi",
            )

    # Compute the trace
    bin_step = float(tidy.tcspc_bins_ns[1] - tidy.tcspc_bins_ns[0])
    period_ns = round(bin_step * (int(12.5 / bin_step)), 4)

    fs = float(tidy.params.get("header_state_samplingfreq", "20.0"))
    if not np.isfinite(fs) or fs <= 0:
        fs = float(session.fs) if session.fs > 0 else 20.0

    bg_per_frame = background.per_frame if background is not None else None

    try:
        result = cached_sliding_tau(
            tidy.tcspc.astype(np.float64),
            tidy.tcspc_bins_ns,
            fs,
            method=str(method),
            window_s=float(window_s),
            step_s=float(step_s),
            period_ns=period_ns,
            bg_per_frame=bg_per_frame,
            bg_scale=float(bg_scale),
            irf_sigma_ns=float(irf_default),
            t0_ns=float(t0_default),
            fit_start_ns=float(fit_lo_default),
            fit_stop_ns=float(fit_hi_default),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    if background is not None:
        st.caption(
            f"Background: `{background.source_label}` × {bg_scale:.2f} "
            f"({background.total_per_frame:.1f} photons/frame avg subtracted)."
        )

    # Event types to overlay
    ev_types = (
        st.multiselect(
            "events to overlay",
            options=session.event_types,
            default=session.event_types,
            key="sliding_events",
        )
        if session.event_types
        else None
    )

    fig = sliding_tau_figure(
        result,
        session=session,
        event_types=ev_types,
        show_n_photons=True,
        height=540,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    finite = np.isfinite(result.tau)
    if finite.sum() == 0:
        st.warning("No finite τ values — check window size, photon counts, or fit settings.")
        return

    tau_mean = float(np.mean(result.tau[finite]))
    tau_std = float(np.std(result.tau[finite]))
    n_steps_total = result.n_steps
    n_steps_ok = int(finite.sum())
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("n windows", f"{n_steps_total}")
    mc2.metric("n with finite τ", f"{n_steps_ok}")
    mc3.metric("mean τ (ns)", f"{tau_mean:.3f}")
    mc4.metric("τ std (ns)", f"{tau_std:.3f}")

    # Optional CSV download
    import io

    csv_buf = io.StringIO()
    df_out = pd.DataFrame(
        {
            "time_s": result.time,
            "tau_ns": result.tau,
            "n_photons": result.n_photons,
            **result.extra,
        }
    )
    df_out.to_csv(csv_buf, index=False)
    st.download_button(
        "Download trace CSV",
        data=csv_buf.getvalue(),
        file_name=f"sliding_tau_{session.blockname}_{method}_w{window_s}s_s{step_s}s.csv",
        mime="text/csv",
    )


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

    # Title: show subject if available, otherwise just blockname
    if session_raw.subject and session_raw.subject != session_raw.blockname:
        st.title(f"{session_raw.subject} · {session_raw.blockname}")
    else:
        st.title(session_raw.blockname)

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
        qc = _compute_qc_for_session(
            session_raw,
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

    tab_overview, tab_interval, tab_phasor, tab_sliding, tab_peth = st.tabs(
        [
            "Session overview",
            "Interval inspector",
            "Phasor explorer",
            "Sliding τ",
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
            render_interval(
                session_raw,
                tcspc_source,
                background=state.background,
                bg_scale=state.bg_scale,
            )

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
                session_raw,
                tcspc_source,
                interval_range=st.session_state["interval_range"],
                background=state.background,
                bg_scale=state.bg_scale,
            )

    with tab_sliding:
        if tcspc_source is None:
            st.warning(
                "Sliding τ needs either a raw/\\*.iFLiP2 file or a "
                "tidy/\\*_data.csv file matching this session (for per-frame "
                "TCSPC histograms)."
            )
        else:
            st.caption(f"TCSPC source: **{state.tcspc_source_label}**")
            render_sliding_tau(
                session_raw,
                tcspc_source,
                background=state.background,
                bg_scale=state.bg_scale,
            )

    with tab_peth:
        # PETH uses the FILTERED session so smoothing propagates
        render_peth(session, filter_label=filter_label)


if __name__ == "__main__":
    main()
