"""Plotly figures for session-level intensity and lifetime traces."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from iflip.io.session_csv import SessionData

# Fixed mapping from event type -> colour/label so colours stay stable
# across re-renders.
EVENT_STYLES: dict[str, dict[str, str]] = {
    "solution_onset": {"color": "#1f77b4", "symbol": "triangle-down"},
    "lick": {"color": "#2ca02c", "symbol": "line-ns-open"},
    "lick_post_solution": {"color": "#d62728", "symbol": "line-ns-open"},
}


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    """Return an index array that thins ``n`` points to at most ``max_points``."""
    if n <= max_points:
        return np.arange(n)
    step = int(np.ceil(n / max_points))
    return np.arange(0, n, step)


def session_traces_figure(
    session: SessionData,
    *,
    max_points: int = 20_000,
    log_intensity: bool = False,
    event_types: Sequence[str] | None = None,
    highlight_range: tuple[float, float] | None = None,
    height: int = 520,
) -> go.Figure:
    """Two-panel plot: intensity (top) + lifetime (bottom), shared x-axis.

    Parameters
    ----------
    session
        Loaded :class:`SessionData`.
    max_points
        Downsample streams to at most this many points per trace (uses
        ``scattergl`` either way, but fewer points keeps zoom snappy).
    log_intensity
        Plot intensity on a log-y axis.
    event_types
        Which behavioural events to overlay. Defaults to all events in
        the session.
    highlight_range
        Optional ``(t_start, t_stop)`` tuple in seconds; drawn as a
        translucent shaded band spanning both panels.
    height
        Figure height in pixels.
    """
    streams = session.streams
    t = streams["time"].to_numpy()
    intensity = streams["intensity"].to_numpy()
    lifetime = streams["lifetime"].to_numpy()

    idx = _downsample_indices(len(t), max_points)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.55],
        subplot_titles=("Intensity (photons/frame)", "Lifetime (ns)"),
    )

    fig.add_trace(
        go.Scattergl(
            x=t[idx],
            y=intensity[idx],
            mode="lines",
            line=dict(color="#444", width=1),
            name="intensity",
            hovertemplate="t=%{x:.2f}s<br>I=%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=t[idx],
            y=lifetime[idx],
            mode="lines",
            line=dict(color="#1f77b4", width=1),
            name="lifetime",
            hovertemplate="t=%{x:.2f}s<br>τ=%{y:.3f}ns<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Event overlays as scatter markers on a dedicated strip at the top of
    # the lifetime panel (rather than vertical lines, which clutter the plot
    # when events are dense like licks).
    events = session.events
    if not events.empty:
        if event_types is None:
            event_types = session.event_types
        lifetime_range = _nice_range(lifetime)
        marker_y = lifetime_range[1] + 0.02 * (lifetime_range[1] - lifetime_range[0])
        for i, ev in enumerate(event_types):
            sub = events[events["event_id_char"] == ev]
            if sub.empty:
                continue
            style = EVENT_STYLES.get(
                ev, {"color": f"hsl({(i * 57) % 360},70%,40%)", "symbol": "circle"}
            )
            fig.add_trace(
                go.Scattergl(
                    x=sub["time"].to_numpy(),
                    y=np.full(len(sub), marker_y),
                    mode="markers",
                    marker=dict(
                        color=style["color"],
                        symbol=style["symbol"],
                        size=9,
                        line=dict(width=1, color="black"),
                    ),
                    name=ev,
                    hovertemplate=f"{ev}<br>t=%{{x:.2f}}s<extra></extra>",
                ),
                row=2,
                col=1,
            )

    if highlight_range is not None:
        t0, t1 = highlight_range
        fig.add_vrect(
            x0=t0,
            x1=t1,
            fillcolor="orange",
            opacity=0.18,
            line_width=0,
            row="all",
            col=1,
        )

    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_yaxes(title_text="photons", row=1, col=1, type="log" if log_intensity else "linear")
    fig.update_yaxes(title_text="τ (ns)", row=2, col=1)
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def _nice_range(x: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    """Return a (min, max) range with small padding for plotting."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    span = hi - lo if hi > lo else 1.0
    return (lo - pad * span, hi + pad * span)


def event_rate_histogram(
    session: SessionData,
    event_type: str,
    *,
    bin_width: float = 1.0,
) -> pd.DataFrame:
    """Compute a session-wide event rate histogram (for a QC sidebar).

    Returns a DataFrame with columns ``bin_start`` (s) and ``count``.
    """
    events = session.events
    sub = events[events["event_id_char"] == event_type]
    if sub.empty:
        return pd.DataFrame({"bin_start": [], "count": []})
    t_end = float(session.streams["time"].max())
    bins = np.arange(0.0, t_end + bin_width, bin_width)
    counts, _ = np.histogram(sub["time"].to_numpy(), bins=bins)
    return pd.DataFrame({"bin_start": bins[:-1], "count": counts})
