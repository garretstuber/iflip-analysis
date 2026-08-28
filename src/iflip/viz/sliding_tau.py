"""Plotly figures for sliding-window τ traces."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from iflip.io.session_csv import SessionData
from iflip.preprocess.sliding_tau import SlidingTauResult
from iflip.viz.traces import EVENT_STYLES


def sliding_tau_figure(
    result: SlidingTauResult,
    *,
    session: SessionData | None = None,
    event_types: Sequence[str] | None = None,
    show_n_photons: bool = True,
    height: int = 500,
) -> go.Figure:
    """Two-panel plot: sliding τ trace (top) + photons-per-window (bottom).

    Optionally overlays event markers from a session, similar to
    :func:`iflip.viz.traces.session_traces_figure`.
    """
    rows = 2 if show_n_photons else 1
    row_heights = [0.7, 0.3] if show_n_photons else [1.0]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        subplot_titles=(
            f"τ ({result.method}) · window={result.window_s}s, step={result.step_s}s",
            "photons / window",
        )[:rows],
    )

    fig.add_trace(
        go.Scattergl(
            x=result.time,
            y=result.tau,
            mode="lines+markers",
            line=dict(color="#1f77b4", width=1.5),
            marker=dict(size=4),
            name="τ (ns)",
            hovertemplate="t=%{x:.2f}s<br>τ=%{y:.3f}ns<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # τ1 / τ2 overlays for fit_double
    if result.method == "fit_double" and "tau1" in result.extra:
        fig.add_trace(
            go.Scattergl(
                x=result.time,
                y=result.extra["tau1"],
                mode="lines",
                line=dict(color="#9467bd", width=1, dash="dot"),
                name="τ₁ (slow)",
                hovertemplate="t=%{x:.2f}s<br>τ₁=%{y:.3f}ns<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=result.time,
                y=result.extra["tau2"],
                mode="lines",
                line=dict(color="#2ca02c", width=1, dash="dot"),
                name="τ₂ (fast)",
                hovertemplate="t=%{x:.2f}s<br>τ₂=%{y:.3f}ns<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Event overlays
    if session is not None and not session.events.empty:
        if event_types is None:
            event_types = session.event_types
        finite_tau = result.tau[np.isfinite(result.tau)]
        if finite_tau.size > 0:
            tau_lo = float(np.min(finite_tau))
            tau_hi = float(np.max(finite_tau))
            marker_y = tau_hi + 0.04 * (tau_hi - tau_lo + 1e-6)
        else:
            marker_y = 1.0
        for i, ev in enumerate(event_types):
            sub = session.events[session.events["event_id_char"] == ev]
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
                        size=8,
                        line=dict(width=1, color="black"),
                    ),
                    name=ev,
                    hovertemplate=f"{ev}<br>t=%{{x:.2f}}s<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if show_n_photons:
        fig.add_trace(
            go.Scattergl(
                x=result.time,
                y=result.n_photons,
                mode="lines",
                line=dict(color="#444", width=1),
                name="photons",
                showlegend=False,
                hovertemplate="t=%{x:.2f}s<br>N=%{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="photons", row=2, col=1)

    fig.update_xaxes(title_text="time (s)", row=rows, col=1)
    fig.update_yaxes(title_text="τ (ns)", row=1, col=1)
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig
