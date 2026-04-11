"""Plotly figures for phasor plots."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from flipr.preprocess.phasor import phasor_for_known_taus, semicircle_points


def phasor_plot_figure(
    *,
    real: np.ndarray | None = None,
    imag: np.ndarray | None = None,
    point_color: np.ndarray | None = None,
    point_label: str = "samples",
    point_hover: np.ndarray | None = None,
    reference_taus: list[float] | None = None,
    highlight: tuple[float, float] | None = None,
    period_ns: float = 12.5,
    title: str | None = None,
    height: int = 520,
) -> go.Figure:
    """Phasor plot with the universal semicircle, an optional cloud of
    per-frame points, and optional reference-lifetime markers.

    Parameters
    ----------
    real, imag
        ``(n,)`` arrays of phasor G, S coordinates to scatter. Pass ``None``
        to draw an empty plot with only the semicircle.
    point_color
        Optional ``(n,)`` scalar array used to colour the points (e.g. time
        in seconds, trial index, τ_phase). When provided a colorbar is
        drawn.
    point_label
        Legend label for the scatter trace.
    point_hover
        Optional ``(n,)`` hovertext strings.
    reference_taus
        Optional list of τ values (ns) to mark as diamonds on the
        semicircle — e.g. the slow/fast components from a double-exp fit.
    highlight
        Optional ``(G, S)`` tuple drawn as a single large star marker
        (e.g. the interval inspector's current window).
    period_ns
        Laser period used for the semicircle and reference computations.
    """
    fig = go.Figure()

    # Universal semicircle
    Gs, Ss = semicircle_points(201)
    fig.add_trace(
        go.Scatter(
            x=Gs,
            y=Ss,
            mode="lines",
            line=dict(color="#888", width=2),
            name="universal semicircle",
            hoverinfo="skip",
            showlegend=True,
        )
    )
    # Axis box: (0,0) to (1,0) baseline
    fig.add_shape(
        type="line",
        x0=0.0,
        x1=1.0,
        y0=0.0,
        y1=0.0,
        line=dict(color="#bbb", width=1, dash="dot"),
    )

    # Data cloud
    if real is not None and imag is not None and len(real) > 0:
        marker = dict(size=5, opacity=0.7)
        if point_color is not None:
            marker["color"] = point_color
            marker["colorscale"] = "Viridis"
            marker["colorbar"] = dict(title="time (s)")
            marker["showscale"] = True
        else:
            marker["color"] = "#1f77b4"

        hover = None
        if point_hover is not None:
            hover = list(point_hover)

        fig.add_trace(
            go.Scattergl(
                x=real,
                y=imag,
                mode="markers",
                marker=marker,
                name=point_label,
                text=hover,
                hovertemplate=(
                    "G=%{x:.3f}<br>S=%{y:.3f}"
                    + ("<br>%{text}" if hover is not None else "")
                    + "<extra></extra>"
                ),
            )
        )

    # Reference τ markers on the semicircle
    if reference_taus:
        G_ref, S_ref = phasor_for_known_taus(reference_taus, period_ns=period_ns)
        labels = [f"τ = {t:.2f} ns" for t in reference_taus]
        fig.add_trace(
            go.Scatter(
                x=G_ref,
                y=S_ref,
                mode="markers+text",
                marker=dict(
                    size=12,
                    symbol="diamond",
                    color="#d62728",
                    line=dict(width=1, color="black"),
                ),
                text=labels,
                textposition="top center",
                name="reference τ",
                hovertemplate="%{text}<br>G=%{x:.3f}<br>S=%{y:.3f}<extra></extra>",
            )
        )

    # Highlight point
    if highlight is not None:
        fig.add_trace(
            go.Scatter(
                x=[highlight[0]],
                y=[highlight[1]],
                mode="markers",
                marker=dict(
                    size=16, symbol="star", color="orange", line=dict(width=2, color="black")
                ),
                name="selected window",
                hovertemplate="G=%{x:.3f}<br>S=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_xaxes(title_text="G (real)", range=[-0.05, 1.05], constrain="domain")
    fig.update_yaxes(title_text="S (imag)", range=[-0.05, 0.6], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=60, r=20, t=40 if title else 20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
    )
    return fig
