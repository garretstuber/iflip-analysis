"""Plotly figures for event-aligned PETH matrices."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flipr.align.peth import PETH


def peth_figure(
    peth: PETH,
    *,
    colorscale: str = "Viridis",
    mean_line_color: str = "#1f77b4",
    ribbon_color: str = "rgba(31, 119, 180, 0.25)",
    title: str | None = None,
    height: int = 620,
) -> go.Figure:
    """Two-panel PETH figure: per-trial heatmap on top, mean ± SEM on bottom.

    Both panels share the x-axis (relative time in seconds).
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            f"per-trial {peth.signal_name}",
            f"mean ± SEM (n = {peth.n_trials} trials)",
        ),
    )

    # Heatmap
    # The y-axis is trial index; values are NaN-safe
    fig.add_trace(
        go.Heatmap(
            x=peth.time_rel,
            y=peth.trial_ids,
            z=peth.values,
            colorscale=colorscale,
            colorbar=dict(title=peth.signal_name, thickness=14),
            zsmooth=False,
            hovertemplate=(
                f"trial %{{y}}<br>t=%{{x:.2f}}s<br>{peth.signal_name}=%{{z:.4f}}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Mean trace with SEM ribbon
    mean = peth.mean()
    sem = peth.sem()
    lower = mean - sem
    upper = mean + sem

    # Ribbon: upper first then lower with fill='tonexty'
    fig.add_trace(
        go.Scatter(
            x=peth.time_rel,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=peth.time_rel,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=ribbon_color,
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=peth.time_rel,
            y=mean,
            mode="lines",
            line=dict(color=mean_line_color, width=2),
            name="mean",
            hovertemplate="t=%{x:.2f}s<br>mean=%{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Event marker at t = 0 on both panels
    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="#d62728",
        line_width=2,
        row="all",
        col=1,
    )

    fig.update_xaxes(title_text="time from event (s)", row=2, col=1)
    fig.update_yaxes(title_text="trial", row=1, col=1, autorange="reversed")
    fig.update_yaxes(title_text=peth.signal_name, row=2, col=1)

    fig.update_layout(
        title=title or f"{peth.event_type} · {peth.signal_name}",
        height=height,
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode="x unified",
        showlegend=False,
    )
    return fig
