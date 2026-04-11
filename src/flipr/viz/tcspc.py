"""Plotly figures for TCSPC decay curves and double-exponential fits."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flipr.preprocess.lifetime import DoubleExpFit


def tcspc_decay_figure(
    histogram: np.ndarray,
    bin_ns: np.ndarray,
    *,
    fit: DoubleExpFit | None = None,
    log_y: bool = True,
    title: str | None = None,
    height: int = 520,
) -> go.Figure:
    """Two-panel plot: TCSPC decay (top) + weighted residuals (bottom).

    Parameters
    ----------
    histogram
        ``(n_bins,)`` photon counts per bin.
    bin_ns
        ``(n_bins,)`` bin centres in ns.
    fit
        Optional :class:`DoubleExpFit` result to overlay as a red curve. When
        provided, the bottom panel shows the Poisson-weighted residuals
        ``(hist - model)/√max(hist,1)`` restricted to the fit range.
    log_y
        Plot the decay on a log-y axis.
    title
        Optional title string (e.g. ``"t ∈ [12.5, 17.5] s"``).
    height
        Figure height in pixels.
    """
    histogram = np.asarray(histogram, dtype=np.float64)
    bin_ns = np.asarray(bin_ns, dtype=np.float64)

    have_residuals = fit is not None
    rows = 2 if have_residuals else 1
    row_heights = [0.75, 0.25] if have_residuals else [1.0]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Scatter(
            x=bin_ns,
            y=histogram,
            mode="lines+markers",
            line=dict(color="#444", width=1),
            marker=dict(size=4),
            name="data",
            hovertemplate="t=%{x:.2f}ns<br>N=%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=bin_ns,
                y=fit.model,
                mode="lines",
                line=dict(color="#d62728", width=2),
                name="double-exp fit",
                hovertemplate="t=%{x:.2f}ns<br>model=%{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # Shade the fit range on both panels
        fit_bins = bin_ns[fit.fit_mask]
        if fit_bins.size:
            x0 = float(fit_bins.min())
            x1 = float(fit_bins.max())
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor="#d62728", opacity=0.06, line_width=0,
                row="all", col=1,
            )

        # Bottom panel: weighted residuals on fit range only
        resid_x = bin_ns[fit.fit_mask]
        resid_y = fit.residuals_weighted
        fig.add_trace(
            go.Scatter(
                x=resid_x,
                y=resid_y,
                mode="markers",
                marker=dict(size=5, color="#d62728"),
                name="residuals",
                showlegend=False,
                hovertemplate="t=%{x:.2f}ns<br>r=%{y:.2f}σ<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="#888", row=2, col=1)
        fig.update_yaxes(title_text="(h−m)/√h", row=2, col=1)

    fig.update_xaxes(title_text="arrival time (ns)", row=rows, col=1)
    fig.update_yaxes(
        title_text="counts",
        type="log" if log_y else "linear",
        row=1,
        col=1,
    )

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=60, r=20, t=50 if title else 30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def fit_params_table(fit: DoubleExpFit) -> list[dict[str, object]]:
    """Return a list of ``{"param", "value"}`` rows for a Streamlit table."""
    return [
        {"param": "τ₁ (slow, ns)", "value": f"{fit.tau1:.3f}"},
        {"param": "τ₂ (fast, ns)", "value": f"{fit.tau2:.3f}"},
        {"param": "τ̄ amplitude-weighted (ns)", "value": f"{fit.tau_amp_weighted:.3f}"},
        {"param": "τ̄ intensity-weighted (ns)", "value": f"{fit.tau_int_weighted:.3f}"},
        {"param": "α₁ / α₂", "value": f"{fit.alpha1:.2e} / {fit.alpha2:.2e}"},
        {"param": "slow population fraction", "value": f"{fit.pop1_fraction*100:.1f}%"},
        {"param": "background", "value": f"{fit.background:.2e}"},
        {"param": "t₀ (ns) / σ_IRF (ns)", "value": f"{fit.t0:.3f} / {fit.sigma:.3f}"},
        {"param": "χ² reduced", "value": f"{fit.chi2_reduced:.2f}"},
        {"param": "photons in fit range", "value": f"{fit.n_photons:.2e}"},
        {"param": "fit status", "value": "ok" if fit.success else fit.message},
    ]
