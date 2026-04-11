"""Smoke tests for the plotly figure constructors.

We don't assert on visual output — just that each function returns a
``plotly.graph_objects.Figure`` without raising against the example session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from flipr.io.session_csv import load_session
from flipr.io.tidy_csv import list_tidy_files, load_tidy, match_tidy_to_session
from flipr.preprocess.lifetime import fit_double_exp
from flipr.preprocess.phasor import phasor_from_histogram, phasor_series_from_tcspc
from flipr.viz.phasor import phasor_plot_figure
from flipr.viz.tcspc import fit_params_table, tcspc_decay_figure
from flipr.viz.traces import event_rate_histogram, session_traces_figure

DATA_ROOT = Path(__file__).resolve().parents[2] / "FLIPR data"
EXAMPLE_BLOCK = "2026_04_09_acz02"

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.is_dir(),
    reason=f"FLIPR data root not present at {DATA_ROOT}",
)


def test_session_traces_figure_builds():
    session = load_session(DATA_ROOT / "sessions" / EXAMPLE_BLOCK)
    fig = session_traces_figure(session, max_points=5000)
    assert isinstance(fig, go.Figure)
    # intensity + lifetime + three event types
    assert len(fig.data) >= 2


def test_session_traces_with_highlight_and_log():
    session = load_session(DATA_ROOT / "sessions" / EXAMPLE_BLOCK)
    fig = session_traces_figure(
        session, max_points=2000, log_intensity=True, highlight_range=(100.0, 150.0)
    )
    assert isinstance(fig, go.Figure)


def test_event_rate_histogram_non_empty():
    session = load_session(DATA_ROOT / "sessions" / EXAMPLE_BLOCK)
    df = event_rate_histogram(session, "lick", bin_width=10.0)
    assert not df.empty
    assert {"bin_start", "count"}.issubset(df.columns)
    assert df["count"].sum() > 0


def test_tcspc_decay_figure_with_fit_builds():
    tidy = load_tidy(match_tidy_to_session(list_tidy_files(DATA_ROOT), EXAMPLE_BLOCK))
    hist = tidy.integrate_histogram(100.0, 120.0).astype(np.float64)
    fit = fit_double_exp(hist, tidy.tcspc_bins_ns)
    fig = tcspc_decay_figure(hist, tidy.tcspc_bins_ns, fit=fit, title="test window")
    assert isinstance(fig, go.Figure)
    # data + fit model + residuals
    assert len(fig.data) >= 3


def test_tcspc_decay_figure_no_fit_builds():
    tidy = load_tidy(match_tidy_to_session(list_tidy_files(DATA_ROOT), EXAMPLE_BLOCK))
    hist = tidy.integrate_histogram(0.0, 10.0).astype(np.float64)
    fig = tcspc_decay_figure(hist, tidy.tcspc_bins_ns, fit=None)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_fit_params_table_has_expected_rows():
    tidy = load_tidy(match_tidy_to_session(list_tidy_files(DATA_ROOT), EXAMPLE_BLOCK))
    hist = tidy.integrate_histogram(100.0, 110.0).astype(np.float64)
    fit = fit_double_exp(hist, tidy.tcspc_bins_ns)
    rows = fit_params_table(fit)
    params = {r["param"] for r in rows}
    assert "τ₁ (slow, ns)" in params
    assert "χ² reduced" in params


def test_phasor_plot_figure_builds_empty():
    fig = phasor_plot_figure()
    assert isinstance(fig, go.Figure)
    # at least the semicircle
    assert len(fig.data) >= 1


def test_phasor_plot_figure_with_cloud_and_references():
    tidy = load_tidy(match_tidy_to_session(list_tidy_files(DATA_ROOT), EXAMPLE_BLOCK))
    tcspc_sub = tidy.tcspc[:500].astype(np.float64)
    real, imag, _, _ = phasor_series_from_tcspc(tcspc_sub, tidy.tcspc_bins_ns)
    finite = np.isfinite(real) & np.isfinite(imag)
    fig = phasor_plot_figure(
        real=real[finite],
        imag=imag[finite],
        reference_taus=[3.8, 0.6],
        highlight=(float(real[finite].mean()), float(imag[finite].mean())),
    )
    assert isinstance(fig, go.Figure)
    # semicircle + cloud + reference markers + highlight = 4 traces
    assert len(fig.data) >= 4
