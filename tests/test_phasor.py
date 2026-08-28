"""Tests for the PhasorPy wrapper in iflip.preprocess.phasor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iflip.io.tidy_csv import list_tidy_files, load_tidy, match_tidy_to_session
from iflip.preprocess.phasor import (
    apparent_lifetimes,
    phasor_for_known_taus,
    phasor_from_histogram,
    phasor_series_from_tcspc,
    semicircle_points,
)

# 126-bin axis matching the FLIPR TCSPC schema (0.0 – 12.5 ns, 0.1 ns step)
BIN_NS = np.round(np.arange(0.0, 12.6, 0.1), 2)

DATA_ROOT = Path(__file__).resolve().parents[2] / "FLIPR data"
EXAMPLE_BLOCK = "2026_04_09_acz02"


def _single_exp_histogram(tau: float, total_photons: float = 1e5) -> np.ndarray:
    """Build an ideal (IRF-free) single-exponential TCSPC histogram
    normalised to ``total_photons``. Starts at t=0 and uses bin *centres*."""
    t = BIN_NS + 0.05  # bin centres
    y = np.exp(-t / tau)
    return y * (total_photons / y.sum())


def test_phasor_single_exp_lands_near_semicircle():
    """A single-exponential decay should land very near the universal
    semicircle point expected for its τ."""
    tau = 3.0
    hist = _single_exp_histogram(tau)
    result = phasor_from_histogram(hist, BIN_NS, period_ns=12.5)

    # Check frequency metadata
    assert result.frequency_mhz == pytest.approx(80.0, rel=1e-6)

    # Known τ → expected semicircle position
    G_expected, S_expected = phasor_for_known_taus([tau])
    G_expected = np.atleast_1d(G_expected).ravel()
    S_expected = np.atleast_1d(S_expected).ravel()
    assert result.real == pytest.approx(float(G_expected[0]), abs=0.015)
    assert result.imag == pytest.approx(float(S_expected[0]), abs=0.015)

    # Inverted apparent lifetime recovers τ to ~1% on modulation, slightly
    # looser on phase (finite-support truncation biases phase more)
    assert result.tau_mod == pytest.approx(tau, rel=0.05)
    assert result.tau_phase == pytest.approx(tau, rel=0.08)


def test_phasor_lifetime_sweep_covers_semicircle():
    """Sweeping τ should trace the universal semicircle from (1,0) at τ→0
    to (0,0) at τ→∞."""
    taus = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0]
    Gs, Ss = [], []
    for tau in taus:
        r = phasor_from_histogram(_single_exp_histogram(tau), BIN_NS)
        Gs.append(r.real)
        Ss.append(r.imag)
    Gs = np.array(Gs)
    Ss = np.array(Ss)
    # G monotonically decreases with τ (shorter τ → closer to (1,0))
    assert np.all(np.diff(Gs) < 0)
    # All imag values are non-negative for physical decays
    assert np.all(Ss >= -0.02)
    # Every point is near the universal semicircle: (G-0.5)² + S² ≈ 0.25
    dist = (Gs - 0.5) ** 2 + Ss**2
    assert np.all(np.abs(dist - 0.25) < 0.04)


def test_phasor_from_known_taus_matches_phasor_from_histogram():
    """phasor_for_known_taus and phasor_from_histogram should land on the
    same semicircle point (within numerical tolerance)."""
    tau = 2.0
    Gh = phasor_from_histogram(_single_exp_histogram(tau), BIN_NS)
    Gk, Sk = phasor_for_known_taus([tau])
    Gk = np.atleast_1d(Gk).ravel()
    Sk = np.atleast_1d(Sk).ravel()
    assert Gh.real == pytest.approx(float(Gk[0]), abs=0.02)
    assert Gh.imag == pytest.approx(float(Sk[0]), abs=0.02)


def test_semicircle_points_lie_on_circle():
    G, S = semicircle_points(51)
    assert G.shape == (51,)
    # Universal semicircle: (G - 0.5)² + S² = 0.25
    dist = (G - 0.5) ** 2 + S**2
    np.testing.assert_allclose(dist, 0.25, atol=1e-10)
    assert np.all(S >= -1e-10)


def test_phasor_series_shapes_and_frequency():
    rng = np.random.default_rng(0)
    n_time = 50
    taus = np.linspace(0.5, 3.0, n_time)
    tcspc = np.stack([_single_exp_histogram(t) for t in taus])
    tcspc_noisy = rng.poisson(tcspc).astype(np.float64)

    real, imag, mean, freq = phasor_series_from_tcspc(tcspc_noisy, BIN_NS)
    assert real.shape == (n_time,)
    assert imag.shape == (n_time,)
    assert mean.shape == (n_time,)
    assert freq == pytest.approx(80.0, rel=1e-6)

    # Shorter τ → larger G (closer to (1, 0) on the semicircle). taus
    # increases from 0.5 to 3.0, so G should decrease across the series.
    assert real[0] > real[-1]
    corr = np.corrcoef(taus, real)[0, 1]
    assert corr < -0.9, f"G should decrease with τ, got corr={corr:.3f}"


def test_apparent_lifetimes_vectorised():
    """apparent_lifetimes should accept arrays and return arrays."""
    Gs, Ss = phasor_for_known_taus([0.5, 1.0, 2.0, 4.0])
    tau_phi, tau_mod = apparent_lifetimes(Gs, Ss, 80.0)
    np.testing.assert_allclose(tau_mod, [0.5, 1.0, 2.0, 4.0], rtol=1e-4)
    np.testing.assert_allclose(tau_phi, [0.5, 1.0, 2.0, 4.0], rtol=1e-4)


def test_empty_histogram_returns_nan():
    hist = np.zeros_like(BIN_NS)
    result = phasor_from_histogram(hist, BIN_NS)
    assert np.isnan(result.real)
    assert np.isnan(result.imag)
    assert result.n_photons == 0.0


# -----------------------------------------------------------------------------
# Real-data smoke
# -----------------------------------------------------------------------------
pytestmark_real = pytest.mark.skipif(
    not DATA_ROOT.is_dir(),
    reason=f"FLIPR data root not present at {DATA_ROOT}",
)


@pytestmark_real
def test_phasor_from_real_session_histogram():
    tidy = load_tidy(match_tidy_to_session(list_tidy_files(DATA_ROOT), EXAMPLE_BLOCK))
    hist = tidy.tcspc.sum(axis=0).astype(np.float64)
    result = phasor_from_histogram(hist, tidy.tcspc_bins_ns)

    # Physical sanity
    assert np.isfinite(result.real) and np.isfinite(result.imag)
    assert result.frequency_mhz == pytest.approx(80.0, rel=1e-6)
    # Apparent lifetime should be within a reasonable range for FLIM-DA0.5
    # (amp-weighted ~0.87 ns per instrument header; modulation/phase τ are
    # in a similar ballpark since the fast component dominates)
    assert 0.3 < result.tau_phase < 3.0, f"tau_phase out of range: {result.tau_phase}"
    assert 0.3 < result.tau_mod < 5.0, f"tau_mod out of range: {result.tau_mod}"


@pytestmark_real
def test_phasor_series_on_real_session_first_2000_frames():
    tidy = load_tidy(match_tidy_to_session(list_tidy_files(DATA_ROOT), EXAMPLE_BLOCK))
    # Take a subset to keep the test fast
    tcspc_sub = tidy.tcspc[:2000].astype(np.float64)
    real, imag, mean, freq = phasor_series_from_tcspc(tcspc_sub, tidy.tcspc_bins_ns)

    assert real.shape == (2000,)
    assert np.all(np.isfinite(real))
    assert np.all(np.isfinite(imag))
    assert freq == pytest.approx(80.0, rel=1e-6)
    # Phasor cloud should be compact (small session, stable baseline)
    assert real.std() < 0.1
    assert imag.std() < 0.1
