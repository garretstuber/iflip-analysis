"""Tests for the double-exponential TCSPC fitter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flipr.io.session_csv import load_session
from flipr.io.tidy_csv import list_tidy_files, load_tidy, match_tidy_to_session
from flipr.preprocess.lifetime import (
    DoubleExpFit,
    double_exp_model,
    fit_double_exp,
)

DATA_ROOT = Path(__file__).resolve().parents[2] / "FLIPR data"
EXAMPLE_BLOCK = "2026_04_09_acz02"

# Standard TCSPC axis: 126 bins @ 0.1 ns, 0.0 – 12.5 ns
BIN_NS = np.round(np.arange(0.0, 12.6, 0.1), 2)


def _synthetic_histogram(
    alpha1: float,
    tau1: float,
    alpha2: float,
    tau2: float,
    background: float,
    t0: float,
    sigma: float,
    total_photons: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a synthetic TCSPC histogram from the forward model.

    Scales the noiseless model so it integrates to ``total_photons`` (over
    positive values only) and optionally draws Poisson counts. Pass
    ``rng=None`` for a noiseless histogram.
    """
    model = double_exp_model(BIN_NS, alpha1, tau1, alpha2, tau2, background, t0, sigma)
    pos = np.clip(model, 0.0, None)
    if pos.sum() <= 0:
        raise ValueError("model is non-positive everywhere")
    scaled = pos * (total_photons / pos.sum())
    if rng is None:
        return scaled
    return rng.poisson(scaled).astype(np.float64)


def test_emg_model_is_non_negative_and_peaks_near_t0():
    """Forward model should be non-negative and peak near t0."""
    y = double_exp_model(
        BIN_NS, alpha1=1.0, tau1=3.5, alpha2=1.5, tau2=0.6, background=0.0, t0=1.0, sigma=0.21
    )
    assert np.all(y >= 0.0)
    peak_t = BIN_NS[int(np.argmax(y))]
    assert 0.8 < peak_t < 1.6  # allow some EMG skew


def test_single_component_limit_integrates_to_alpha_tau():
    """A single-component non-periodic EMG (α2 = 0) integrates to α·τ."""
    alpha, tau, t0, sigma = 1.0, 3.0, 1.0, 0.21
    # Fine grid well past the decay, and disable wrap-around so we're testing
    # the single-pulse analytical form against its known integral.
    t = np.arange(0.0, 50.0, 0.01)
    y = double_exp_model(t, alpha, tau, 0.0, 1.0, 0.0, t0, sigma, period_ns=0.0, n_wraps=0)
    integral = np.trapezoid(y, t)
    assert integral == pytest.approx(alpha * tau, rel=2e-3)


def test_noiseless_recovery_of_known_params():
    """With zero noise, the fit should recover the generating params to high precision."""
    true_params = dict(
        alpha1=100.0, tau1=3.8, alpha2=800.0, tau2=0.6,
        background=20.0, t0=1.012, sigma=0.21,
    )
    hist = _synthetic_histogram(**true_params, total_photons=5e5)
    fit = fit_double_exp(hist, BIN_NS)
    assert fit.success, fit.message
    # Lifetimes recoverable with tight tolerance
    assert fit.tau1 == pytest.approx(true_params["tau1"], rel=0.02)
    assert fit.tau2 == pytest.approx(true_params["tau2"], rel=0.02)
    # Amplitude ratio is the meaningful scale-free quantity
    true_ratio = true_params["alpha2"] / true_params["alpha1"]
    fit_ratio = fit.alpha2 / fit.alpha1
    assert fit_ratio == pytest.approx(true_ratio, rel=0.05)


def test_poisson_noise_recovery():
    """With realistic Poisson noise, recovery is ~5%."""
    rng = np.random.default_rng(seed=42)
    true_params = dict(
        alpha1=100.0, tau1=3.8, alpha2=800.0, tau2=0.6,
        background=5.0, t0=1.012, sigma=0.21,
    )
    hist = _synthetic_histogram(**true_params, total_photons=2e5, rng=rng)
    fit = fit_double_exp(hist, BIN_NS)
    assert fit.success
    assert fit.tau1 == pytest.approx(true_params["tau1"], rel=0.06)
    assert fit.tau2 == pytest.approx(true_params["tau2"], rel=0.08)
    assert fit.chi2_reduced < 3.0  # sanity: fit is reasonable


def test_canonical_tau_ordering():
    """After the fit, tau1 is always the slower component regardless of
    how the generator labelled the two decays."""
    rng = np.random.default_rng(seed=7)
    # Generator labels slow as alpha2/tau2 — fitter must canonicalise.
    hist = _synthetic_histogram(
        alpha1=500.0, tau1=0.5, alpha2=50.0, tau2=3.5,
        background=10.0, t0=1.0, sigma=0.21, total_photons=2e5, rng=rng,
    )
    fit = fit_double_exp(hist, BIN_NS)
    assert fit.success, fit.message
    assert fit.tau1 >= fit.tau2
    assert fit.tau1 == pytest.approx(3.5, rel=0.10)
    assert fit.tau2 == pytest.approx(0.5, rel=0.15)


def test_derived_mean_lifetimes_match_manual():
    """Amplitude- and intensity-weighted mean lifetime match hand calculation."""
    fit = DoubleExpFit(
        alpha1=0.141, tau1=3.8114, alpha2=1.5836, tau2=0.60638,
        background=0.0, t0=1.012, sigma=0.21, chi2_reduced=1.0,
        residuals=np.zeros(5), residuals_weighted=np.zeros(5),
        model=np.zeros(5), fit_mask=np.ones(5, dtype=bool), n_photons=1.0,
        success=True, message="",
    )
    # Matches header_state_avgtau = 0.86871 from the example param file
    assert fit.tau_amp_weighted == pytest.approx(0.8687, abs=0.001)
    # Pop fraction of slow (slow is ~8.2% per header)
    assert fit.pop1_fraction == pytest.approx(0.0819, abs=0.002)


def test_fit_range_excludes_outside_bins():
    """Bins outside [fit_start, fit_stop] must not influence chi²."""
    rng = np.random.default_rng(seed=1)
    hist = _synthetic_histogram(
        alpha1=100.0, tau1=3.8, alpha2=800.0, tau2=0.6,
        background=5.0, t0=1.012, sigma=0.21, total_photons=2e5, rng=rng,
    )
    # Poison the first bin with a huge spike — fit should still succeed
    hist[0] += 50_000
    fit = fit_double_exp(hist, BIN_NS, fit_start_ns=0.4, fit_stop_ns=12.3)
    assert fit.success
    # Bin at 0.0 ns is outside the fit mask
    assert not fit.fit_mask[0]
    # Recovery is still good because the spike is excluded
    assert fit.tau1 == pytest.approx(3.8, rel=0.06)


# -----------------------------------------------------------------------------
# Real-data fit: compare against instrument-reported fit in the header
# -----------------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    not DATA_ROOT.is_dir(),
    reason=f"FLIPR data root not present at {DATA_ROOT}",
)


@pytestmark_real
def test_fit_real_session_summed_histogram():
    """Fit the session-wide summed TCSPC histogram and sanity-check the result.

    Note: the instrument software fits with a measured (non-Gaussian) IRF and
    an afterpulse-aware background, whereas this fitter uses a Gaussian IRF
    plus constant background. The two models agree tightly on the
    amplitude-weighted mean lifetime (a well-constrained first moment) but
    can redistribute the individual τ1 and τ2 components, so we anchor the
    test on ``tau_amp_weighted`` and physical ranges rather than on exact
    header match.
    """
    tidy_files = list_tidy_files(DATA_ROOT)
    picked = match_tidy_to_session(tidy_files, EXAMPLE_BLOCK)
    assert picked is not None
    tidy = load_tidy(picked)

    session = load_session(DATA_ROOT / "sessions" / EXAMPLE_BLOCK)

    total = tidy.tcspc.sum(axis=0).astype(np.float64)
    fit = fit_double_exp(
        total,
        tidy.tcspc_bins_ns,
        irf_sigma_ns=float(session.meta.get("IRF", "0.212")),
        t0_ns=float(session.meta.get("t0", "1.012")),
        fit_start_ns=float(tidy.params.get("header_state_spcrangelow", "0.4")),
        fit_stop_ns=float(tidy.params.get("header_state_spcrangehigh", "12.3")),
    )
    assert fit.success, fit.message

    header_avg = float(tidy.params["header_state_avgtau"])

    # Well-constrained: amp-weighted mean lifetime should match header within 10%
    assert fit.tau_amp_weighted == pytest.approx(header_avg, rel=0.10), (
        f"avgtau mismatch: fit={fit.tau_amp_weighted:.3f}, header={header_avg:.3f}"
    )

    # Canonical ordering and physical ranges for FLIM-DA0.5
    assert fit.tau1 >= fit.tau2
    assert 1.0 < fit.tau1 < 6.0, f"tau1 out of physical range: {fit.tau1}"
    assert 0.2 < fit.tau2 < 1.5, f"tau2 out of physical range: {fit.tau2}"

    # Fast component should dominate (pop2 ~92% per header)
    assert fit.alpha2 > fit.alpha1, "expected fast component to dominate the amplitude"
    assert fit.alpha2 / fit.alpha1 > 3.0

    # Fit actually converged (finite chi²) and consumed most of the photons
    assert np.isfinite(fit.chi2_reduced)
    assert fit.n_photons > 1e9  # full session should have >>1e9 photons
