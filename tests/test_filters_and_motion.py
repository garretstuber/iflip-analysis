"""Tests for the filters and motion-QC modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flipr.io.session_csv import SessionData, load_session
from flipr.preprocess.filters import apply_filter, filtered_session
from flipr.preprocess.motion import (
    compute_qc,
    detect_intensity_jumps,
    detect_motion_correlation,
    detect_photon_starvation,
    rolling_corr,
)

DATA_ROOT = Path(__file__).resolve().parents[2] / "FLIPR data"
EXAMPLE_BLOCK = "2026_04_09_acz02"


# -----------------------------------------------------------------------------
# Filter tests
# -----------------------------------------------------------------------------
def test_apply_filter_none_is_passthrough():
    x = np.random.default_rng(0).normal(size=200)
    y = apply_filter(x, mode="none", fs=20.0, window_s=1.0)
    np.testing.assert_allclose(y, x)


def test_boxcar_reduces_noise_std():
    rng = np.random.default_rng(1)
    signal = np.sin(np.linspace(0, 10, 200))
    noisy = signal + rng.normal(0, 0.2, 200)
    smoothed = apply_filter(noisy, mode="boxcar", fs=20.0, window_s=0.5)
    # Smoothed residual should be smaller than the raw residual
    assert np.std(smoothed - signal) < np.std(noisy - signal)
    assert smoothed.shape == noisy.shape


def test_savgol_preserves_low_frequency_peak_height():
    t = np.linspace(0, 10, 200)
    # Pure smooth signal (no noise) — savgol of appropriate order should
    # reproduce it almost exactly
    signal = np.exp(-((t - 5) ** 2) / 0.5)
    smoothed = apply_filter(signal, mode="savgol", fs=20.0, window_s=0.3, polyorder=3)
    # Peak heights should match within a small tolerance
    assert abs(smoothed.max() - signal.max()) < 0.02


def test_median_filter_removes_spikes():
    x = np.full(200, 2.0)
    x[50] = 100.0  # single spike
    x[100] = -50.0
    smoothed = apply_filter(x, mode="median", fs=20.0, window_s=0.25)
    # Spikes should be gone
    assert smoothed[50] == pytest.approx(2.0, abs=0.01)
    assert smoothed[100] == pytest.approx(2.0, abs=0.01)


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="unknown filter mode"):
        apply_filter(np.zeros(10), mode="bogus", fs=20.0)  # type: ignore[arg-type]


def _fake_session(
    intensity: np.ndarray,
    lifetime: np.ndarray,
    fs: float = 20.0,
) -> SessionData:
    n = len(intensity)
    t = (np.arange(n) + 0.5) / fs
    streams = pd.DataFrame(
        {
            "time": t,
            "intensity": intensity,
            "lifetime": lifetime,
            "marks": np.zeros(n, dtype=int),
        }
    )
    meta = {"fs": str(fs), "subject": "fake", "blockname": "fake"}
    return SessionData(
        blockname="fake",
        path=Path("/tmp/fake"),
        meta=meta,
        events=pd.DataFrame({"event_id_char": [], "time": []}),
        streams=streams,
        peth=pd.DataFrame(),
    )


def test_filtered_session_returns_new_streams():
    rng = np.random.default_rng(2)
    n = 400
    intensity = 1000 + rng.normal(0, 50, n)
    lifetime = 2.7 + rng.normal(0, 0.05, n)
    session = _fake_session(intensity, lifetime)
    out = filtered_session(session, mode="boxcar", window_s=0.5)

    # Different object, same length
    assert out is not session
    assert len(out.streams) == len(session.streams)
    # Streams are filtered: std < raw std
    assert out.streams["intensity"].std() < session.streams["intensity"].std()
    assert out.streams["lifetime"].std() < session.streams["lifetime"].std()
    # Raw session unchanged
    np.testing.assert_allclose(session.streams["intensity"].to_numpy(), intensity)


def test_filtered_session_none_is_identity():
    session = _fake_session(np.arange(100.0), np.arange(100.0) * 0.01)
    out = filtered_session(session, mode="none")
    # Short-circuit: should return the same object
    assert out is session


# -----------------------------------------------------------------------------
# Motion QC tests
# -----------------------------------------------------------------------------
def test_rolling_corr_detects_anticorrelation():
    rng = np.random.default_rng(3)
    n = 500
    signal = rng.normal(0, 1, n)
    x = signal
    y = -signal + rng.normal(0, 0.01, n)
    r = rolling_corr(x, y, window=21)
    # The rolling correlation should be near -1 in the middle
    assert np.nanmean(r[50:-50]) < -0.95


def test_rolling_corr_edges_are_nan():
    x = np.arange(100, dtype=float)
    y = x**2
    r = rolling_corr(x, y, window=11)
    assert np.isnan(r[0])
    assert np.isnan(r[-1])


def test_rolling_corr_rejects_even_window():
    with pytest.raises(ValueError, match="odd"):
        rolling_corr(np.zeros(10), np.zeros(10), window=10)


def test_detect_photon_starvation_absolute_threshold():
    intensity = np.array([1000, 1100, 950, 50, 40, 1050, 980])
    mask = detect_photon_starvation(intensity, min_photons=200)
    np.testing.assert_array_equal(mask, [False, False, False, True, True, False, False])


def test_detect_photon_starvation_mad_threshold_catches_real_drops():
    """MAD-based auto-threshold should catch genuine drops but not flag
    nominally stable data."""
    rng = np.random.default_rng(5)
    n = 1000
    # Stable population around 1000 with small noise, plus 3 genuine drops
    intensity = 1000 + rng.normal(0, 10, n)
    intensity[500] = 200.0
    intensity[700] = 150.0
    intensity[800] = 50.0
    mask = detect_photon_starvation(intensity, min_photons=None, mad_k=5.0)
    # The three injected drops should be flagged
    assert mask[500] and mask[700] and mask[800]
    # And nothing else
    assert mask.sum() == 3


def test_detect_photon_starvation_mad_threshold_stable_session_zero_flags():
    """A stable session with only Gaussian noise should have ~0 flags."""
    rng = np.random.default_rng(6)
    intensity = 1000 + rng.normal(0, 20, 10_000)
    mask = detect_photon_starvation(intensity, min_photons=None, mad_k=5.0)
    # Expected false-positive rate at 5σ is ~3e-7 — essentially zero.
    assert mask.sum() <= 2


def test_detect_intensity_jumps_flags_spike():
    x = np.full(100, 1000.0)
    x[50] = 5000.0  # huge positive jump
    flags = detect_intensity_jumps(x, z_threshold=6.0)
    assert flags[50]
    assert flags.sum() < 5  # no spurious flags elsewhere


def test_detect_motion_correlation_on_clean_data():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)  # uncorrelated with x
    flag, r = detect_motion_correlation(x, y, window=21, corr_threshold=0.7)
    # Very few flags expected for uncorrelated data
    assert flag.sum() / flag.size < 0.05


# -----------------------------------------------------------------------------
# Real data integration
# -----------------------------------------------------------------------------
pytestmark_real = pytest.mark.skipif(
    not DATA_ROOT.is_dir(),
    reason=f"FLIPR data root not present at {DATA_ROOT}",
)


@pytestmark_real
def test_filtered_session_on_real_data_preserves_shape():
    session = load_session(DATA_ROOT / "sessions" / EXAMPLE_BLOCK)
    filt = filtered_session(session, mode="savgol", window_s=0.3, polyorder=2)
    assert len(filt.streams) == len(session.streams)
    assert filt.streams["lifetime"].std() < session.streams["lifetime"].std()


@pytestmark_real
def test_compute_qc_on_real_head_fixed_session_is_near_zero():
    """The example session is a head-fixed recording with stable fibre
    coupling. With the new defaults (MAD-based photon threshold, no
    correlation in any_flag) the flagged rate must be ~0%."""
    session = load_session(DATA_ROOT / "sessions" / EXAMPLE_BLOCK)
    qc = compute_qc(session)
    assert qc.time.shape == (len(session.streams),)
    summary = qc.summary()
    # Clean head-fixed recording: both starvation and jumps should be
    # ~0%, so "any" should be well under 0.5%
    assert summary["any"] < 0.005, (
        f"too many QC flags on a clean head-fixed session: {summary}"
    )
    assert summary["intensity"] < 0.005
    assert summary["jump"] < 0.005


@pytestmark_real
def test_compute_qc_correlation_is_diagnostic_not_in_any_flag():
    """The rolling correlation trace and its threshold-highlighting must
    not affect any_flag."""
    session = load_session(DATA_ROOT / "sessions" / EXAMPLE_BLOCK)
    # Even with a very permissive correlation threshold, any_flag should
    # not grow — correlation is purely diagnostic.
    strict = compute_qc(session, motion_corr_threshold=0.99)
    permissive = compute_qc(session, motion_corr_threshold=0.3)
    assert strict.any_flag.sum() == permissive.any_flag.sum()
    # But the diagnostic corr_above_threshold SHOULD grow with lower threshold
    assert permissive.corr_above_threshold.sum() > strict.corr_above_threshold.sum() * 3
