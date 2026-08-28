"""Tests for background subtraction and sliding-window τ."""

from __future__ import annotations

import numpy as np
import pytest

from iflip.preprocess.background import (
    BackgroundEstimate,
    compute_background,
    subtract_background,
    subtract_background_per_frame,
)
from iflip.preprocess.sliding_tau import SLIDING_METHODS, sliding_tau

# --------------------------------------------------------------------------- #
# Synthetic data helpers
# --------------------------------------------------------------------------- #

N_BINS = 126
BINS_NS = np.linspace(0.0, 12.5, N_BINS, endpoint=False) + 0.05  # 0.1 ns bins, centred


def _synth_decay(tau: float, amp: float, t0: float = 1.0, sigma: float = 0.2,
                 bg: float = 0.0, n_bins: int = N_BINS) -> np.ndarray:
    """Synthetic single-exp + Gaussian IRF + flat bg histogram (per-frame)."""
    t = BINS_NS
    # Simple convolution-free approximation: shift + decay
    rising = np.exp(-(t - t0) ** 2 / (2 * sigma * sigma))
    decay = np.where(t > t0, np.exp(-(t - t0) / tau), 0.0)
    # Use the EMG shape from rising * decay with smooth merge
    shape = rising * decay + 0.5 * decay
    shape = shape / shape.max()
    return amp * shape + bg


def _synth_session_tcspc(n_frames: int, tau_fn, photons_per_frame: float = 5000.0,
                        bg_per_frame: float = 5.0) -> np.ndarray:
    """A (n_frames, n_bins) matrix where each frame has lifetime tau_fn(frame_idx)."""
    out = np.zeros((n_frames, N_BINS), dtype=np.float64)
    for i in range(n_frames):
        tau = float(tau_fn(i))
        shape = _synth_decay(tau, amp=1.0, bg=0.0)
        shape = shape / shape.sum() * photons_per_frame
        out[i] = shape + bg_per_frame
    return out


# --------------------------------------------------------------------------- #
# compute_background / subtract_background
# --------------------------------------------------------------------------- #


def test_compute_background_full_session():
    rng = np.random.default_rng(0)
    n_frames = 100
    bg_shape = np.linspace(2.0, 5.0, N_BINS)
    tcspc = rng.poisson(lam=bg_shape, size=(n_frames, N_BINS)).astype(np.float64)
    bg = compute_background(tcspc, BINS_NS, source_label="synthetic")
    assert bg.n_frames == n_frames
    assert bg.per_frame.shape == (N_BINS,)
    # Should recover bg_shape to within Poisson tolerance.
    # 100 frames at λ ≈ 3 → per-bin SE ≈ √3/√100 ≈ 0.17, so 4σ ≈ 0.7
    np.testing.assert_allclose(bg.per_frame, bg_shape, atol=0.8)


def test_compute_background_time_window():
    n_frames = 100
    fs = 20.0
    times = (np.arange(n_frames) + 0.5) / fs  # 0–5 s
    tcspc = np.full((n_frames, N_BINS), 3.0)
    # Different bg in the second half
    tcspc[50:] = 10.0
    bg = compute_background(
        tcspc, BINS_NS, time_index=times, time_window=(0.0, 2.5)
    )
    assert bg.n_frames == 50  # first half only
    np.testing.assert_allclose(bg.per_frame, 3.0)


def test_compute_background_empty_window_raises():
    tcspc = np.zeros((50, N_BINS))
    with pytest.raises(ValueError, match="no frames"):
        compute_background(tcspc, BINS_NS, time_index=np.arange(50) * 0.05,
                          time_window=(100.0, 200.0))


def test_compute_background_wrong_shape_raises():
    with pytest.raises(ValueError, match="2D"):
        compute_background(np.zeros(10), BINS_NS)
    with pytest.raises(ValueError, match="bin axis"):
        compute_background(np.zeros((5, N_BINS - 1)), BINS_NS)


def test_subtract_background_window():
    bg = BackgroundEstimate(
        per_frame=np.full(N_BINS, 2.0),
        bins_ns=BINS_NS,
        n_frames=100,
        source_label="test",
    )
    hist = np.full(N_BINS, 50.0)
    out = subtract_background(hist, bg, n_frames=10, scale=1.0)
    # Each bin: 50 - 1.0 * 10 * 2.0 = 30.0
    np.testing.assert_allclose(out, 30.0)


def test_subtract_background_clips_negative():
    bg = BackgroundEstimate(
        per_frame=np.full(N_BINS, 100.0),
        bins_ns=BINS_NS,
        n_frames=1,
        source_label="test",
    )
    hist = np.full(N_BINS, 5.0)
    out = subtract_background(hist, bg, n_frames=1, scale=1.0, clip_negative=True)
    np.testing.assert_array_equal(out, 0.0)
    out_unclip = subtract_background(hist, bg, n_frames=1, scale=1.0, clip_negative=False)
    np.testing.assert_allclose(out_unclip, -95.0)


def test_subtract_background_per_frame_shape():
    bg = BackgroundEstimate(
        per_frame=np.full(N_BINS, 1.0),
        bins_ns=BINS_NS,
        n_frames=1,
        source_label="test",
    )
    tcspc = np.full((20, N_BINS), 10.0)
    out = subtract_background_per_frame(tcspc, bg, scale=1.0)
    assert out.shape == tcspc.shape
    np.testing.assert_allclose(out, 9.0)


def test_subtract_background_bin_mismatch():
    bg = BackgroundEstimate(
        per_frame=np.zeros(50),
        bins_ns=np.linspace(0, 5, 50),
        n_frames=1,
        source_label="t",
    )
    with pytest.raises(ValueError, match="mismatch"):
        subtract_background(np.zeros(N_BINS), bg)


# --------------------------------------------------------------------------- #
# sliding_tau — phasor methods
# --------------------------------------------------------------------------- #


def test_sliding_tau_phasor_constant_signal():
    """A flat session should give an approximately flat τ trace."""
    n_frames = 200
    fs = 20.0
    tcspc = _synth_session_tcspc(n_frames, tau_fn=lambda i: 1.5)

    res = sliding_tau(
        tcspc, BINS_NS, fs,
        method="phasor_phase",
        window_s=1.0, step_s=0.5,
    )
    assert res.method == "phasor_phase"
    assert res.window_s == 1.0
    # All windows should give ~similar phasor τ
    finite = np.isfinite(res.tau)
    assert finite.sum() > 5
    assert np.std(res.tau[finite]) < 0.1  # tight cluster


def test_sliding_tau_picks_up_step_change():
    """A step change in lifetime mid-session should show up in the trace."""
    n_frames = 400
    fs = 20.0

    def tau_fn(i):
        return 1.0 if i < n_frames // 2 else 2.5

    tcspc = _synth_session_tcspc(n_frames, tau_fn=tau_fn)
    res = sliding_tau(
        tcspc, BINS_NS, fs,
        method="phasor_phase",
        window_s=1.0, step_s=1.0,
    )
    # Mean τ in first half should be << mean τ in second half
    half_t = (n_frames / fs) / 2
    early = res.tau[res.time < half_t - 1.0]
    late = res.tau[res.time > half_t + 1.0]
    early = early[np.isfinite(early)]
    late = late[np.isfinite(late)]
    assert early.size > 0 and late.size > 0
    assert late.mean() > early.mean() + 0.3


def test_sliding_tau_n_steps_matches_window_layout():
    fs = 20.0
    n_frames = 200
    tcspc = np.zeros((n_frames, N_BINS))
    res = sliding_tau(
        tcspc, BINS_NS, fs,
        method="phasor_phase",
        window_s=1.0, step_s=0.5,
    )
    # frames per window = 20, step = 10 → starts at 0, 10, ..., 180 = 19 steps
    expected = (n_frames - 20) // 10 + 1
    assert res.n_steps == expected


def test_sliding_tau_window_too_long():
    tcspc = np.zeros((10, N_BINS))
    with pytest.raises(ValueError, match="longer than session"):
        sliding_tau(tcspc, BINS_NS, fs=20.0, window_s=10.0, step_s=1.0)


def test_sliding_tau_invalid_method_raises():
    tcspc = np.full((100, N_BINS), 10.0)
    with pytest.raises(ValueError, match="unknown method"):
        sliding_tau(tcspc, BINS_NS, fs=20.0, method="bogus", window_s=1.0, step_s=0.5)  # type: ignore[arg-type]


def test_sliding_tau_with_background():
    """Background subtraction should change the resulting τ trace."""
    n_frames = 100
    fs = 20.0
    # Use a bg shape that is NOT flat — flat offsets affect the phasor
    # via normalization but a sloped bg shape changes the phasor more
    # dramatically and is more representative of real data.
    bg_shape = np.linspace(50.0, 5.0, N_BINS)
    tcspc = _synth_session_tcspc(n_frames, tau_fn=lambda i: 2.0, bg_per_frame=0.0)
    tcspc = tcspc + bg_shape[None, :]

    bg = BackgroundEstimate(
        per_frame=bg_shape,
        bins_ns=BINS_NS,
        n_frames=1000,
        source_label="bg",
    )

    res_no_bg = sliding_tau(
        tcspc, BINS_NS, fs,
        method="phasor_phase",
        window_s=1.0, step_s=0.5,
    )
    res_bg = sliding_tau(
        tcspc, BINS_NS, fs,
        method="phasor_phase",
        window_s=1.0, step_s=0.5,
        background=bg,
    )
    finite_no = res_no_bg.tau[np.isfinite(res_no_bg.tau)]
    finite_bg = res_bg.tau[np.isfinite(res_bg.tau)]
    assert finite_no.size > 0 and finite_bg.size > 0
    # bg subtraction should move the τ readout substantially
    assert abs(finite_no.mean() - finite_bg.mean()) > 0.05


def test_sliding_methods_all_listed():
    assert "phasor_phase" in SLIDING_METHODS
    assert "phasor_mod" in SLIDING_METHODS
    assert "fit_double" in SLIDING_METHODS
    assert "fit_single" in SLIDING_METHODS
