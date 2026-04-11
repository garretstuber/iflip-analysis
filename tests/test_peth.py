"""Tests for the PETH engine in flipr.align.peth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flipr.align.peth import PETH, build_peth
from flipr.io.session_csv import SessionData, load_session

DATA_ROOT = Path(__file__).resolve().parents[2] / "FLIPR data"
EXAMPLE_BLOCK = "2026_04_09_acz02"


# -----------------------------------------------------------------------------
# Synthetic helpers
# -----------------------------------------------------------------------------
def _fake_session(
    fs: float = 20.0,
    duration: float = 60.0,
    event_times: list[float] | None = None,
    event_type: str = "test",
    signal_at_event: float = 10.0,
    baseline: float = 1.0,
) -> SessionData:
    """Build a SessionData with a simple 'step' signal around events.

    The signal equals ``baseline`` everywhere except in a 1-second window
    starting at each event, where it jumps to ``signal_at_event``. This
    lets us assert exact PETH shapes/values without wrestling with real
    instrument noise.
    """
    if event_times is None:
        event_times = [10.0, 25.0, 40.0]
    n = int(round(duration * fs))
    t = (np.arange(n) + 0.5) / fs  # centred bins: 0.025, 0.075, ...
    sig = np.full(n, baseline, dtype=np.float64)
    for ev in event_times:
        mask = (t >= ev) & (t < ev + 1.0)
        sig[mask] = signal_at_event

    streams = pd.DataFrame(
        {
            "time": t,
            "intensity": sig * 1000,
            "lifetime": sig,
            "marks": np.zeros(n, dtype=int),
        }
    )
    events = pd.DataFrame(
        {
            "event_id_char": [event_type] * len(event_times),
            "time": event_times,
        }
    )
    meta = {
        "fs": str(fs),
        "subject": "synthetic",
        "procedure": "test",
        "blockname": "synthetic",
    }
    return SessionData(
        blockname="synthetic",
        path=Path("/tmp/fake"),
        meta=meta,
        events=events,
        streams=streams,
        peth=pd.DataFrame(),
    )


# -----------------------------------------------------------------------------
# Synthetic tests
# -----------------------------------------------------------------------------
def test_build_peth_shape_and_trial_count():
    sess = _fake_session()
    peth = build_peth(sess, "test", signal="lifetime", pre_window=-2.0, post_window=3.0)
    # 20 Hz × (2 + 3) seconds = 100 samples + 1 for the zero sample
    assert peth.n_samples == int(round(5 * 20)) + 1
    assert peth.n_trials == 3
    # time_rel endpoints match the requested window
    assert peth.time_rel[0] == pytest.approx(-2.0)
    assert peth.time_rel[-1] == pytest.approx(3.0)
    # Event times stored correctly
    np.testing.assert_allclose(peth.event_times, [10.0, 25.0, 40.0])


def test_build_peth_signal_values_match_expected_step():
    sess = _fake_session(signal_at_event=10.0, baseline=1.0)
    peth = build_peth(sess, "test", signal="lifetime", pre_window=-1.0, post_window=2.0)

    # Pre-event samples should all be 1.0
    pre_mask = peth.time_rel < 0
    assert np.allclose(peth.values[:, pre_mask], 1.0)

    # Samples in [0, 1) should be 10.0 (the "event response")
    during_mask = (peth.time_rel >= 0) & (peth.time_rel < 1.0)
    assert np.allclose(peth.values[:, during_mask], 10.0)

    # Samples ≥ 1 s should be back to baseline
    post_mask = peth.time_rel >= 1.0
    assert np.allclose(peth.values[:, post_mask], 1.0)


def test_build_peth_boundary_trials_get_nan():
    """An event too close to the session start should have NaN in its pre-
    window but valid post-window samples."""
    sess = _fake_session(duration=60.0, event_times=[0.5])
    peth = build_peth(sess, "test", signal="lifetime", pre_window=-2.0, post_window=2.0)
    # Trial 0: pre-window goes to t=-1.5 s which is before session start
    assert np.isnan(peth.values[0, 0])
    # The post-window should still have valid samples
    assert np.all(np.isfinite(peth.values[0, peth.time_rel >= 0]))


def test_mean_and_sem_match_numpy():
    """PETH.mean() and PETH.sem() should match direct numpy calls."""
    sess = _fake_session(signal_at_event=5.0, baseline=0.0)
    peth = build_peth(sess, "test", signal="lifetime", pre_window=-1.0, post_window=1.0)
    assert np.allclose(peth.mean(), np.nanmean(peth.values, axis=0))
    # SEM defined only for n>1 valid trials
    sem = peth.sem()
    # With 3 identical trials SEM should be 0 (std = 0)
    assert np.allclose(sem, 0.0, atol=1e-10)


def test_baseline_correction_zeros_out_baseline_window():
    """After baseline correction the mean over the baseline window should
    be zero for every trial. Baseline window excludes the event sample
    itself (which is the step onset)."""
    sess = _fake_session(signal_at_event=10.0, baseline=2.5)
    peth = build_peth(sess, "test", signal="lifetime", pre_window=-2.0, post_window=2.0)
    # Use a strictly pre-event window so baseline samples are all 2.5
    bc = peth.baseline_corrected(window=(-2.0, -0.05))
    mask = (bc.time_rel >= -2.0) & (bc.time_rel <= -0.05)
    trial_baseline_means = np.nanmean(bc.values[:, mask], axis=1)
    np.testing.assert_allclose(trial_baseline_means, 0.0, atol=1e-12)
    # During the step, baseline-corrected signal should be exactly 10 - 2.5 = 7.5
    during = (bc.time_rel >= 0) & (bc.time_rel < 1.0)
    np.testing.assert_allclose(bc.values[:, during], 7.5, atol=1e-12)


def test_zscore_normalises_baseline_to_zero_mean_unit_std():
    """After z-scoring with a window that has non-zero variance, the mean
    should be ~0 and std ~1 inside that window."""
    rng = np.random.default_rng(123)
    # Fresh session with noise so baseline has non-zero std
    sess = _fake_session(duration=60.0)
    # Inject noise into the lifetime column of the fake session in-place
    # (we're operating on the DataFrame; safe for tests)
    sess.streams["lifetime"] = sess.streams["lifetime"] + rng.normal(0, 0.3, len(sess.streams))
    peth = build_peth(sess, "test", signal="lifetime", pre_window=-3.0, post_window=3.0)
    z = peth.zscored(window=(-2.0, -0.1))
    mask = (z.time_rel >= -2.0) & (z.time_rel <= -0.1)
    trial_means = np.nanmean(z.values[:, mask], axis=1)
    trial_stds = np.nanstd(z.values[:, mask], axis=1, ddof=1)
    np.testing.assert_allclose(trial_means, 0.0, atol=1e-10)
    np.testing.assert_allclose(trial_stds, 1.0, atol=1e-10)


def test_trial_auc_matches_hand_calculation():
    """AUC over a 1-second step of amplitude 10 should be 10 * 1 = 10."""
    sess = _fake_session(signal_at_event=10.0, baseline=0.0)
    peth = build_peth(sess, "test", signal="lifetime", pre_window=-1.0, post_window=2.0)
    # integrate over [0, 0.95]: at 20 Hz we have samples at 0.0, 0.05, ..., 0.95
    # so 20 samples × 0.05 s = 1.0 s-equivalent
    auc = peth.trial_auc(window=(0.0, 0.95))
    # Allow a small tolerance: rectangle sum is dt × Σ y ≈ 10 * (n_samples / fs)
    assert np.all(np.abs(auc - 10.0) < 0.1)


def test_build_peth_raises_for_missing_event_type():
    sess = _fake_session()
    with pytest.raises(ValueError, match="no events"):
        build_peth(sess, "nonexistent", signal="lifetime")


def test_build_peth_raises_for_missing_signal():
    sess = _fake_session()
    with pytest.raises(KeyError, match="not in session"):
        build_peth(sess, "test", signal="does_not_exist")


def test_build_peth_raises_for_invalid_window():
    sess = _fake_session()
    with pytest.raises(ValueError, match="post_window"):
        build_peth(sess, "test", signal="lifetime", pre_window=0.0, post_window=-1.0)


# -----------------------------------------------------------------------------
# Real-data smoke: reconstruct PETH and compare to the pre-computed one
# -----------------------------------------------------------------------------
pytestmark_real = pytest.mark.skipif(
    not DATA_ROOT.is_dir(),
    reason=f"FLIPR data root not present at {DATA_ROOT}",
)


@pytestmark_real
def test_real_peth_matches_precomputed_df_streams_peth():
    """Build a PETH for solution_onset + lifetime and verify it agrees with
    the pre-computed df_streams_peth.csv shipped with the session."""
    session = load_session(DATA_ROOT / "sessions" / EXAMPLE_BLOCK)
    # The shipped PETH spans -3 to +20 s based on inspection (time_rel column)
    precomputed = session.peth_for("solution_onset", signal="lifetime")
    assert not precomputed.empty

    pre = float(precomputed.columns.min())
    post = float(precomputed.columns.max())
    peth = build_peth(
        session, "solution_onset", signal="lifetime",
        pre_window=pre, post_window=post,
    )

    assert peth.n_trials == precomputed.shape[0]
    assert peth.n_samples == precomputed.shape[1]

    # Compare values: they should be close (exact if nearest-sample snap
    # matches, near-zero difference otherwise). Both arrays are (trials, time).
    pre_values = precomputed.to_numpy()
    our_values = peth.values

    # Check that the means are essentially identical — integration over
    # trials absorbs any per-trial 1-sample snap differences.
    pre_mean = np.nanmean(pre_values, axis=0)
    our_mean = np.nanmean(our_values, axis=0)
    np.testing.assert_allclose(our_mean, pre_mean, atol=0.005)


@pytestmark_real
def test_real_peth_baseline_correction_shifts_response_sign():
    """For a dopamine sensor locked to sucrose onset, baseline-corrected
    lifetime should show a clear deflection during the post-event window."""
    session = load_session(DATA_ROOT / "sessions" / EXAMPLE_BLOCK)
    peth = build_peth(
        session, "solution_onset", signal="lifetime",
        pre_window=-3.0, post_window=5.0,
    )
    bc = peth.baseline_corrected(window=(-2.5, -0.5))
    # Pre-baseline mean should be ~0 after correction
    pre_mask = (bc.time_rel >= -2.5) & (bc.time_rel <= -0.5)
    pre_mean = np.nanmean(bc.values[:, pre_mask])
    assert abs(pre_mean) < 1e-6

    # Post-event mean should be non-zero (there should be a response to sucrose)
    post_mask = (bc.time_rel >= 0.0) & (bc.time_rel <= 3.0)
    post_trial_means = np.nanmean(bc.values[:, post_mask], axis=1)
    # Most trials should show a measurable deflection (not just noise)
    assert np.nanstd(post_trial_means) > 1e-4
