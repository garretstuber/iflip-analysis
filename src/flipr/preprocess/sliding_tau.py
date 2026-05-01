"""Sliding-window lifetime over the session.

Sweeps a window of fixed width across the TCSPC matrix, computes a
single τ readout per step, and returns ``(time, τ, n_photons)``. Useful
for spotting drift over a session (sensor bleaching, photodamage, slow
state changes) and for making event-aligned τ heatmaps without paying
the cost of fitting every frame.

Two kinds of methods:

- **phasor** (``phasor_phase`` / ``phasor_mod``) — one FFT per window,
  ~µs per step. Returns the apparent single-exponential lifetime
  computed from the phasor angle (``τ_phase``) or magnitude
  (``τ_modulation``). Equal for true single-exp data; divergence
  signals biexponential mixture or instrument drift. Default for
  interactive use.
- **fit** (``fit_double`` / ``fit_single``) — full curve_fit per
  window, ~10–100 ms per step. Returns the amplitude-weighted mean
  lifetime (double) or the single τ (single). Use for ground truth or
  for individual τ₁/τ₂ traces, but expect each step size halving to
  4× the runtime.

Background subtraction (per-window) is applied if a
:class:`flipr.preprocess.background.BackgroundEstimate` is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from flipr.preprocess.background import BackgroundEstimate, subtract_background
from flipr.preprocess.lifetime import fit_double_exp, fit_single_exp
from flipr.preprocess.phasor import phasor_from_histogram

SlidingMethod = Literal["phasor_phase", "phasor_mod", "fit_double", "fit_single"]

SLIDING_METHODS: tuple[SlidingMethod, ...] = (
    "phasor_phase",
    "phasor_mod",
    "fit_double",
    "fit_single",
)


@dataclass
class SlidingTauResult:
    """Output of :func:`sliding_tau`.

    ``tau`` is the primary readout — amplitude-weighted τ for fits,
    apparent τ_phase / τ_modulation for phasor methods. ``extra`` holds
    method-specific extras: ``tau1``/``tau2``/``chi2`` for fit_double,
    ``chi2`` for fit_single, etc.
    """

    method: SlidingMethod
    time: np.ndarray  # (n_steps,) window-centre times in seconds
    tau: np.ndarray  # (n_steps,) primary lifetime (ns)
    n_photons: np.ndarray  # (n_steps,) photons in each window
    window_s: float
    step_s: float
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.time.size)


def sliding_tau(
    tcspc: np.ndarray,
    bins_ns: np.ndarray,
    fs: float,
    *,
    method: SlidingMethod = "phasor_phase",
    window_s: float = 1.0,
    step_s: float = 0.5,
    period_ns: float = 12.5,
    background: BackgroundEstimate | None = None,
    bg_scale: float = 1.0,
    fit_kwargs: dict | None = None,
    progress_callback: callable | None = None,  # type: ignore[valid-type]
) -> SlidingTauResult:
    """Compute a lifetime trace by sliding a window across the TCSPC matrix.

    Parameters
    ----------
    tcspc
        ``(n_time, n_bins)`` integer histogram array.
    bins_ns
        ``(n_bins,)`` bin centres in ns.
    fs
        Sampling rate in Hz (used to convert ``window_s`` / ``step_s`` to
        frame counts).
    method
        One of :data:`SLIDING_METHODS`.
    window_s
        Window width (s).
    step_s
        Step between window centres (s). For ``step_s == window_s`` the
        windows are non-overlapping.
    period_ns
        Laser period (ns), passed through to the phasor / fit calls.
    background
        Optional :class:`BackgroundEstimate`. When supplied, each
        per-window summed histogram has ``bg_scale * window_n *
        bg.per_frame`` subtracted before fitting / phasor.
    bg_scale
        Scalar multiplier on the background. ``1.0`` assumes the bg
        was acquired under matched conditions.
    fit_kwargs
        Extra kwargs forwarded to :func:`fit_double_exp` or
        :func:`fit_single_exp` (e.g. ``irf_sigma_ns``, ``t0_ns``,
        ``fit_start_ns``, ``fit_stop_ns``). Ignored for phasor methods.
    progress_callback
        Optional ``f(i, n_steps)`` called every step — useful for
        a Streamlit progress bar with the slow fit methods.
    """
    tcspc = np.asarray(tcspc, dtype=np.float64)
    if tcspc.ndim != 2:
        raise ValueError(f"tcspc must be 2D (n_time, n_bins), got shape {tcspc.shape}")
    if window_s <= 0 or step_s <= 0:
        raise ValueError("window_s and step_s must be positive")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    n_frames_total = tcspc.shape[0]
    window_n = max(int(round(window_s * fs)), 1)
    step_n = max(int(round(step_s * fs)), 1)
    if window_n > n_frames_total:
        raise ValueError(
            f"window {window_s}s ({window_n} frames) longer than session "
            f"({n_frames_total} frames at {fs} Hz)"
        )

    starts = np.arange(0, n_frames_total - window_n + 1, step_n)
    n_steps = int(len(starts))
    if n_steps == 0:
        raise ValueError("no windows fit — pick a smaller window_s")

    # Window-centre time in seconds. Frame i has centre time (i + 0.5)/fs;
    # a window starting at frame s with width window_n has its centre at
    # frame s + window_n/2.
    times = (starts.astype(np.float64) + window_n / 2.0) / float(fs)
    taus = np.full(n_steps, np.nan, dtype=np.float64)
    n_photons = np.zeros(n_steps, dtype=np.float64)
    extra: dict[str, np.ndarray] = {}
    if method == "fit_double":
        extra["tau1"] = np.full(n_steps, np.nan)
        extra["tau2"] = np.full(n_steps, np.nan)
        extra["chi2"] = np.full(n_steps, np.nan)
    elif method == "fit_single":
        extra["chi2"] = np.full(n_steps, np.nan)

    fit_kwargs = dict(fit_kwargs or {})

    for i, s in enumerate(starts):
        e = s + window_n
        hist = tcspc[s:e].sum(axis=0)
        if background is not None:
            hist = subtract_background(hist, background, n_frames=window_n, scale=bg_scale)
        n_photons[i] = float(hist.sum())
        if n_photons[i] <= 0:
            if progress_callback is not None:
                progress_callback(i + 1, n_steps)
            continue

        if method == "phasor_phase":
            ph = phasor_from_histogram(hist, bins_ns, period_ns=period_ns)
            taus[i] = ph.tau_phase
        elif method == "phasor_mod":
            ph = phasor_from_histogram(hist, bins_ns, period_ns=period_ns)
            taus[i] = ph.tau_mod
        elif method == "fit_double":
            try:
                f = fit_double_exp(hist, bins_ns, period_ns=period_ns, **fit_kwargs)
                taus[i] = f.tau_amp_weighted
                extra["tau1"][i] = f.tau1
                extra["tau2"][i] = f.tau2
                extra["chi2"][i] = f.chi2_reduced
            except Exception:  # noqa: BLE001
                pass  # leave as NaN
        elif method == "fit_single":
            try:
                f = fit_single_exp(hist, bins_ns, period_ns=period_ns, **fit_kwargs)
                taus[i] = f.tau
                extra["chi2"][i] = f.chi2_reduced
            except Exception:  # noqa: BLE001
                pass
        else:
            raise ValueError(f"unknown method: {method!r}")

        if progress_callback is not None:
            progress_callback(i + 1, n_steps)

    return SlidingTauResult(
        method=method,
        time=times,
        tau=taus,
        n_photons=n_photons,
        window_s=window_s,
        step_s=step_s,
        extra=extra,
    )
