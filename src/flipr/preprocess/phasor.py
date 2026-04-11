"""Phasor analysis of TCSPC histograms, thin wrapper over PhasorPy.

The phasor approach to fluorescence lifetime analysis maps each TCSPC decay
curve to a single point ``(G, S)`` in 2-D phasor space, where

    G = ⟨cos(ωt)⟩        S = ⟨sin(ωt)⟩

are the normalised real and imaginary parts of the first Fourier harmonic
of the photon-arrival-time distribution, and ``ω = 2π / T`` is the
fundamental angular frequency set by the laser period ``T`` (12.5 ns at
80 MHz).

Single-exponential decays land on the *universal semicircle* centred at
(0.5, 0) with radius 0.5. Multi-exponential decays sit on the chord joining
the component positions, with the distance along the chord giving the
fractional contribution. This gives a model-free visual summary: a cloud
of points drifting around a chord is a signature of sensor state changes,
while a cloud drifting off the semicircle signals instrument drift,
afterpulsing, or fit-model mismatch.

PhasorPy handles all the heavy lifting (``phasor_from_signal``,
``phasor_to_apparent_lifetime``, ``phasor_from_lifetime``). This module
provides a small number of FLIPR-specific convenience functions:

- :func:`phasor_from_histogram` — one histogram → one (G, S) point.
- :func:`phasor_series_from_tcspc` — a full ``(n_time, n_bins)`` matrix →
  per-frame phasor trajectory for session-wide visualisation.
- :func:`phasor_for_known_taus` — semicircle positions for known τ values
  (useful for overlaying the component positions from a double-exp fit).
- :func:`semicircle_points` — universal semicircle curve for plotting.
- :func:`apparent_lifetimes` — invert a phasor point back to ``τ_phase``
  and ``τ_modulation`` for a quick single-lifetime readout.

Bin alignment
-------------
``phasor_from_signal`` treats its input as exactly one period of the
signal; the fundamental frequency is therefore ``1 / (N · bin_step)``. For
FLIPR data sampled at 0.1 ns, we slice the 0.0–12.5 ns window to exactly
125 bins before calling PhasorPy, giving an effective fundamental of
80 MHz (matching the rig's laser rep rate). Bins beyond 12.5 ns (or
partial last bins) are discarded.

Calibration
-----------
This module currently returns *uncalibrated* phasor coordinates: the
position includes both the physical decay and an instrument-dependent
phase rotation from the IRF. For absolute lifetime readout you would
typically acquire a reference dye with a known τ and call
:func:`phasorpy.lifetime.phasor_calibrate` to rotate + scale the cloud
onto the semicircle. A calibration helper will be added when we have a
reference acquisition to validate against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from phasorpy.lifetime import (
    phasor_from_lifetime,
    phasor_semicircle,
    phasor_to_apparent_lifetime,
)
from phasorpy.phasor import phasor_from_signal


@dataclass
class PhasorResult:
    """Phasor coordinates computed from a single TCSPC histogram."""

    real: float  # G
    imag: float  # S
    mean: float  # DC component (normalised intensity)
    frequency_mhz: float  # fundamental frequency used
    tau_phase: float  # ns, apparent lifetime from phase
    tau_mod: float  # ns, apparent lifetime from modulation
    n_bins_used: int
    n_photons: float

    @property
    def gs(self) -> tuple[float, float]:
        return (self.real, self.imag)


def _slice_to_period(
    histogram: np.ndarray,
    bin_ns: np.ndarray,
    period_ns: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Slice a histogram + bin axis to exactly ``period_ns``.

    Drops any bins whose centre (or left edge, whichever is stricter) lies
    beyond ``period_ns``. Returns ``(sliced_hist, sliced_bins, bin_step)``.
    """
    bin_ns = np.asarray(bin_ns, dtype=np.float64)
    histogram = np.asarray(histogram)
    if bin_ns.size < 2:
        raise ValueError("bin_ns must have at least two entries")
    bin_step = float(bin_ns[1] - bin_ns[0])
    if bin_step <= 0:
        raise ValueError("bin_ns must be monotonically increasing")
    # Use a tolerance to avoid FP surprises (0.1 × n → 12.500000000000004)
    tol = 1e-9
    keep = bin_ns < (period_ns - tol)
    if keep.sum() == 0:
        raise ValueError(f"no bins within period {period_ns}")
    return histogram[..., keep], bin_ns[keep], bin_step


def phasor_from_histogram(
    histogram: np.ndarray,
    bin_ns: np.ndarray,
    *,
    period_ns: float = 12.5,
) -> PhasorResult:
    """Phasor coordinates for a single TCSPC histogram.

    Parameters
    ----------
    histogram
        ``(n_bins,)`` photon counts per bin.
    bin_ns
        ``(n_bins,)`` bin centres in ns.
    period_ns
        Laser period. Data is sliced to the first full period before the
        phasor transform so the fundamental frequency equals ``1 / period``.
    """
    hist = np.asarray(histogram, dtype=np.float64)
    if hist.ndim != 1:
        raise ValueError(f"histogram must be 1D, got shape {hist.shape}")
    hist_sliced, _, bin_step = _slice_to_period(hist, bin_ns, period_ns)
    n_used = hist_sliced.shape[-1]
    if hist_sliced.sum() <= 0:
        return PhasorResult(
            real=float("nan"), imag=float("nan"), mean=0.0,
            frequency_mhz=1000.0 / (n_used * bin_step),
            tau_phase=float("nan"), tau_mod=float("nan"),
            n_bins_used=n_used, n_photons=0.0,
        )

    mean, real, imag = phasor_from_signal(hist_sliced, axis=-1)
    freq_mhz = 1000.0 / (n_used * bin_step)  # MHz
    tau_phi, tau_mod = phasor_to_apparent_lifetime(
        float(real), float(imag), freq_mhz
    )
    return PhasorResult(
        real=float(real),
        imag=float(imag),
        mean=float(mean),
        frequency_mhz=freq_mhz,
        tau_phase=float(tau_phi),
        tau_mod=float(tau_mod),
        n_bins_used=n_used,
        n_photons=float(hist_sliced.sum()),
    )


def phasor_series_from_tcspc(
    tcspc: np.ndarray,
    bin_ns: np.ndarray,
    *,
    period_ns: float = 12.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per-frame phasor coordinates for a ``(n_time, n_bins)`` TCSPC matrix.

    Useful for drawing a phasor *trajectory* over a session: each timepoint
    becomes a point in ``(G, S)`` space.

    Returns
    -------
    real : (n_time,) ndarray
        G coordinates.
    imag : (n_time,) ndarray
        S coordinates.
    mean : (n_time,) ndarray
        DC photon mean per frame (proportional to intensity).
    frequency_mhz : float
        Fundamental frequency used.

    Notes
    -----
    NaNs are returned for frames with zero photons in the sliced period.
    """
    tcspc = np.asarray(tcspc, dtype=np.float64)
    if tcspc.ndim != 2:
        raise ValueError(f"tcspc must be 2D (n_time, n_bins), got {tcspc.shape}")

    tcspc_sliced, _, bin_step = _slice_to_period(tcspc, bin_ns, period_ns)
    n_bins = tcspc_sliced.shape[-1]
    freq_mhz = 1000.0 / (n_bins * bin_step)

    mean, real, imag = phasor_from_signal(tcspc_sliced, axis=-1)
    mean_arr = np.asarray(mean, dtype=np.float64)
    real_arr = np.asarray(real, dtype=np.float64)
    imag_arr = np.asarray(imag, dtype=np.float64)

    # phasor_from_signal returns NaN where the frame sum is 0 — leave it;
    # callers that want to mask empty frames can use np.isfinite.
    return real_arr, imag_arr, mean_arr, freq_mhz


def apparent_lifetimes(
    real: np.ndarray | float,
    imag: np.ndarray | float,
    frequency_mhz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert phasor coordinates to ``(τ_phase, τ_modulation)`` in ns.

    ``τ_phase`` is dominated by the phase angle, ``τ_modulation`` by the
    distance from origin. For a single-exponential decay the two are equal;
    divergence indicates multi-exponential mixture or instrument drift.
    """
    tau_phi, tau_mod = phasor_to_apparent_lifetime(real, imag, frequency_mhz)
    return np.asarray(tau_phi, dtype=np.float64), np.asarray(tau_mod, dtype=np.float64)


def phasor_for_known_taus(
    tau_values: np.ndarray | list[float],
    *,
    period_ns: float = 12.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return semicircle positions for a set of known single-exponential τ values.

    Always returns 1-D ndarrays, even for single-element input (PhasorPy's
    underlying function can return 0-D arrays which are awkward to index).

    Use this to overlay the expected positions of the slow/fast components
    from a double-exponential fit on the phasor plot.
    """
    frequency_mhz = 1000.0 / period_ns
    taus = np.atleast_1d(np.asarray(tau_values, dtype=np.float64))
    G, S = phasor_from_lifetime(frequency_mhz, taus)
    return np.atleast_1d(np.asarray(G)).ravel(), np.atleast_1d(np.asarray(S)).ravel()


def semicircle_points(samples: int = 201) -> tuple[np.ndarray, np.ndarray]:
    """(G, S) arrays tracing the universal phasor semicircle for plotting."""
    return phasor_semicircle(samples)
