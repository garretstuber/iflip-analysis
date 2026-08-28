"""QC and motion-artifact detection for fluorescence lifetime photometry.

Unlike intensity-only photometry (dLight, GCaMP), lifetime is intrinsically
ratiometric — the arrival-time distribution doesn't depend on how many
photons you collect, only on how they're distributed. That means the
classic 405-nm-isosbestic motion regression doesn't transfer directly.
Two things actually go wrong on a FLIPR rig in practice:

1. **Photon starvation** — when fibre coupling drops too low, there
   aren't enough photons per frame to estimate a lifetime reliably and
   the per-frame τ becomes noisy. Detected as intensity samples ``k``
   MADs below the session median (default ``k = 5``). A head-fixed
   recording with stable coupling should return ~0 flags; a loose ferule
   or fibre bending will light up.

2. **Sudden intensity jumps or drops** — fibre bumps, touch events,
   cable snags. Detected by robust z-scoring the frame-to-frame
   intensity derivative (median + MAD).

These two conditions are the ``any_flag`` output of :func:`compute_qc`.

The module additionally computes a **rolling Pearson correlation**
between intensity and lifetime as a purely *diagnostic* trace. It is
**not** part of the flag mask because for FLIM biosensors (e.g.
FLIM-DA0.5), ligand binding changes both the fluorescence lifetime and
the absorption/emission cross-sections, so real biological signal
naturally produces moderate rolling correlations — a head-fixed
recording can easily hit |r| > 0.85 from the sensor responding to
reward delivery, not from motion. The trace is still worth plotting
because it helps users spot unusual coupling between the two channels
over the session, but it should not be used as a rejection criterion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from iflip.io.session_csv import SessionData


@dataclass
class QCResult:
    """Per-sample QC diagnostics for a session.

    ``any_flag`` is the union of ``intensity_flag`` and ``jump_flag`` —
    the two detectors that correspond to genuine artifacts on a FLIPR
    rig. ``rolling_corr`` and ``corr_above_threshold`` are purely
    diagnostic and are NOT part of ``any_flag``.
    """

    time: np.ndarray  # seconds, matching session.streams.time
    intensity_flag: np.ndarray  # bool, photon-starvation mask
    jump_flag: np.ndarray  # bool, large dI/dt mask
    rolling_corr: np.ndarray  # diagnostic: rolling Pearson r(intensity, lifetime)
    corr_above_threshold: np.ndarray  # diagnostic: |rolling_corr| > user threshold
    any_flag: np.ndarray  # intensity_flag | jump_flag

    @property
    def fraction_flagged(self) -> float:
        return float(self.any_flag.mean())

    def summary(self) -> dict[str, float]:
        """Return a small dict of fractional flag rates for display.

        ``intensity``, ``jump`` and ``any`` are true flag rates.
        ``corr_above_threshold``, ``mean_corr`` and ``max_abs_corr`` are
        diagnostic-only (not part of ``any``).
        """
        return {
            "intensity": float(self.intensity_flag.mean()),
            "jump": float(self.jump_flag.mean()),
            "any": self.fraction_flagged,
            "corr_above_threshold": float(self.corr_above_threshold.mean()),
            "mean_corr": float(np.nanmean(self.rolling_corr)),
            "max_abs_corr": float(np.nanmax(np.abs(self.rolling_corr))),
        }


def rolling_corr(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Centred rolling Pearson correlation between ``x`` and ``y``.

    Parameters
    ----------
    x, y
        1-D arrays of equal length.
    window
        Rolling window size in samples. Must be odd and ≥ 3.

    Returns
    -------
    ndarray of length ``len(x)``; edge samples where the window doesn't
    fit are filled with NaN.
    """
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    n = x.size
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and ≥ 3")
    half = window // 2
    out = np.full(n, np.nan, dtype=np.float64)

    # pandas rolling is well-vectorised and handles NaN sensibly
    s_x = pd.Series(x, dtype=np.float64)
    s_y = pd.Series(y, dtype=np.float64)
    corr = s_x.rolling(window=window, center=True, min_periods=window).corr(s_y)
    out[:] = corr.to_numpy()
    # Ensure edges are NaN (rolling can sometimes fill them)
    out[:half] = np.nan
    out[-half:] = np.nan
    return out


def detect_photon_starvation(
    intensity: np.ndarray,
    *,
    min_photons: float | None = None,
    mad_k: float = 5.0,
) -> np.ndarray:
    """Mark samples whose intensity is implausibly low.

    If ``min_photons`` is given, it is used as an absolute threshold.
    Otherwise the threshold is set adaptively to ``median − k · 1.4826 · MAD``
    with ``k = mad_k`` (default 5), i.e. a robust k-σ lower bound. On a
    stable recording this flags essentially nothing; on a session with
    real coupling drops or fibre bends it flags exactly the transient
    low-intensity epochs.

    Note: the percentile-based auto-threshold that previous versions
    used was tautological (always flagged ~1% of samples) — this robust
    version produces a flag rate of ~0% on clean sessions and lets the
    user see at a glance that the recording is fine.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    if min_photons is None:
        med = float(np.median(intensity))
        mad = float(np.median(np.abs(intensity - med))) + 1e-12
        threshold = med - mad_k * 1.4826 * mad
        # Don't go below zero photons — nonsensical
        min_photons = max(threshold, 0.0)
    return intensity < min_photons


def detect_motion_correlation(
    intensity: np.ndarray,
    lifetime: np.ndarray,
    *,
    window: int,
    corr_threshold: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a diagnostic rolling |r(intensity, lifetime)| mask.

    .. warning::
       For FLIM biosensors this should NOT be used as a motion rejection
       criterion — ligand binding naturally produces correlated changes
       in intensity and lifetime. The returned mask is kept for
       visualisation, but :func:`compute_qc` does not include it in the
       session-level ``any_flag`` output.
    """
    r = rolling_corr(intensity, lifetime, window=window)
    flag = np.zeros_like(r, dtype=bool)
    flag[np.isfinite(r)] = np.abs(r[np.isfinite(r)]) > corr_threshold
    return flag, r


def detect_intensity_jumps(
    intensity: np.ndarray,
    *,
    z_threshold: float = 6.0,
) -> np.ndarray:
    """Flag samples where the absolute frame-to-frame intensity derivative
    exceeds ``z_threshold`` robust z-scores.

    Uses median and MAD for robustness against isolated spikes.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    di = np.diff(intensity, prepend=intensity[0])
    med = np.median(di)
    mad = np.median(np.abs(di - med)) + 1e-12
    # robust z-score: (x - median) / (1.4826 * MAD)
    z = (di - med) / (1.4826 * mad)
    return np.abs(z) > z_threshold


def compute_qc(
    session: SessionData,
    *,
    min_photons: float | None = None,
    photon_mad_k: float = 5.0,
    motion_window_s: float = 1.0,
    motion_corr_threshold: float = 0.85,
    jump_z_threshold: float = 6.0,
) -> QCResult:
    """Run the QC detectors on a session.

    Parameters
    ----------
    session
        Loaded :class:`SessionData`.
    min_photons
        Absolute intensity threshold (photons per frame) below which the
        sample is flagged as photon-starved. Default ``None`` →
        robust ``median − photon_mad_k · 1.4826 · MAD`` threshold.
    photon_mad_k
        k-σ multiplier for the robust photon-starvation threshold.
        Default 5.0 ≈ essentially zero false positives on Gaussian-
        distributed data.
    motion_window_s
        Rolling window (seconds) for the diagnostic intensity/lifetime
        correlation. Does not affect the flag mask.
    motion_corr_threshold
        Cosmetic threshold above which correlation samples are
        highlighted in the UI. Does not affect the flag mask.
    jump_z_threshold
        Robust z-score threshold for intensity jumps. Default 6.0.

    Returns
    -------
    QCResult

    Notes
    -----
    ``any_flag`` is ``intensity_flag | jump_flag`` — the rolling
    correlation diagnostic is NOT included because, for FLIM biosensors,
    ligand binding naturally produces correlated intensity/lifetime
    changes that don't correspond to motion artifacts.
    """
    fs = session.fs
    t = session.streams["time"].to_numpy()
    intensity = session.streams["intensity"].to_numpy(dtype=np.float64)
    lifetime = session.streams["lifetime"].to_numpy(dtype=np.float64)

    window = max(3, int(round(motion_window_s * fs)))
    if window % 2 == 0:
        window += 1

    intensity_flag = detect_photon_starvation(
        intensity, min_photons=min_photons, mad_k=photon_mad_k
    )
    corr_flag, rcorr = detect_motion_correlation(
        intensity, lifetime, window=window, corr_threshold=motion_corr_threshold
    )
    jump_flag = detect_intensity_jumps(intensity, z_threshold=jump_z_threshold)

    # Only real-artifact detectors count toward any_flag
    any_flag = intensity_flag | jump_flag

    return QCResult(
        time=t,
        intensity_flag=intensity_flag,
        jump_flag=jump_flag,
        rolling_corr=rcorr,
        corr_above_threshold=corr_flag,
        any_flag=any_flag,
    )
