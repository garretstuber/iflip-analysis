"""QC and motion-artifact detection for fluorescence lifetime photometry.

Unlike intensity-only photometry (dLight, GCaMP), lifetime is intrinsically
ratiometric — the arrival-time distribution doesn't depend on how many
photons you collect, only on how they're distributed. That means the
classic 405-nm-isosbestic motion regression doesn't transfer directly.
What *can* still go wrong on a FLIPR rig:

1. **Photon starvation** — when the fibre coupling drops too low, there
   aren't enough photons per bin to estimate a lifetime reliably, and the
   per-frame τ estimate gets noisy. Detected as ``intensity`` dropping
   below a user-chosen threshold.

2. **Motion-coupled lifetime changes** — if the mouse moves and lifetime
   AND intensity change together in a short window, something
   non-biological is perturbing both (e.g. mechanical stress on the
   fibre, PMT voltage change, scattering layer shift). Detected as high
   |rolling correlation| between intensity and lifetime.

   **Caveat:** for FLIM biosensors (e.g. FLIM-DA0.5), ligand binding
   changes both the fluorescence lifetime and the absorption/emission
   — so real biological signal can also produce moderate rolling
   correlations. Treat this flag as a diagnostic, not a rejection
   criterion. The default ``corr_threshold = 0.85`` is chosen to be
   conservative enough not to clip normal biosensor dynamics; lower it
   if you're investigating known motion epochs.

3. **Sudden intensity drops or jumps** — fibre bumps and touch events
   are well flagged by z-scoring the intensity derivative.

This module provides three detectors that each return a 1-D boolean mask
over the session (True = flagged), plus a convenience
:func:`compute_qc` that runs all three and returns a small DataFrame
with per-sample flags and the rolling correlation trace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flipr.io.session_csv import SessionData


@dataclass
class QCResult:
    """Per-sample QC diagnostics for a session."""

    time: np.ndarray  # seconds, matching session.streams.time
    intensity_flag: np.ndarray  # bool, low-photon mask
    motion_flag: np.ndarray  # bool, high |corr| mask
    jump_flag: np.ndarray  # bool, large dI/dt mask
    rolling_corr: np.ndarray  # rolling Pearson r(intensity, lifetime)
    any_flag: np.ndarray  # union of the three masks

    @property
    def fraction_flagged(self) -> float:
        return float(self.any_flag.mean())

    def summary(self) -> dict[str, float]:
        """Return a small dict of fractional flag rates for display."""
        return {
            "intensity": float(self.intensity_flag.mean()),
            "motion": float(self.motion_flag.mean()),
            "jump": float(self.jump_flag.mean()),
            "any": self.fraction_flagged,
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
    low_percentile: float = 1.0,
) -> np.ndarray:
    """Mark samples whose intensity is below ``min_photons``.

    If ``min_photons`` is None, it is set to the ``low_percentile``-th
    percentile of the intensity distribution (default 1%) — useful as a
    session-adaptive default.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    if min_photons is None:
        min_photons = float(np.percentile(intensity, low_percentile))
    return intensity < min_photons


def detect_motion_correlation(
    intensity: np.ndarray,
    lifetime: np.ndarray,
    *,
    window: int,
    corr_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Flag samples where rolling |r(intensity, lifetime)| exceeds a
    threshold. Returns ``(flag, rolling_corr)``.
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
    motion_window_s: float = 1.0,
    motion_corr_threshold: float = 0.85,
    jump_z_threshold: float = 6.0,
) -> QCResult:
    """Run the full QC suite on a session and return per-sample flags.

    Parameters
    ----------
    session
        Loaded :class:`SessionData`.
    min_photons
        Intensity threshold (photons per frame) below which the sample is
        flagged as photon-starved. Default ``None`` → 1st percentile.
    motion_window_s
        Rolling window (seconds) for the intensity/lifetime correlation.
    motion_corr_threshold
        Absolute correlation above which a window is flagged as likely
        motion.
    jump_z_threshold
        Robust z-score threshold for intensity jumps.
    """
    fs = session.fs
    t = session.streams["time"].to_numpy()
    intensity = session.streams["intensity"].to_numpy(dtype=np.float64)
    lifetime = session.streams["lifetime"].to_numpy(dtype=np.float64)

    window = max(3, int(round(motion_window_s * fs)))
    if window % 2 == 0:
        window += 1

    intensity_flag = detect_photon_starvation(intensity, min_photons=min_photons)
    motion_flag, rcorr = detect_motion_correlation(
        intensity, lifetime, window=window, corr_threshold=motion_corr_threshold
    )
    jump_flag = detect_intensity_jumps(intensity, z_threshold=jump_z_threshold)

    any_flag = intensity_flag | motion_flag | jump_flag

    return QCResult(
        time=t,
        intensity_flag=intensity_flag,
        motion_flag=motion_flag,
        jump_flag=jump_flag,
        rolling_corr=rcorr,
        any_flag=any_flag,
    )
