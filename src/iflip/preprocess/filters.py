"""Temporal smoothing / filtering of session streams.

Provides a small set of 1-D filters commonly used in photometry and a
:func:`filtered_session` helper that returns a new :class:`SessionData`
with ``intensity`` and ``lifetime`` replaced by their smoothed versions.
All other fields (events, TCSPC, metadata) are untouched — so downstream
analyses see a filtered stream while per-window TCSPC re-fits and phasor
transforms continue to work from the raw histograms.

Filter modes
------------
- ``"none"`` — passthrough; returns the stream unchanged.
- ``"boxcar"`` — moving-average filter of width ``window_s`` seconds.
  Simplest to reason about; introduces a flat delay of ``window_s / 2``.
- ``"savgol"`` — Savitzky-Golay polynomial filter. Preserves peak heights
  and derivatives better than a boxcar; good default for lifetime traces.
- ``"median"`` — running median. Best for single-sample artefact removal
  (spikes, dropouts) since outliers don't bias the centre value.

Window sizes are specified in **seconds** and converted to sample counts
using the session's sampling rate. For Savitzky-Golay and median filters
the sample count is rounded up to the nearest odd number (required by
both filters). The minimum usable window is clamped to 3 samples
(Savitzky-Golay additionally requires ``window > polyorder``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import savgol_filter

from iflip.io.session_csv import SessionData

FilterMode = Literal["none", "boxcar", "savgol", "median"]

FILTER_MODES: tuple[FilterMode, ...] = ("none", "boxcar", "savgol", "median")


def _odd_at_least(n: int, minimum: int = 3) -> int:
    """Round ``n`` up to the nearest odd integer ≥ ``minimum``."""
    n = max(int(n), minimum)
    if n % 2 == 0:
        n += 1
    return n


def apply_filter(
    x: np.ndarray,
    *,
    mode: FilterMode,
    fs: float,
    window_s: float = 0.25,
    polyorder: int = 2,
) -> np.ndarray:
    """Return a smoothed copy of ``x``.

    Parameters
    ----------
    x
        1-D array of samples (typically ``session.streams["lifetime"]``).
    mode
        One of :data:`FILTER_MODES`.
    fs
        Sampling rate in Hz. Used to convert ``window_s`` to samples.
    window_s
        Filter width in seconds. Ignored for ``mode="none"``.
    polyorder
        Polynomial order for Savitzky-Golay. Clipped to ``window - 1``.
    """
    x = np.asarray(x, dtype=np.float64)
    if mode == "none" or window_s <= 0:
        return x.copy()

    width = max(1, int(round(window_s * fs)))
    if mode == "boxcar":
        if width < 2:
            return x.copy()
        return uniform_filter1d(x, size=width, mode="nearest")

    if mode == "median":
        size = _odd_at_least(width, minimum=3)
        return median_filter(x, size=size, mode="nearest")

    if mode == "savgol":
        window_length = _odd_at_least(width, minimum=5)
        if window_length >= x.size:
            window_length = _odd_at_least(x.size - 1, minimum=5)
        po = min(int(polyorder), window_length - 1)
        po = max(po, 1)
        return savgol_filter(x, window_length=window_length, polyorder=po, mode="nearest")

    raise ValueError(f"unknown filter mode: {mode!r}")


def filtered_session(
    session: SessionData,
    *,
    mode: FilterMode = "none",
    window_s: float = 0.25,
    polyorder: int = 2,
) -> SessionData:
    """Return a :class:`SessionData` whose ``intensity`` and ``lifetime``
    streams have been smoothed by :func:`apply_filter`.

    Short-circuits to the input session when ``mode == "none"`` so callers
    can wrap this unconditionally without paying a copy cost.
    """
    if mode == "none":
        return session

    new_streams = session.streams.copy()
    new_streams["intensity"] = apply_filter(
        new_streams["intensity"].to_numpy(),
        mode=mode,
        fs=session.fs,
        window_s=window_s,
        polyorder=polyorder,
    )
    new_streams["lifetime"] = apply_filter(
        new_streams["lifetime"].to_numpy(),
        mode=mode,
        fs=session.fs,
        window_s=window_s,
        polyorder=polyorder,
    )
    return replace(session, streams=new_streams)
