"""Peri-event time histogram (PETH) construction from session streams.

This module builds trial-aligned matrices for any continuous signal
(``intensity`` or ``lifetime``) around any event type in a session. It
deliberately rebuilds the PETH from the raw streams and event list rather
than reading the pre-computed ``df_streams_peth.csv`` that ships with each
session — that way the user can pick arbitrary pre/post windows, baseline
windows, and z-scoring reference periods without being tied to whatever
the acquisition pipeline chose.

Design choices
--------------
- Streams are assumed to be uniformly sampled at ``session.fs``. We nearest-
  neighbour snap event times to sample indices and take a fixed-width slice
  around each. For events near the session boundary, out-of-range samples
  are filled with NaN so downstream means (via ``nanmean``) stay well
  defined.
- The returned :class:`PETH` object is an immutable snapshot. Baseline
  correction and z-scoring return *new* PETH objects, so an original raw
  matrix can be inspected alongside its normalised variants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from flipr.io.session_csv import SessionData


@dataclass(frozen=True)
class PETH:
    """Trial-aligned matrix for a single (event_type, signal) combination."""

    time_rel: np.ndarray  # (n_samples,) seconds relative to event
    values: np.ndarray  # (n_trials, n_samples) signal values (NaN at boundaries)
    trial_ids: np.ndarray  # (n_trials,) 0-indexed trial numbers
    event_times: np.ndarray  # (n_trials,) absolute event times used
    signal_name: str  # "intensity" | "lifetime"
    event_type: str  # e.g. "solution_onset"
    fs: float  # Hz

    @property
    def n_trials(self) -> int:
        return self.values.shape[0]

    @property
    def n_samples(self) -> int:
        return self.values.shape[1]

    def mean(self) -> np.ndarray:
        """Nan-safe mean across trials, shape ``(n_samples,)``."""
        return np.nanmean(self.values, axis=0)

    def sem(self) -> np.ndarray:
        """Nan-safe SEM across trials, shape ``(n_samples,)``.

        Computed as ``nanstd(ddof=1) / sqrt(n_valid)`` so it behaves
        gracefully when some trials are missing samples at boundaries.
        """
        n_valid = np.sum(~np.isnan(self.values), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            std = np.nanstd(self.values, axis=0, ddof=1)
            sem = np.where(n_valid > 1, std / np.sqrt(n_valid), np.nan)
        return sem

    def baseline_corrected(self, window: tuple[float, float] = (-1.0, 0.0)) -> PETH:
        """Subtract each trial's mean over ``window`` (pre-event default)."""
        mask = self._window_mask(window)
        baseline = np.nanmean(self.values[:, mask], axis=1, keepdims=True)
        return replace(self, values=self.values - baseline)

    def zscored(self, window: tuple[float, float] = (-1.0, 0.0)) -> PETH:
        """Z-score each trial using the mean & std over ``window``."""
        mask = self._window_mask(window)
        mu = np.nanmean(self.values[:, mask], axis=1, keepdims=True)
        sigma = np.nanstd(self.values[:, mask], axis=1, keepdims=True, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sigma_safe = np.where(sigma > 0, sigma, np.nan)
            z = (self.values - mu) / sigma_safe
        return replace(self, values=z)

    def trial_auc(self, window: tuple[float, float]) -> np.ndarray:
        """Area under curve per trial over ``window`` (trapezoidal, NaN-safe).

        Returns shape ``(n_trials,)``; the integrand is the raw values.
        """
        mask = self._window_mask(window)
        if not mask.any():
            return np.full(self.n_trials, np.nan)
        dt = 1.0 / self.fs
        sub = self.values[:, mask]
        # Simple rectangle sum is robust to NaNs via nansum
        return np.nansum(sub, axis=1) * dt

    def _window_mask(self, window: tuple[float, float]) -> np.ndarray:
        lo, hi = window
        if hi <= lo:
            raise ValueError(f"window end must exceed start, got {window}")
        mask = (self.time_rel >= lo) & (self.time_rel <= hi)
        if not mask.any():
            raise ValueError(
                f"window {window} lies outside the PETH range "
                f"[{self.time_rel[0]:.2f}, {self.time_rel[-1]:.2f}]"
            )
        return mask


def build_peth(
    session: SessionData,
    event_type: str,
    *,
    signal: str = "lifetime",
    pre_window: float = -3.0,
    post_window: float = 5.0,
) -> PETH:
    """Build a PETH for ``event_type`` and ``signal`` from ``session``.

    Parameters
    ----------
    session
        Loaded :class:`SessionData`.
    event_type
        Name of the event (e.g. ``"solution_onset"``). Must be a value in
        ``session.events.event_id_char``.
    signal
        Column name in ``session.streams`` to extract (``"lifetime"`` or
        ``"intensity"``).
    pre_window
        Start of the extraction window in seconds (negative = before
        event). Default ``-3.0`` s.
    post_window
        End of the extraction window in seconds (positive = after event).
        Default ``5.0`` s.

    Returns
    -------
    PETH

    Raises
    ------
    KeyError
        If ``signal`` is not a column in ``session.streams``.
    ValueError
        If no events of ``event_type`` exist in the session, or if the
        sampling rate is non-positive, or if ``post_window <= pre_window``.
    """
    if signal not in session.streams.columns:
        raise KeyError(
            f"signal {signal!r} not in session.streams columns {list(session.streams.columns)}"
        )
    fs = session.fs
    if not (fs > 0):
        raise ValueError(f"invalid sampling rate: {fs}")
    if post_window <= pre_window:
        raise ValueError(f"post_window ({post_window}) must exceed pre_window ({pre_window})")

    t_stream = session.streams["time"].to_numpy()
    signal_arr = session.streams[signal].to_numpy()

    ev = session.events
    event_times = ev.loc[ev["event_id_char"] == event_type, "time"].to_numpy()
    if event_times.size == 0:
        raise ValueError(f"no events of type {event_type!r} in session")

    # Integer sample offsets for the window [pre, post]
    n_pre = int(round(abs(pre_window) * fs))
    n_post = int(round(post_window * fs))
    offsets = np.arange(-n_pre, n_post + 1)
    time_rel = offsets / fs  # honours the exact pre/post sign
    n_samples = offsets.size

    # Nearest-sample index for each event
    t0 = float(t_stream[0])
    event_idx = np.round((event_times - t0) * fs).astype(np.int64)

    # Absolute indices matrix: (n_trials, n_samples)
    abs_idx = event_idx[:, None] + offsets[None, :]
    valid = (abs_idx >= 0) & (abs_idx < signal_arr.size)

    values = np.full((event_times.size, n_samples), np.nan, dtype=np.float64)
    values[valid] = signal_arr[abs_idx[valid]]

    return PETH(
        time_rel=time_rel,
        values=values,
        trial_ids=np.arange(event_times.size),
        event_times=event_times,
        signal_name=signal,
        event_type=event_type,
        fs=fs,
    )
