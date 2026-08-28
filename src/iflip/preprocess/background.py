"""Background subtraction for TCSPC histograms.

A background acquisition (typically labelled ``bg*`` in the FLIPR
data folder) records the autofluorescence + detector dark counts +
ambient light contribution under matched optical conditions but with
no biosensor signal. Subtracting a properly scaled bg histogram from
each test-window histogram before fitting or phasor analysis removes
this offset and gives a more accurate readout of the actual sensor
state — particularly important at low SNR where the bg shape can bias
the fit toward shorter τ.

The model is intensity-only: we assume the bg histogram shape is
stationary over the session and scales linearly with integration time.
That means the per-frame mean histogram from the bg acquisition,
multiplied by the number of frames in the test window, equals the
expected total bg contribution. A scalar ``scale`` knob lets the user
compensate for differences in laser power, fiber coupling, etc.
between the bg and test recordings.

Negative bins after subtraction (Poisson noise can push a few bins
below zero) are clamped to zero by default — ``curve_fit`` and the
phasor transform don't handle negative counts gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BackgroundEstimate:
    """Per-frame mean TCSPC histogram from a background recording."""

    per_frame: np.ndarray  # (n_bins,) mean histogram per frame, float64
    bins_ns: np.ndarray  # (n_bins,) matching bin centres in ns
    n_frames: int  # frames averaged into per_frame
    source_label: str  # human-readable description (blockname or path)

    @property
    def total_per_frame(self) -> float:
        """Total expected bg photons per frame (sum across all bins)."""
        return float(self.per_frame.sum())


def compute_background(
    tcspc: np.ndarray,
    bins_ns: np.ndarray,
    *,
    time_index: np.ndarray | None = None,
    time_window: tuple[float, float] | None = None,
    source_label: str = "background",
) -> BackgroundEstimate:
    """Compute the mean per-frame TCSPC histogram from a recording.

    Parameters
    ----------
    tcspc
        ``(n_time, n_bins)`` integer histogram array (e.g.
        ``IFlip2File.tcspc`` or ``TidyData.tcspc``).
    bins_ns
        ``(n_bins,)`` bin centres in ns.
    time_index
        Optional ``(n_time,)`` time axis (s). Required if ``time_window``
        is given.
    time_window
        Optional ``(t0, t1)`` time range (s) — only frames within
        ``[t0, t1]`` contribute. Useful when deriving a bg estimate from
        a quiet pre-stimulus window of the same recording. Default:
        use all frames.
    source_label
        Human-readable description for the UI (e.g. blockname of the
        bg acquisition).
    """
    tcspc = np.asarray(tcspc, dtype=np.float64)
    if tcspc.ndim != 2:
        raise ValueError(f"tcspc must be 2D (n_time, n_bins), got shape {tcspc.shape}")
    if tcspc.shape[1] != len(bins_ns):
        raise ValueError(
            f"tcspc bin axis ({tcspc.shape[1]}) and bins_ns ({len(bins_ns)}) mismatch"
        )

    if time_window is not None:
        if time_index is None:
            raise ValueError("time_window requires time_index to mask frames")
        t = np.asarray(time_index, dtype=np.float64)
        if t.shape[0] != tcspc.shape[0]:
            raise ValueError("time_index length does not match tcspc n_time")
        mask = (t >= time_window[0]) & (t <= time_window[1])
    else:
        mask = np.ones(tcspc.shape[0], dtype=bool)

    n = int(mask.sum())
    if n == 0:
        raise ValueError("no frames in background time window")

    per_frame = tcspc[mask].sum(axis=0) / float(n)
    return BackgroundEstimate(
        per_frame=per_frame,
        bins_ns=np.asarray(bins_ns, dtype=np.float64),
        n_frames=n,
        source_label=source_label,
    )


def subtract_background(
    histogram: np.ndarray,
    bg: BackgroundEstimate,
    *,
    n_frames: int = 1,
    scale: float = 1.0,
    clip_negative: bool = True,
) -> np.ndarray:
    """Subtract a scaled background from a TCSPC histogram.

    ``n_frames`` is the number of acquisition frames that were summed
    into ``histogram``. The bg total subtracted is
    ``scale * n_frames * bg.per_frame``. For per-frame inputs use
    ``n_frames=1`` (the default).

    Negative bins are clipped to zero unless ``clip_negative=False``.
    """
    if bg.per_frame.shape[0] != histogram.shape[-1]:
        raise ValueError(
            f"bg bins ({bg.per_frame.shape[0]}) and histogram bins "
            f"({histogram.shape[-1]}) mismatch"
        )
    bg_total = scale * float(n_frames) * bg.per_frame
    out = np.asarray(histogram, dtype=np.float64) - bg_total
    if clip_negative:
        out = np.maximum(out, 0.0)
    return out


def subtract_background_per_frame(
    tcspc: np.ndarray,
    bg: BackgroundEstimate,
    *,
    scale: float = 1.0,
    clip_negative: bool = True,
) -> np.ndarray:
    """Subtract per-frame background from a ``(n_time, n_bins)`` matrix.

    Each frame has ``scale * bg.per_frame`` subtracted independently.
    Output has the same shape as input.
    """
    tcspc = np.asarray(tcspc, dtype=np.float64)
    if tcspc.ndim != 2:
        raise ValueError(f"tcspc must be 2D, got shape {tcspc.shape}")
    if bg.per_frame.shape[0] != tcspc.shape[1]:
        raise ValueError(
            f"bg bins ({bg.per_frame.shape[0]}) and tcspc bins "
            f"({tcspc.shape[1]}) mismatch"
        )
    out = tcspc - (scale * bg.per_frame)[None, :]
    if clip_negative:
        out = np.maximum(out, 0.0)
    return out
