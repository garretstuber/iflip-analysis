"""Preprocessing: lifetime re-fitting, phasor transforms, QC metrics."""

from flipr.preprocess.filters import (
    FILTER_MODES,
    FilterMode,
    apply_filter,
    filtered_session,
)
from flipr.preprocess.lifetime import (
    DoubleExpFit,
    double_exp_model,
    fit_double_exp,
)
from flipr.preprocess.motion import (
    QCResult,
    compute_qc,
    detect_intensity_jumps,
    detect_motion_correlation,
    detect_photon_starvation,
    rolling_corr,
)
from flipr.preprocess.phasor import (
    PhasorResult,
    apparent_lifetimes,
    phasor_for_known_taus,
    phasor_from_histogram,
    phasor_series_from_tcspc,
    semicircle_points,
)

__all__ = [
    "DoubleExpFit",
    "FILTER_MODES",
    "FilterMode",
    "PhasorResult",
    "QCResult",
    "apparent_lifetimes",
    "apply_filter",
    "compute_qc",
    "detect_intensity_jumps",
    "detect_motion_correlation",
    "detect_photon_starvation",
    "double_exp_model",
    "filtered_session",
    "fit_double_exp",
    "phasor_for_known_taus",
    "phasor_from_histogram",
    "phasor_series_from_tcspc",
    "rolling_corr",
    "semicircle_points",
]
