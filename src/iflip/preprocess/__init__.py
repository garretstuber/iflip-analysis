"""Preprocessing: lifetime re-fitting, phasor transforms, QC metrics."""

from iflip.preprocess.background import (
    BackgroundEstimate,
    compute_background,
    subtract_background,
    subtract_background_per_frame,
)
from iflip.preprocess.filters import (
    FILTER_MODES,
    FilterMode,
    apply_filter,
    filtered_session,
)
from iflip.preprocess.lifetime import (
    DoubleExpFit,
    LifetimeFit,
    SingleExpFit,
    double_exp_model,
    fit_double_exp,
    fit_information_criteria,
    fit_single_exp,
    single_exp_model,
)
from iflip.preprocess.motion import (
    QCResult,
    compute_qc,
    detect_intensity_jumps,
    detect_motion_correlation,
    detect_photon_starvation,
    rolling_corr,
)
from iflip.preprocess.phasor import (
    PhasorResult,
    apparent_lifetimes,
    phasor_for_known_taus,
    phasor_from_histogram,
    phasor_series_from_tcspc,
    semicircle_points,
)
from iflip.preprocess.sliding_tau import (
    SLIDING_METHODS,
    SlidingMethod,
    SlidingTauResult,
    sliding_tau,
)

__all__ = [
    "BackgroundEstimate",
    "DoubleExpFit",
    "FILTER_MODES",
    "FilterMode",
    "LifetimeFit",
    "PhasorResult",
    "QCResult",
    "SLIDING_METHODS",
    "SingleExpFit",
    "SlidingMethod",
    "SlidingTauResult",
    "apparent_lifetimes",
    "apply_filter",
    "compute_background",
    "compute_qc",
    "detect_intensity_jumps",
    "detect_motion_correlation",
    "detect_photon_starvation",
    "double_exp_model",
    "filtered_session",
    "fit_double_exp",
    "fit_information_criteria",
    "fit_single_exp",
    "phasor_for_known_taus",
    "phasor_from_histogram",
    "phasor_series_from_tcspc",
    "rolling_corr",
    "semicircle_points",
    "single_exp_model",
    "sliding_tau",
    "subtract_background",
    "subtract_background_per_frame",
]
