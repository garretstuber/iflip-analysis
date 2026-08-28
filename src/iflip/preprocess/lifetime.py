"""Single- and double-exponential fitting of TCSPC decay histograms with a Gaussian IRF.

The forward model for a photon-count histogram ``h(t)`` sampled at bin centers
``t`` (ns) is a two-component exponentially-modified Gaussian (EMG) plus a
constant background. Because the excitation laser fires periodically with
period ``T`` (12.5 ns on an 80 MHz rig), the observed decay in any one period
is the sum of responses from the current pulse AND all previous pulses that
have not yet fully decayed:

    h(t) = Σ_n EMG_1(t, τ1, t0 − n·T) · α1
         + Σ_n EMG_1(t, τ2, t0 − n·T) · α2
         + background

where the sum over ``n`` includes the current pulse (``n = 0``) plus a small
number of wrap-around terms from previous pulses (``n ≥ 1``). For τ ≪ T the
wrap terms are negligible; for τ ≈ T/3 (as in the slow component of
FLIM-DA0.5, ``τ1 ≈ 3.8 ns``) the wrap adds several percent and biases a
constant-background fit toward shorter ``τ1`` if ignored.

Each ``EMG_1`` component is the convolution of a Gaussian IRF (center ``t0``,
1-σ width ``σ``) with a causal single exponential ``exp(−(t−t0)/τ)``:

    EMG_1(t, τ, t0) = (1/2) · exp(σ²/(2τ²) − (t−t0)/τ) · erfc((σ/τ − (t−t0)/σ)/√2)

The ``t0`` values of wrap copies are shifted by ``−n·T``; because n ≥ 1 puts
the rising edge many σ outside the current window, the erfc factor
asymptotes to 2 there and each wrap copy reduces to a pure exponential.

The implementation evaluates the component in two regimes for numerical
stability:

- ``a ≥ 0`` (rising edge / IRF region): use ``erfcx(a) · exp(-(t-t0)²/(2σ²))``
  (``erfcx`` is bounded for positive arguments).
- ``a < 0`` (decay tail): use ``erfc(a) · exp(σ²/(2τ²) - (t-t0)/τ)``
  (the exponent is bounded above by 0 in this regime, so ``exp`` is safe).

where ``a = σ/(τ√2) − (t−t0)/(σ√2)``. These two pieces agree analytically;
splitting keeps both finite on a TCSPC histogram spanning 0–12.5 ns.

Fits use weighted least-squares via :func:`scipy.optimize.curve_fit` with
Poisson-style per-bin uncertainties ``σᵢ = √(max(hᵢ, 1))``.

The reported ``alpha1`` / ``alpha2`` are raw scaling factors (their ratio is
meaningful; their absolute values scale with total photon count in the
fitted histogram).

Limitations of the v1 model
---------------------------
The IRF is modelled as a Gaussian. Real FLIPR IRFs have non-Gaussian tails
from detector after-pulsing, PMT transit-time spread, and reference-pulse
reflections. Fitting real rig data with a Gaussian IRF typically recovers
the amplitude-weighted mean lifetime accurately (within a few percent) but
can redistribute the two individual components — so ``τ1`` and ``τ2`` from
this fitter will not exactly match values reported by the acquisition
software, even though ``tau_amp_weighted`` does. Upgrading to a numerical
convolution with a measured IRF (from a reference-dye or scatterer
acquisition) is the standard fix and is a good v2 addition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc, erfcx


def _emg_component(
    t: np.ndarray,
    alpha: float,
    tau: float,
    t0: float,
    sigma: float,
) -> np.ndarray:
    """Single EMG component: Gaussian IRF (center ``t0``, width ``sigma``)
    convolved with ``alpha · exp(-(t-t0)/tau)`` for ``t ≥ t0``.
    """
    if tau <= 0.0 or sigma <= 0.0:
        return np.zeros_like(t)

    dt = t - t0
    inv_s_sqrt2 = 1.0 / (sigma * np.sqrt(2.0))
    a = sigma / (tau * np.sqrt(2.0)) - dt * inv_s_sqrt2

    out = np.zeros_like(t)
    mask_pos = a >= 0.0
    if mask_pos.any():
        gauss = np.exp(-(dt[mask_pos] ** 2) / (2.0 * sigma * sigma))
        out[mask_pos] = 0.5 * alpha * erfcx(a[mask_pos]) * gauss
    mask_neg = ~mask_pos
    if mask_neg.any():
        exponent = (sigma * sigma) / (2.0 * tau * tau) - dt[mask_neg] / tau
        out[mask_neg] = 0.5 * alpha * np.exp(exponent) * erfc(a[mask_neg])
    return out


def _emg_periodic(
    t: np.ndarray,
    alpha: float,
    tau: float,
    t0: float,
    sigma: float,
    period_ns: float,
    n_wraps: int,
) -> np.ndarray:
    """Periodic EMG: current pulse + ``n_wraps`` previous pulses summed.

    A wrap count of 0 recovers the single-pulse EMG. The number of wraps
    needed to reach numerical convergence depends on ``τ / period_ns``: with
    ``n_wraps = 5`` the residual is ``exp(−5·T/τ)``, which for τ = 3.8 ns and
    T = 12.5 ns is ~7·10⁻⁸ — safely below shot noise.
    """
    out = _emg_component(t, alpha, tau, t0, sigma)
    if period_ns <= 0 or n_wraps <= 0 or tau <= 0:
        return out
    for n in range(1, n_wraps + 1):
        out = out + _emg_component(t, alpha, tau, t0 - n * period_ns, sigma)
    return out


def double_exp_model(
    t: np.ndarray,
    alpha1: float,
    tau1: float,
    alpha2: float,
    tau2: float,
    background: float,
    t0: float,
    sigma: float,
    *,
    period_ns: float = 12.5,
    n_wraps: int = 5,
) -> np.ndarray:
    """Full forward model: sum of two periodic EMG components plus background.

    Parameters
    ----------
    period_ns
        Laser period (ns). Default 12.5 matches an 80 MHz rep rate. Set to 0
        (or ``n_wraps = 0``) to disable wrap-around for a single-pulse model.
    n_wraps
        Number of previous pulses to include. 5 is enough for any
        τ ≲ period_ns / 2; bump higher only if you have τ approaching T.
    """
    t = np.asarray(t, dtype=np.float64)
    return (
        _emg_periodic(t, alpha1, tau1, t0, sigma, period_ns, n_wraps)
        + _emg_periodic(t, alpha2, tau2, t0, sigma, period_ns, n_wraps)
        + background
    )


@dataclass
class DoubleExpFit:
    """Result of a :func:`fit_double_exp` call.

    ``tau1 ≥ tau2`` by convention after the fit (slow, fast).
    """

    alpha1: float
    tau1: float  # ns, slow component
    alpha2: float
    tau2: float  # ns, fast component
    background: float
    t0: float  # ns
    sigma: float  # ns, IRF 1-σ width
    chi2_reduced: float
    residuals: np.ndarray  # full-range (histogram - model)
    residuals_weighted: (
        np.ndarray
    )  # (histogram - model) / sqrt(max(histogram, 1)), restricted to fit_mask
    model: np.ndarray  # full-range model values
    fit_mask: np.ndarray  # bool, which bins were used for fit + chi²
    n_photons: float  # sum of histogram inside fit range
    success: bool
    message: str

    @property
    def tau_amp_weighted(self) -> float:
        """Amplitude-weighted mean lifetime: ``(α1·τ1 + α2·τ2)/(α1+α2)``.

        Matches the instrument's ``avgtau`` convention.
        """
        total = self.alpha1 + self.alpha2
        if total <= 0.0:
            return float("nan")
        return (self.alpha1 * self.tau1 + self.alpha2 * self.tau2) / total

    @property
    def tau_int_weighted(self) -> float:
        """Intensity-weighted mean lifetime: ``(α1·τ1² + α2·τ2²)/(α1·τ1 + α2·τ2)``."""
        num = self.alpha1 * self.tau1 * self.tau1 + self.alpha2 * self.tau2 * self.tau2
        den = self.alpha1 * self.tau1 + self.alpha2 * self.tau2
        if den <= 0.0:
            return float("nan")
        return num / den

    @property
    def pop1_fraction(self) -> float:
        """Fractional amplitude of the slow component: ``α1 / (α1 + α2)``."""
        total = self.alpha1 + self.alpha2
        if total <= 0.0:
            return float("nan")
        return self.alpha1 / total

    @property
    def pop2_fraction(self) -> float:
        return 1.0 - self.pop1_fraction


def fit_double_exp(
    histogram: np.ndarray,
    bin_ns: np.ndarray,
    *,
    irf_sigma_ns: float = 0.21,
    t0_ns: float = 1.012,
    fit_start_ns: float = 0.4,
    fit_stop_ns: float = 12.3,
    period_ns: float = 12.5,
    n_wraps: int = 5,
    initial: dict[str, float] | None = None,
    fix_t0: bool = True,
    fix_sigma: bool = True,
) -> DoubleExpFit:
    """Fit a Gaussian-IRF-convolved double exponential to a TCSPC histogram.

    Parameters
    ----------
    histogram
        ``(n_bins,)`` photon counts per bin.
    bin_ns
        ``(n_bins,)`` bin centers in ns. Must match ``histogram`` shape.
    irf_sigma_ns
        Gaussian IRF 1-σ width in ns. Default 0.21 matches the FLIM-DA0.5 rig
        (``header_state_beta6 = 0.212``).
    t0_ns
        Arrival time zero (IRF center) in ns. Default 1.012 matches the rig
        (``header_state_t0``).
    fit_start_ns, fit_stop_ns
        SPC range (ns) used for the fit and χ² calculation. Bins outside this
        range are still included in the full-range ``model`` and ``residuals``
        arrays but do not contribute to parameter estimation.
    initial
        Optional starting values. Keys: ``alpha1, tau1, alpha2, tau2, background``.
    period_ns
        Laser period in ns. Default 12.5 (80 MHz rep rate). The model sums
        contributions from the current pulse and ``n_wraps`` previous pulses
        so that slow components (τ ≳ period/5) are fit correctly rather than
        having their wrap-around photons absorbed into ``background``.
    n_wraps
        Number of previous-pulse wraps to include. See :func:`double_exp_model`.
    fix_t0, fix_sigma
        If ``True`` (default), ``t0`` and IRF σ are held fixed. If ``False``,
        they become free parameters with modest bounds.

    Returns
    -------
    DoubleExpFit

    Notes
    -----
    The fit is a weighted least-squares (Neyman χ²) with per-bin σ = √max(N,1).
    For histograms with < ~500 photons total the fit becomes noisy; check
    ``chi2_reduced`` and ``n_photons`` before trusting the parameters.
    """
    histogram = np.asarray(histogram, dtype=np.float64)
    bin_ns = np.asarray(bin_ns, dtype=np.float64)

    if histogram.shape != bin_ns.shape:
        raise ValueError(f"histogram {histogram.shape} and bin_ns {bin_ns.shape} must match")
    if fit_stop_ns <= fit_start_ns:
        raise ValueError(f"fit_stop_ns ({fit_stop_ns}) must exceed fit_start_ns ({fit_start_ns})")

    fit_mask = (bin_ns >= fit_start_ns) & (bin_ns <= fit_stop_ns)
    if not fit_mask.any():
        raise ValueError(f"fit range [{fit_start_ns}, {fit_stop_ns}] ns contains no bins")

    t_fit = bin_ns[fit_mask]
    y_fit = histogram[fit_mask]
    sigma_fit = np.sqrt(np.maximum(y_fit, 1.0))
    n_photons = float(y_fit.sum())

    # Initial guesses: peak-normalized amplitudes + background from pre-t0 bins
    pre_mask = bin_ns < max(t0_ns - 0.5, bin_ns[0])
    bg_guess = float(np.median(histogram[pre_mask])) if pre_mask.any() else 0.0
    peak = float(np.max(y_fit)) if y_fit.size else 1.0
    init: dict[str, float] = {
        "alpha1": peak * 0.3,
        "tau1": 3.5,
        "alpha2": peak * 1.2,
        "tau2": 0.6,
        "background": max(bg_guess, 0.0),
    }
    if initial:
        init.update(initial)

    # Build model wrapper + bounds depending on what's fixed. period_ns and
    # n_wraps are baked in (they come from the laser rep rate, not the fit).
    if fix_t0 and fix_sigma:

        def _model(t, a1, tau1, a2, tau2, bg):
            return double_exp_model(
                t,
                a1,
                tau1,
                a2,
                tau2,
                bg,
                t0_ns,
                irf_sigma_ns,
                period_ns=period_ns,
                n_wraps=n_wraps,
            )

        p0 = [init["alpha1"], init["tau1"], init["alpha2"], init["tau2"], init["background"]]
        lo = [0.0, 0.05, 0.0, 0.05, 0.0]
        hi = [np.inf, 20.0, np.inf, 20.0, np.inf]
    elif fix_sigma:

        def _model(t, a1, tau1, a2, tau2, bg, t0):  # noqa: PLR0913
            return double_exp_model(
                t,
                a1,
                tau1,
                a2,
                tau2,
                bg,
                t0,
                irf_sigma_ns,
                period_ns=period_ns,
                n_wraps=n_wraps,
            )

        p0 = [
            init["alpha1"],
            init["tau1"],
            init["alpha2"],
            init["tau2"],
            init["background"],
            t0_ns,
        ]
        lo = [0.0, 0.05, 0.0, 0.05, 0.0, float(bin_ns[0])]
        hi = [np.inf, 20.0, np.inf, 20.0, np.inf, float(bin_ns[-1])]
    else:

        def _model(t, a1, tau1, a2, tau2, bg, t0, sig):  # noqa: PLR0913
            return double_exp_model(
                t,
                a1,
                tau1,
                a2,
                tau2,
                bg,
                t0,
                sig,
                period_ns=period_ns,
                n_wraps=n_wraps,
            )

        p0 = [
            init["alpha1"],
            init["tau1"],
            init["alpha2"],
            init["tau2"],
            init["background"],
            t0_ns,
            irf_sigma_ns,
        ]
        lo = [0.0, 0.05, 0.0, 0.05, 0.0, float(bin_ns[0]), 0.01]
        hi = [np.inf, 20.0, np.inf, 20.0, np.inf, float(bin_ns[-1]), 2.0]

    try:
        popt, _ = curve_fit(
            _model,
            t_fit,
            y_fit,
            p0=p0,
            bounds=(lo, hi),
            sigma=sigma_fit,
            absolute_sigma=False,
            maxfev=5000,
        )
        success = True
        message = "ok"
    except Exception as exc:  # noqa: BLE001
        popt = np.asarray(p0, dtype=np.float64)
        success = False
        message = f"curve_fit failed: {exc}"

    # Unpack back into named params
    if fix_t0 and fix_sigma:
        alpha1, tau1, alpha2, tau2, background = popt
        t0_final, sigma_final = t0_ns, irf_sigma_ns
    elif fix_sigma:
        alpha1, tau1, alpha2, tau2, background, t0_final = popt
        sigma_final = irf_sigma_ns
    else:
        alpha1, tau1, alpha2, tau2, background, t0_final, sigma_final = popt

    # Canonical ordering: tau1 is the slower component
    if tau2 > tau1:
        tau1, tau2 = tau2, tau1
        alpha1, alpha2 = alpha2, alpha1

    model_full = double_exp_model(
        bin_ns,
        alpha1,
        tau1,
        alpha2,
        tau2,
        background,
        t0_final,
        sigma_final,
        period_ns=period_ns,
        n_wraps=n_wraps,
    )
    residuals_full = histogram - model_full
    residuals_weighted_fit = residuals_full[fit_mask] / sigma_fit
    dof = max(int(fit_mask.sum()) - len(popt), 1)
    chi2_reduced = float(np.sum(residuals_weighted_fit**2) / dof)

    return DoubleExpFit(
        alpha1=float(alpha1),
        tau1=float(tau1),
        alpha2=float(alpha2),
        tau2=float(tau2),
        background=float(background),
        t0=float(t0_final),
        sigma=float(sigma_final),
        chi2_reduced=chi2_reduced,
        residuals=residuals_full,
        residuals_weighted=residuals_weighted_fit,
        model=model_full,
        fit_mask=fit_mask,
        n_photons=n_photons,
        success=success,
        message=message,
    )


# --------------------------------------------------------------------------- #
# Single-exponential model + fit
# --------------------------------------------------------------------------- #


def single_exp_model(
    t: np.ndarray,
    alpha: float,
    tau: float,
    background: float,
    t0: float,
    sigma: float,
    *,
    period_ns: float = 12.5,
    n_wraps: int = 5,
) -> np.ndarray:
    """Single periodic EMG component plus background.

    Same forward model as :func:`double_exp_model` with the second component
    set to zero; useful when the decay is monoexponential or as a null
    model for AIC/BIC comparison.
    """
    t = np.asarray(t, dtype=np.float64)
    return _emg_periodic(t, alpha, tau, t0, sigma, period_ns, n_wraps) + background


@dataclass
class SingleExpFit:
    """Result of a :func:`fit_single_exp` call."""

    alpha: float
    tau: float  # ns
    background: float
    t0: float  # ns
    sigma: float  # ns, IRF 1-σ width
    chi2_reduced: float
    residuals: np.ndarray
    residuals_weighted: np.ndarray
    model: np.ndarray
    fit_mask: np.ndarray
    n_photons: float
    success: bool
    message: str

    # Convenience aliases so the viz layer can treat single/double uniformly
    @property
    def tau_amp_weighted(self) -> float:
        return float(self.tau)

    @property
    def tau_int_weighted(self) -> float:
        return float(self.tau)

    @property
    def n_components(self) -> int:
        return 1


def fit_single_exp(
    histogram: np.ndarray,
    bin_ns: np.ndarray,
    *,
    irf_sigma_ns: float = 0.21,
    t0_ns: float = 1.012,
    fit_start_ns: float = 0.4,
    fit_stop_ns: float = 12.3,
    period_ns: float = 12.5,
    n_wraps: int = 5,
    initial: dict[str, float] | None = None,
    fix_t0: bool = True,
    fix_sigma: bool = True,
) -> SingleExpFit:
    """Fit a Gaussian-IRF-convolved single exponential to a TCSPC histogram.

    Parameters mirror :func:`fit_double_exp` exactly. Useful when the
    decay is genuinely monoexponential (e.g. a reference dye) or as a
    null model for AIC/BIC comparison against the double-exponential
    fit on the same data.
    """
    histogram = np.asarray(histogram, dtype=np.float64)
    bin_ns = np.asarray(bin_ns, dtype=np.float64)

    if histogram.shape != bin_ns.shape:
        raise ValueError(f"histogram {histogram.shape} and bin_ns {bin_ns.shape} must match")
    if fit_stop_ns <= fit_start_ns:
        raise ValueError(f"fit_stop_ns ({fit_stop_ns}) must exceed fit_start_ns ({fit_start_ns})")

    fit_mask = (bin_ns >= fit_start_ns) & (bin_ns <= fit_stop_ns)
    if not fit_mask.any():
        raise ValueError(f"fit range [{fit_start_ns}, {fit_stop_ns}] ns contains no bins")

    t_fit = bin_ns[fit_mask]
    y_fit = histogram[fit_mask]
    sigma_fit = np.sqrt(np.maximum(y_fit, 1.0))
    n_photons = float(y_fit.sum())

    pre_mask = bin_ns < max(t0_ns - 0.5, bin_ns[0])
    bg_guess = float(np.median(histogram[pre_mask])) if pre_mask.any() else 0.0
    peak = float(np.max(y_fit)) if y_fit.size else 1.0
    init: dict[str, float] = {
        "alpha": peak * 1.0,
        "tau": 2.5,
        "background": max(bg_guess, 0.0),
    }
    if initial:
        init.update(initial)

    if fix_t0 and fix_sigma:

        def _model(t, a, tau, bg):
            return single_exp_model(
                t, a, tau, bg, t0_ns, irf_sigma_ns,
                period_ns=period_ns, n_wraps=n_wraps,
            )

        p0 = [init["alpha"], init["tau"], init["background"]]
        lo = [0.0, 0.05, 0.0]
        hi = [np.inf, 20.0, np.inf]
    elif fix_sigma:

        def _model(t, a, tau, bg, t0):
            return single_exp_model(
                t, a, tau, bg, t0, irf_sigma_ns,
                period_ns=period_ns, n_wraps=n_wraps,
            )

        p0 = [init["alpha"], init["tau"], init["background"], t0_ns]
        lo = [0.0, 0.05, 0.0, float(bin_ns[0])]
        hi = [np.inf, 20.0, np.inf, float(bin_ns[-1])]
    else:

        def _model(t, a, tau, bg, t0, sig):
            return single_exp_model(
                t, a, tau, bg, t0, sig,
                period_ns=period_ns, n_wraps=n_wraps,
            )

        p0 = [init["alpha"], init["tau"], init["background"], t0_ns, irf_sigma_ns]
        lo = [0.0, 0.05, 0.0, float(bin_ns[0]), 0.01]
        hi = [np.inf, 20.0, np.inf, float(bin_ns[-1]), 2.0]

    try:
        popt, _ = curve_fit(
            _model,
            t_fit,
            y_fit,
            p0=p0,
            bounds=(lo, hi),
            sigma=sigma_fit,
            absolute_sigma=False,
            maxfev=5000,
        )
        success = True
        message = "ok"
    except Exception as exc:  # noqa: BLE001
        popt = np.asarray(p0, dtype=np.float64)
        success = False
        message = f"curve_fit failed: {exc}"

    if fix_t0 and fix_sigma:
        alpha, tau, background = popt
        t0_final, sigma_final = t0_ns, irf_sigma_ns
    elif fix_sigma:
        alpha, tau, background, t0_final = popt
        sigma_final = irf_sigma_ns
    else:
        alpha, tau, background, t0_final, sigma_final = popt

    model_full = single_exp_model(
        bin_ns, alpha, tau, background, t0_final, sigma_final,
        period_ns=period_ns, n_wraps=n_wraps,
    )
    residuals_full = histogram - model_full
    residuals_weighted_fit = residuals_full[fit_mask] / sigma_fit
    dof = max(int(fit_mask.sum()) - len(popt), 1)
    chi2_reduced = float(np.sum(residuals_weighted_fit**2) / dof)

    return SingleExpFit(
        alpha=float(alpha),
        tau=float(tau),
        background=float(background),
        t0=float(t0_final),
        sigma=float(sigma_final),
        chi2_reduced=chi2_reduced,
        residuals=residuals_full,
        residuals_weighted=residuals_weighted_fit,
        model=model_full,
        fit_mask=fit_mask,
        n_photons=n_photons,
        success=success,
        message=message,
    )


# Convenience union for downstream code
LifetimeFit = SingleExpFit | DoubleExpFit


def fit_information_criteria(fit: LifetimeFit, n_params: int) -> dict[str, float]:
    """Compute Akaike (AIC) and Bayesian (BIC) information criteria for a fit.

    Uses the Gaussian (Neyman χ²) approximation:

        −2·log L ≈ Σᵢ ((hᵢ − mᵢ) / σᵢ)²

    so that ``AIC = 2k + Σ(weighted residuals²)`` and
    ``BIC = k·ln(n) + Σ(weighted residuals²)``. Lower is better; the
    *difference* between two models is what's interpretable.
    """
    rss = float(np.sum(fit.residuals_weighted**2))
    n = int(fit.fit_mask.sum())
    aic = 2 * n_params + rss
    bic = n_params * np.log(max(n, 1)) + rss
    return {"aic": float(aic), "bic": float(bic), "rss": rss, "n": n}
