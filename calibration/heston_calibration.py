"""
calibration/heston_calibration.py
====================================
Calibrates Heston's (kappa, theta, xi, rho, v0) directly from a historical
equity RETURN series -- no option data needed. This is a legitimate
alternative to the (more common in industry) option-implied-vol
calibration: we use the "method of moments" idea applied to the CIR
variance process, using rolling realized variance as an observable proxy
for the latent variance v(t).

THE MOMENT CONDITIONS USED (all standard, exact properties of a
stationary CIR process -- see e.g. Cox-Ingersoll-Ross 1985):
  1. E[v]        = theta                        (long-run mean level)
  2. Var[v]       = theta * xi^2 / (2*kappa)      (stationary variance)
  3. ACF(v, lag)  = exp(-kappa * lag)             (exact exponential decay
                                                    of the autocorrelation
                                                    function of a CIR
                                                    process -- this is a
                                                    textbook result, not
                                                    an approximation)
  4. rho          = leading-order correlation between return shocks and
                     variance shocks (see analysis/realized_variance.py
                     for the derivation) -- estimated directly, not part
                     of the least-squares fit below.

We fit (kappa, theta, xi) by nonlinear least squares, matching the MODEL's
predicted moments (1)-(3) to the EMPIRICAL moments computed from a rolling
realized-variance series -- using the ACF at MANY lags (not just one)
over-identifies the system (3 unknowns, dozens of moment conditions),
making the fit robust to noise in any single lag's ACF estimate. This is
a form of the Generalized Method of Moments (GMM) with an identity/simple
diagonal weighting (a full optimal GMM weighting matrix, which requires
estimating the covariance of the moment conditions themselves, is a
further refinement out of scope for this project -- flagged as a
simplification worth mentioning in the viva).

THIS APPROACH'S LIMITATION (state clearly in the report): realized
variance is a noisy, backward-looking ESTIMATE of v(t), not v(t) itself,
so this calibration is only as good as that proxy. It also uses
approximate (leading-order-in-dt / stationary-process) moment formulas
rather than the exact finite-sample transition density of the CIR
process -- a full Kalman/particle-filter MLE approach would use the exact
(or near-exact) likelihood and is more statistically efficient, but is
substantially more complex to implement correctly. Given the project's
scope and timeline, GMM-on-realized-variance is the right complexity
tradeoff -- correct in its own right, just less efficient (in the
statistical sense) than the alternative.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, differential_evolution

from analysis.realized_variance import realized_variance, acf, leverage_correlation
from models.heston import feller_condition
from models.heston_mc import simulate_heston_paths


def _model_acf(kappa, lags):
    """Model-implied ACF(lag) = exp(-kappa*lag) for a CIR process, lag in years."""
    return np.exp(-kappa * lags)


def _model_var(kappa, theta, xi):
    """Model-implied stationary variance of v: theta*xi^2/(2*kappa)."""
    return theta * xi**2 / (2 * kappa)


def estimate_rho_attenuation_factor(kappa, theta, xi, window=21, trading_days_per_year=252,
                                     T_calib=10, n_calib_seeds=10, reference_rho=0.7):
    """
    ATTENUATION BIAS CORRECTION for the leverage-correlation estimator
    (analysis.realized_variance.leverage_correlation).

    THE PROBLEM: that function's Ito-derived identity Corr(dr,dv)=rho only
    holds for the TRUE, continuously-observed instantaneous variance
    process. We only have daily returns, and even after fixing the
    contamination bug (see leverage_correlation's docstring), estimating
    dv via any daily-return-based proxy (single squared returns, or
    windowed realized variance) is extremely noisy -- daily r^2 is a
    chi-squared(1) estimate of v*dt, with huge relative variance. This
    inflates the DENOMINATOR of the correlation estimate (Var of the noisy
    proxy) without changing the numerator (true covariance with r_t) by
    nearly as much, which SEVERELY attenuates (shrinks toward zero) the
    estimated correlation -- verified empirically: simulating Heston with
    a KNOWN rho and applying the estimator recovers only ~10-20% of the
    true rho's magnitude (correct sign, wrong scale). This is a
    well-known phenomenon in the realized-volatility literature (why
    practitioners use intraday data for this specific estimate, which
    daily OHLC from yfinance cannot provide).

    THE FIX: since attenuation bias, for a fixed (kappa, theta, xi, window),
    is empirically close to LINEAR in rho (checked directly: the ratio
    raw_estimate/true_rho is roughly constant across true_rho in
    [-0.8, 0.8], up to Monte Carlo sampling noise), we calibrate the
    attenuation factor directly using our own validated Monte Carlo
    simulator: simulate several independent Heston paths using the
    CANDIDATE (kappa, theta, xi) -- i.e. the closed-form initial guess
    already computed from the real data, so the correction is tailored to
    this specific ticker's vol-of-vol/mean-reversion regime, not a
    universal constant -- with a known reference_rho, run the SAME
    leverage_correlation estimator on the simulated data, and take the
    ratio raw/reference_rho, averaged over several seeds to reduce Monte
    Carlo noise in the correction factor itself (averaging 10 independent
    10-year simulations reduces the correction factor's own relative
    standard error from ~20% per single path to a few percent).

    We then divide the REAL DATA's raw estimate by this factor to get the
    bias-corrected rho.

    CAVEAT (state in the report): this correction is only as good as the
    linearity assumption and the accuracy of (kappa, theta, xi) used to
    generate the calibration simulations -- it is a practical, defensible
    approximation, not an exact debiasing, and the corrected rho should be
    read as "the right order of magnitude and sign," not a high-precision
    estimate.

    Returns
    -------
    float, the estimated attenuation factor (raw_corr / true_rho), > 0.
    """
    ratios = []
    n_steps = int(trading_days_per_year * T_calib)
    for seed in range(n_calib_seeds):
        _, S, _ = simulate_heston_paths(100.0, theta, T=T_calib, n_steps=n_steps, n_paths=1,
                                         kappa=kappa, theta=theta, xi=xi, rho=reference_rho,
                                         r=0.0, q=0.0, seed=seed, antithetic=False)
        price_path = S[0]

        # ROBUSTNESS GUARD: this helper runs on whatever (kappa, theta, xi)
        # a calibration in progress currently proposes -- including,
        # during multi-start / walk-forward exploration, occasional
        # extreme or poorly-conditioned combinations (very large xi with
        # small kappa). Over T_calib*trading_days_per_year compounded
        # Euler steps, an extreme combination can drive the simulated
        # price to underflow to exactly 0.0 in float64, which would make
        # np.log(0) = -inf and corrupt this seed's correlation estimate
        # with NaN/inf. Since we only need an approximate, order-of-
        # magnitude correction factor averaged over several seeds (see
        # docstring), the correct fix is to simply discard a seed whose
        # simulation misbehaved this way, not to let it silently poison
        # the average -- this is a numerical safety measure on this
        # internal diagnostic simulation only, and does not affect the
        # validated core Heston Monte Carlo engine (models/heston_mc.py)
        # used for actual pricing/VaR, which is tested separately.
        if not np.all(np.isfinite(price_path)) or np.any(price_path <= 0):
            continue

        sim_returns = pd.Series(np.diff(np.log(price_path)))
        raw = leverage_correlation(sim_returns, window=window, trading_days_per_year=trading_days_per_year)
        if np.isfinite(raw):
            ratios.append(raw / reference_rho)

    if len(ratios) == 0:
        return 1.0  # fallback: no correction possible, leave raw estimate as-is
    return float(np.mean(ratios))


def compute_empirical_moments(log_returns, window=21, max_lag_days=60,
                               trading_days_per_year=252):
    """
    Compute the empirical moments used as GMM targets from a historical
    daily log-return series.

    Returns
    -------
    dict with keys: 'theta_emp' (mean RV), 'var_emp' (var of RV),
    'lags_years' (array), 'acf_emp' (array, same length as lags_years),
    'rho_emp_raw' (RAW, attenuation-biased leverage correlation -- see
    estimate_rho_attenuation_factor() for why this needs correcting before
    use), 'v0_emp' (most recent RV), and 'rv' (the full realized-variance
    pd.Series, for plotting).
    """
    rv = realized_variance(log_returns, window=window,
                            trading_days_per_year=trading_days_per_year)
    rv_clean = rv.dropna()

    theta_emp = rv_clean.mean()
    var_emp = rv_clean.var(ddof=1)

    acf_emp = acf(rv_clean, max_lag=max_lag_days)
    lags_years = np.arange(1, max_lag_days + 1) / trading_days_per_year

    rho_emp_raw = leverage_correlation(log_returns, window=window, trading_days_per_year=trading_days_per_year)
    v0_emp = rv_clean.iloc[-1]

    return {
        "theta_emp": theta_emp,
        "var_emp": var_emp,
        "lags_years": lags_years,
        "acf_emp": acf_emp,
        "rho_emp_raw": rho_emp_raw,
        "v0_emp": v0_emp,
        "rv": rv,
    }


def closed_form_initial_guess(moments):
    """
    Fast, deterministic initial parameter guess using closed-form CIR
    moment relationships (before any numerical optimization):

      theta0 = mean(RV)
      kappa0 : fit via linear regression of ln(ACF(lag)) on lag (since
               ln(ACF(lag)) = -kappa*lag for a CIR process), using only
               the lags where the empirical ACF is still positive (log of
               a negative/noisy-near-zero ACF is undefined -- a real
               practical issue with noisy high-lag ACF estimates).
      xi0    : solved from Var(RV) = theta*xi^2/(2*kappa) => xi0 = sqrt(Var(RV)*2*kappa0/theta0)

    This closed-form solution is EXACTLY-identified (3 equations, 3
    unknowns) -- it's not yet using the extra information in the full ACF
    curve, which is what the least-squares refinement step
    (calibrate_heston_gmm) does next.
    """
    theta0 = moments["theta_emp"]
    lags = moments["lags_years"]
    acf_emp = moments["acf_emp"]

    positive_mask = acf_emp > 0.02  # drop near-zero/negative ACF values (log undefined / pure noise)
    if positive_mask.sum() < 3:
        kappa0 = 5.0  # fallback: fast mean reversion if ACF decays to noise almost immediately
    else:
        # ln(ACF) = -kappa * lag  =>  linear regression through the origin... but we allow
        # an intercept too (should be near 0 in theory; a nonzero fitted intercept flags how
        # much finite-sample/estimation noise is present, useful as a diagnostic).
        log_acf = np.log(acf_emp[positive_mask])
        lag_sub = lags[positive_mask]
        slope, intercept = np.polyfit(lag_sub, log_acf, deg=1)
        kappa0 = max(-slope, 0.01)  # kappa must be positive; guard against noisy positive slope

    var_emp = moments["var_emp"]
    xi0 = np.sqrt(max(var_emp * 2 * kappa0 / theta0, 1e-8))

    return kappa0, theta0, xi0


def _gmm_residuals(log_params, moments, feller_penalty_weight=5.0):
    """
    Residual vector for scipy.optimize.least_squares: model-implied minus
    empirical moments, for [theta, var(v), ACF(lag_1), ..., ACF(lag_L)],
    plus a soft Feller-condition penalty term appended as an extra
    "residual" (least_squares minimizes sum of squares of everything
    passed back, so an extra penalty term added here directly shapes the
    objective without needing a separate constrained-optimization API).

    Parameters are optimized in LOG SPACE (log_params = ln(kappa),
    ln(theta), ln(xi)) -- this is a clean, standard reparametrization
    trick that keeps kappa, theta, xi strictly positive automatically
    (exp of anything is positive) without needing explicit bounds, and
    also makes the optimizer's step sizes more natural since these
    parameters are more log-normally than normally distributed in
    practice (small relative changes matter more than absolute ones).
    """
    # ROBUSTNESS GUARD: unconstrained Levenberg-Marquardt (method="lm")
    # can occasionally propose a trial step far enough from zero that
    # np.exp() overflows to inf -- clip to [-30,30] (i.e. params roughly
    # in [1e-13, 1e13], far wider than any sane Heston parameter) so a
    # transient bad step can't corrupt the optimizer with inf/nan. Same
    # guard as calibration/double_heston_calibration.py's residual
    # function, added after this was observed in practice during
    # walk-forward backtesting.
    log_params = np.clip(log_params, -30.0, 30.0)
    kappa, theta, xi = np.exp(log_params)

    theta_resid = (theta - moments["theta_emp"]) / moments["theta_emp"]
    var_model = _model_var(kappa, theta, xi)
    var_resid = (var_model - moments["var_emp"]) / moments["var_emp"]

    acf_model = _model_acf(kappa, moments["lags_years"])
    # Downweight longer lags: their empirical ACF estimates are noisier
    # (fewer effective independent observations contribute to a longer-lag
    # autocovariance estimate) -- inverse-lag weighting is a simple,
    # standard way to reflect that without a full GMM covariance estimate.
    weights = 1.0 / np.sqrt(np.arange(1, len(acf_model) + 1))
    acf_resid = (acf_model - moments["acf_emp"]) * weights

    satisfied, slack = feller_condition(kappa, theta, xi)
    feller_resid = 0.0 if satisfied else feller_penalty_weight * (-slack / max(theta, 1e-6))

    return np.concatenate([[theta_resid, var_resid], acf_resid, [feller_resid]])


def calibrate_heston_gmm(log_returns, window=21, max_lag_days=60,
                          trading_days_per_year=252, n_multistarts=8, seed=0):
    """
    Full Heston calibration pipeline from a historical return series:
      1. Compute empirical moments (realized variance, its ACF, leverage
         correlation).
      2. Get a fast closed-form initial guess for (kappa, theta, xi).
      3. Refine via scipy.optimize.least_squares (Levenberg-Marquardt-style
         trust-region reflective), MULTI-START from several randomly
         perturbed initial guesses around the closed-form one -- since
         this is a nonlinear least-squares problem it can have multiple
         local minima, so we take the best (lowest total squared
         residual) result across starts, not just the first one found.
      4. rho and v0 are set directly from their closed-form estimates
         (see analysis/realized_variance.leverage_correlation) -- they
         don't need iterative refinement since their moment conditions
         are already exactly-identified (one equation, one unknown).

    JUDGMENT CALL -- multi-start local optimization vs. a global
    optimizer (e.g. differential_evolution): the ORIGINAL option-price-
    based calibration objective (see the project's initial spec) is known
    to be highly multimodal in 5 parameters simultaneously. This
    return-based GMM objective is lower-dimensional (3 free parameters:
    kappa, theta, xi) and built from smooth, analytic moment formulas
    (no simulation noise in the objective), so it is much better behaved;
    multi-start Levenberg-Marquardt is fast (each start converges in
    milliseconds) and sufficient here. `calibrate_heston_gmm_global()`
    below provides a differential_evolution alternative for cases with a
    harder-to-fit ticker (e.g. very short history, unusually noisy ACF).

    Returns
    -------
    dict with keys: 'kappa', 'theta', 'xi', 'rho', 'v0', 'feller_satisfied',
    'feller_slack', 'moments' (the empirical moments dict, for plotting),
    'objective_value' (best sum-of-squared-residuals), 'n_starts_tried'.
    """
    moments = compute_empirical_moments(log_returns, window, max_lag_days, trading_days_per_year)
    kappa0, theta0, xi0 = closed_form_initial_guess(moments)

    rng = np.random.default_rng(seed)
    best_result = None
    best_cost = np.inf

    for i in range(n_multistarts):
        if i == 0:
            start = np.log([kappa0, theta0, xi0])  # always include the closed-form guess itself
        else:
            jitter = rng.uniform(0.5, 1.5, size=3)  # +/-50% multiplicative jitter
            start = np.log([kappa0 * jitter[0], theta0 * jitter[1], xi0 * jitter[2]])

        result = least_squares(_gmm_residuals, start, args=(moments,), method="lm", max_nfev=2000)
        cost = np.sum(result.fun**2)
        if cost < best_cost:
            best_cost = cost
            best_result = result

    kappa_hat, theta_hat, xi_hat = np.exp(best_result.x)

    atten_factor = estimate_rho_attenuation_factor(kappa_hat, theta_hat, xi_hat, window,
                                                     trading_days_per_year)
    rho_hat = float(np.clip(moments["rho_emp_raw"] / atten_factor, -0.999, 0.999))
    v0_hat = moments["v0_emp"]

    satisfied, slack = feller_condition(kappa_hat, theta_hat, xi_hat)

    return {
        "kappa": kappa_hat, "theta": theta_hat, "xi": xi_hat,
        "rho": rho_hat, "v0": v0_hat,
        "rho_raw": moments["rho_emp_raw"], "rho_attenuation_factor": atten_factor,
        "feller_satisfied": satisfied, "feller_slack": slack,
        "moments": moments, "objective_value": best_cost, "n_starts_tried": n_multistarts,
    }


def calibrate_heston_gmm_global(log_returns, window=21, max_lag_days=60,
                                 trading_days_per_year=252, seed=0):
    """
    Alternative calibration using scipy.optimize.differential_evolution
    (a genuine global optimizer: population-based, doesn't need a good
    starting guess) instead of multi-start local optimization. Useful as
    a robustness cross-check against calibrate_heston_gmm() -- if both
    methods agree, that's good evidence the multi-start LM result is a
    true global optimum and not just the best of a locally-clustered set
    of starts.

    Search bounds are wide but economically sensible for equities:
      kappa in [0.05, 15]   (mean-reversion speed, per year)
      theta in [0.0005, 1.0] (annualized variance; sqrt gives ~2%-100% vol)
      xi    in [0.01, 3.0]   (vol-of-vol)
    """
    moments = compute_empirical_moments(log_returns, window, max_lag_days, trading_days_per_year)

    def objective(params):
        kappa, theta, xi = params
        resid = _gmm_residuals(np.log([kappa, theta, xi]), moments)
        return np.sum(resid**2)

    bounds = [(0.05, 15.0), (0.0005, 1.0), (0.01, 3.0)]
    result = differential_evolution(objective, bounds, seed=seed, maxiter=300, tol=1e-10,
                                     polish=True, workers=1)

    kappa_hat, theta_hat, xi_hat = result.x
    atten_factor = estimate_rho_attenuation_factor(kappa_hat, theta_hat, xi_hat, window,
                                                     trading_days_per_year)
    rho_hat = float(np.clip(moments["rho_emp_raw"] / atten_factor, -0.999, 0.999))
    v0_hat = moments["v0_emp"]
    satisfied, slack = feller_condition(kappa_hat, theta_hat, xi_hat)

    return {
        "kappa": kappa_hat, "theta": theta_hat, "xi": xi_hat,
        "rho": rho_hat, "v0": v0_hat,
        "rho_raw": moments["rho_emp_raw"], "rho_attenuation_factor": atten_factor,
        "feller_satisfied": satisfied, "feller_slack": slack,
        "moments": moments, "objective_value": result.fun,
    }
