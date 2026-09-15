"""
analysis/realized_variance.py
================================
Realized variance is our observable PROXY for Heston's latent (unobserved)
instantaneous variance process v(t). We never see v(t) directly in real
data -- all we see is the price series -- so calibrating Heston from
returns alone requires first estimating v(t) some other way. Rolling-
window realized variance (the sample variance of returns over a trailing
window, annualized) is the standard, simplest such estimator, and is what
the calibration in calibration/heston_calibration.py is built on.
"""

import numpy as np
import pandas as pd


def realized_variance(log_returns, window=21, trading_days_per_year=252):
    """
    Rolling-window realized variance: the annualized sample variance of
    daily log returns over a trailing `window`-day window.

        RV_t = Var(r_{t-window+1}, ..., r_t) * trading_days_per_year

    JUDGMENT CALL: window=21 (~1 trading month), same choice as
    analysis/gbm_diagnostics.py's rolling volatility plot, for consistency
    across the report. Shorter windows track true v(t) more responsively
    but are noisier (fewer samples per estimate); longer windows are
    smoother but lag behind genuine regime changes. 21 days is a standard
    choice balancing the two, widely used in realized-vol literature.
    """
    return log_returns.rolling(window=window).var(ddof=1) * trading_days_per_year


def acf(series, max_lag):
    """
    Sample autocorrelation function of a (already-stationary-ish) series,
    at lags 1..max_lag. Uses the standard estimator:

        ACF(k) = Cov(x_t, x_{t+k}) / Var(x_t)

    computed on de-meaned, NaN-dropped data (rolling-window realized
    variance has NaNs at the start where the window isn't full yet).

    Returns
    -------
    np.ndarray, shape (max_lag,), ACF(1) through ACF(max_lag)
    """
    x = series.dropna().values
    x = x - x.mean()
    n = len(x)
    var0 = np.dot(x, x) / n

    acf_vals = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        cov_k = np.dot(x[:-k], x[k:]) / n
        acf_vals[k - 1] = cov_k / var0
    return acf_vals


def leverage_correlation(log_returns, window=21, trading_days_per_year=252):
    """
    Empirical estimate of Heston's rho (correlation between price shocks
    and variance shocks), derived as follows:

    From the Heston SDE, to LEADING ORDER in dt (Ito's isometry / quadratic
    covariation, dropping the drift terms which are O(dt) in the mean but
    contribute negligibly to variance/covariance at O(dt) as dt -> 0):

        Var(dr)      ~ v * dt
        Var(dv)      ~ xi^2 * v * dt
        Cov(dr, dv)  ~ rho * xi * v * dt          (since dr's diffusion
                                                    coefficient is sqrt(v),
                                                    dv's is xi*sqrt(v), and
                                                    their driving Brownian
                                                    motions have correlation rho)

    So:
        Corr(dr, dv) = Cov(dr,dv) / sqrt(Var(dr)*Var(dv)) = rho

    i.e. to leading order in dt, the correlation between the return and
    the CHANGE in variance over the same interval equals rho directly.
    Verified directly against the TRUE (simulated, not proxied) variance
    path in tests/test_heston_calibration.py: corr(r_t, v_{t+1}-v_t) on a
    known Heston simulation with rho=-0.6 comes out to -0.593 -- the
    identity holds essentially exactly, so any error in the ESTIMATE below
    comes only from how we proxy the unobservable v(t), not from this
    formula.

    PROXY CONSTRUCTION -- IMPORTANT SUBTLETY: a naive proxy using
    trailing rolling-window realized variance, dv_t ~= RV_t - RV_{t-1},
    is CONTAMINATED: RV_t (a window ending at t) mechanically includes
    today's own return r_t (via its contribution r_t^2 to the window's
    sample variance), so RV_t - RV_{t-1} partly just reflects "how big was
    r_t" rather than "how did volatility shift because of r_t" -- this
    spurious mechanical link swamped the true leverage signal in initial
    testing (recovered rho had the WRONG SIGN before this fix). The fix:
    compare a BACKWARD window (ending the day BEFORE today, excluding r_t
    entirely) to a FORWARD window (starting the day AFTER today, also
    excluding r_t entirely) -- neither window's variance estimate can be
    mechanically moved by r_t itself, so any correlation with r_t reflects
    a genuine shift in surrounding volatility, not an accounting artifact.

        RV_bwd_t = realized variance over returns [t-window, ..., t-1]
        RV_fwd_t = realized variance over returns [t+1, ..., t+window]
        proxy for dv_t  :=  RV_fwd_t - RV_bwd_t

    Returns
    -------
    float, the estimated rho (in [-1, 1] by construction, since it's a
    Pearson correlation coefficient).
    """
    rv = realized_variance(log_returns, window=window, trading_days_per_year=trading_days_per_year)
    rv_bwd = rv.shift(1)          # variance of the window ending yesterday (excludes r_t)
    rv_fwd = rv.shift(-window)    # variance of the window starting tomorrow (excludes r_t)
    proxy_dv = rv_fwd - rv_bwd

    aligned = pd.concat([log_returns, proxy_dv], axis=1, join="inner").dropna()
    aligned.columns = ["r", "proxy_dv"]

    # ROBUSTNESS GUARD: if either series is (numerically) constant --
    # e.g. proxy_dv is exactly flat because the simulation this ran on
    # used a near-zero vol-of-vol/variance-level parameter combination
    # (this function is also called internally by
    # calibration/heston_calibration.py's estimate_rho_attenuation_factor
    # on SIMULATED data during calibration, which can legitimately explore
    # near-degenerate parameter regions) -- a correlation is mathematically
    # undefined (division by a zero standard deviation). Return NaN
    # directly rather than letting numpy/pandas raise a RuntimeWarning
    # about it internally; NaN is already the correct "not defined here"
    # signal the callers (leverage_correlation's consumers) check for.
    if aligned["r"].std(ddof=0) < 1e-14 or aligned["proxy_dv"].std(ddof=0) < 1e-14:
        return np.nan

    return aligned["r"].corr(aligned["proxy_dv"])
