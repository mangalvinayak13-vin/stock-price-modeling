"""
analysis/backtest.py
======================
Walk-forward out-of-sample backtest of GBM vs Heston over up to 30 years
of real price history, with STRICT NO-LOOKAHEAD discipline: at every
point, a model is calibrated using ONLY data up to that point in time,
then judged on data immediately AFTER that point which it never saw
during calibration. This is the standard methodology real quant
researchers use to judge whether a model is actually useful (an in-sample
fit can look perfect and still be worthless for prediction -- only
out-of-sample performance tells you anything about that).

WALK-FORWARD PROCEDURE:
    |---- calibration window (e.g. 5y) ----|-- test window (e.g. 1y) --|
                                            ^
                                 model calibrated using ONLY data left of
                                 this point; judged on data to the right,
                                 which is invisible to the calibration step.

    Then the whole window slides forward by `step_years` and repeats,
    walking through the full history. Over ~30 years with a 5y
    calibration / 1y test / 1y step setup, this produces ~25 independent
    out-of-sample test periods spanning multiple real market regimes
    (dot-com bust, 2008 GFC, COVID crash, etc.) -- exactly the kind of
    regime diversity a real backtest needs to be credible.

TWO EVALUATION METRICS, BOTH STANDARD QUANT PRACTICE:

  1. VOLATILITY FORECAST ACCURACY: at the start of each test window, ask
     each model "what average volatility do you expect over the next
     `test_years`?", then compare to what volatility ACTUALLY occurred.
       - GBM's forecast is trivial: its constant sigma_hat estimated from
         the calibration window (GBM has no way to expect vol to change).
       - Heston's forecast uses the CLOSED-FORM expected-variance formula
         for a mean-reverting CIR process:
             E[v_t] = theta + (v0 - theta) * exp(-kappa*t)
         Averaging over the test horizon T:
             E[avg variance over [0,T]] = theta + (v0-theta)*(1-exp(-kappa*T))/(kappa*T)
         (this is an exact closed-form result -- the CIR mean solves a
         linear ODE -- not a simulation estimate.) This lets Heston say
         "volatility is currently elevated/depressed relative to normal,
         and I expect it to partially mean-revert by the time the test
         window is over" -- something GBM structurally cannot do.

  2. VALUE-AT-RISK (VaR) BACKTESTING WITH THE KUPIEC COVERAGE TEST: at the
     start of each test window, each model forecasts a 1-day 95% VaR
     (the daily loss threshold that should only be exceeded 5% of the
     time), held fixed through that test window (re-forecast at the next
     window). We then count, across ALL ~25 windows pooled together, how
     often the ACTUAL realized daily return breached that threshold. If a
     model is well-calibrated, breaches should occur at close to the
     stated 5% rate. The Kupiec (1995) unconditional coverage test
     formalizes "close to 5%" into a proper statistical hypothesis test
     (this is literally the test banking regulators require institutions
     to run on their internal VaR models under Basel market-risk rules --
     not a made-up metric for this project).

JUDGMENT CALLS (state these in the report):
  - calib_years=5, test_years=1, step_years=1: a common industry
    convention (enough calibration data for the CIR moment estimators to
    be reasonably stable, per the earlier finding that very long
    calibration windows violate stationarity; test windows short enough
    to give ~25 independent regimes over 30 years for statistical power).
  - VaR forecasts are held FIXED for the whole test window rather than
    updated daily -- this matches how periodic (not continuous)
    recalibration is actually used in practice, and keeps the backtest
    computationally tractable (no need to refit at every single day).
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from models.gbm import estimate_gbm_params
from models.heston_mc import simulate_heston_paths
from models.double_heston_mc import simulate_double_heston_paths
from calibration.heston_calibration import calibrate_heston_gmm
from calibration.double_heston_calibration import calibrate_double_heston_gmm


def heston_expected_avg_variance(v0, theta, kappa, T):
    """
    Closed-form expected AVERAGE variance over [0, T] under the CIR
    process dv = kappa(theta-v)dt + xi*sqrt(v)dW:

        E[v_t] = theta + (v0 - theta) * exp(-kappa*t)     (solves the ODE
                  obtained by taking expectations of the SDE -- the
                  stochastic term has zero mean, so E[v_t] follows the
                  deterministic mean-reversion ODE exactly, for ANY xi)

        E[avg variance over [0,T]] = (1/T) * Integral_0^T E[v_t] dt
                                    = theta + (v0-theta) * (1 - exp(-kappa*T)) / (kappa*T)

    Note this does NOT depend on xi or rho at all -- it's a statement
    purely about the mean path of variance, which only involves
    (kappa, theta, v0). This is an exact result, not an approximation.
    """
    if kappa * T < 1e-8:
        return v0  # degenerate limit: negligible mean reversion over a near-zero horizon
    return theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T)


def kupiec_test(n_breaches, n_trials, p=0.05):
    """
    Kupiec (1995) unconditional coverage likelihood-ratio test: tests
    whether the observed VaR breach rate (n_breaches / n_trials) is
    statistically consistent with the model's stated confidence level p.

    Null hypothesis: true breach probability equals p (the model is
    correctly calibrated). Test statistic:

        LR_uc = -2 * ln[ (1-p)^(n-x) * p^x / ( (1-x/n)^(n-x) * (x/n)^x ) ]

    which is asymptotically chi-squared with 1 degree of freedom under
    the null. This is the exact test used in banking regulation (Basel
    market-risk framework) to validate internal VaR models -- using it
    here is a direct, standard, defensible way to grade whether the
    model's risk forecasts are honest, not just its point forecasts.

    Returns
    -------
    (LR_statistic, p_value) -- a LOW p-value (e.g. < 0.05) means REJECT
    the model as miscalibrated (too many or too few breaches to be
    consistent with the stated confidence level).
    """
    x, n = n_breaches, n_trials
    if x == 0 or x == n:
        # Log-likelihood degenerates at the boundary; use a tiny epsilon
        # to avoid log(0) -- standard practical fix, doesn't change the
        # qualitative conclusion (0 or 100% breach rate is obviously far
        # from p=5% for any reasonable n).
        x_frac = np.clip(x / n, 1e-6, 1 - 1e-6)
    else:
        x_frac = x / n

    log_L_null = (n - x) * np.log(1 - p) + x * np.log(p)
    log_L_alt = (n - x) * np.log(1 - x_frac) + x * np.log(x_frac)
    LR = -2 * (log_L_null - log_L_alt)
    p_value = chi2.sf(LR, df=1)
    return LR, p_value


def _make_windows(n_obs, calib_days, test_days, step_days):
    """Generate (calib_start, calib_end, test_end) index triples walking
    forward through the data. calib_end == test_start (strict boundary,
    no overlap, no lookahead: test data starts exactly where calibration
    data ends)."""
    windows = []
    calib_start = 0
    while True:
        calib_end = calib_start + calib_days
        test_end = calib_end + test_days
        if test_end > n_obs:
            break
        windows.append((calib_start, calib_end, test_end))
        calib_start += step_days
    return windows


def run_walk_forward_backtest(log_returns, calib_years=5, test_years=1, step_years=1,
                               trading_days_per_year=252, var_confidence=0.95,
                               heston_n_multistarts=6, double_heston_n_multistarts=8,
                               mc_paths_for_var=20_000, include_double_heston=True, seed=0):
    """
    Run the full walk-forward backtest described in the module docstring,
    now comparing THREE models: GBM, single Heston, and (optionally, since
    it's the most expensive to calibrate per window) Double Heston.

    Double Heston's forecast vol and VaR are computed exactly the same way
    as single Heston's, just summing the two independent factors' closed-
    form contributions:
        E[avg total variance] = E[avg v1] + E[avg v2]
                               = heston_expected_avg_variance(v1_0,theta1,kappa1,T)
                                 + heston_expected_avg_variance(v2_0,theta2,kappa2,T)
    (valid because v1, v2 are independent CIR processes, so their expected
    paths simply add -- see models/double_heston.py's module docstring).
    Its 1-day VaR uses models/double_heston_mc.py's two-factor Monte Carlo
    simulator, warm-started each window from that window's own single-
    Heston fit (per the calibration module's design), preserving the
    no-lookahead discipline exactly as for the other two models.

    Parameters
    ----------
    log_returns : pd.Series of daily log returns, DatetimeIndex, in
        chronological order (this function relies on order, not on the
        index values themselves, but a DatetimeIndex makes the output
        directly plottable against real dates).
    include_double_heston : bool, default True. Double Heston calibration
        (single-Heston warm start + 6-parameter multi-start refinement) is
        meaningfully slower per window than single Heston -- set False to
        skip it for a quick GBM-vs-Heston-only run.

    Returns
    -------
    pd.DataFrame, one row per walk-forward window, with columns for each
    model's forecast_vol, var, breaches, plus calibrated parameters
    (prefixed heston_* and dh_*), and realized_vol/dates/n_test_days.
    Plus a second returned dict with pooled Kupiec test results for every
    model included.
    """
    calib_days = int(calib_years * trading_days_per_year)
    test_days = int(test_years * trading_days_per_year)
    step_days = int(step_years * trading_days_per_year)

    values = log_returns.values
    dates = log_returns.index
    n_obs = len(values)

    windows = _make_windows(n_obs, calib_days, test_days, step_days)
    if len(windows) == 0:
        raise ValueError(
            f"Not enough data for even one walk-forward window: need at least "
            f"{calib_days + test_days} observations, got {n_obs}. Use a longer "
            f"history or shorter calib_years/test_years."
        )

    z_var = norm.ppf(1 - var_confidence)  # e.g. -1.645 for 95% VaR
    rng = np.random.default_rng(seed)

    rows = []
    for calib_start, calib_end, test_end in windows:
        calib_returns = pd.Series(values[calib_start:calib_end], index=dates[calib_start:calib_end])
        test_returns = values[calib_end:test_end]
        T_test = len(test_returns) / trading_days_per_year

        # --- GBM: calibrate on the window, forecast forward ---
        mu_hat, sigma_hat = estimate_gbm_params(calib_returns, trading_days_per_year)
        gbm_forecast_vol = sigma_hat
        mu_daily = (mu_hat - 0.5 * sigma_hat**2) / trading_days_per_year
        sigma_daily = sigma_hat / np.sqrt(trading_days_per_year)
        gbm_var = mu_daily + z_var * sigma_daily

        # --- Heston: calibrate on the window (GMM, return-based, no lookahead) ---
        heston_result = calibrate_heston_gmm(calib_returns, n_multistarts=heston_n_multistarts,
                                              seed=int(rng.integers(0, 1_000_000)))
        heston_forecast_vol = np.sqrt(heston_expected_avg_variance(
            heston_result["v0"], heston_result["theta"], heston_result["kappa"], T_test))

        # 1-day-ahead VaR via Monte Carlo (our validated Full Truncation
        # Euler simulator), using the calibrated params and v0 as the
        # starting variance. mu_hat is used as the physical-measure drift
        # (r slot of the simulator repurposed as the real-world drift
        # here, since we're forecasting REAL outcomes, not risk-neutral
        # prices -- q=0 since dividends don't separately apply to a pure
        # return simulation).
        _, S_1day, _ = simulate_heston_paths(
            S0=1.0, v0=heston_result["v0"], T=1 / trading_days_per_year, n_steps=1,
            n_paths=mc_paths_for_var, kappa=heston_result["kappa"], theta=heston_result["theta"],
            xi=heston_result["xi"], rho=heston_result["rho"], r=mu_hat, q=0.0,
            seed=int(rng.integers(0, 1_000_000)), antithetic=True,
        )
        sim_1day_log_returns = np.log(S_1day[:, -1])
        heston_var = np.quantile(sim_1day_log_returns, 1 - var_confidence)

        gbm_breaches = int(np.sum(test_returns < gbm_var))
        heston_breaches = int(np.sum(test_returns < heston_var))
        realized_vol = float(np.std(test_returns, ddof=1) * np.sqrt(trading_days_per_year))

        row = {
            "test_start_date": dates[calib_end],
            "test_end_date": dates[test_end - 1],
            "n_test_days": len(test_returns),
            "realized_vol": realized_vol,
            "gbm_forecast_vol": gbm_forecast_vol,
            "heston_forecast_vol": heston_forecast_vol,
            "gbm_var": gbm_var,
            "heston_var": heston_var,
            "gbm_breaches": gbm_breaches,
            "heston_breaches": heston_breaches,
            "heston_kappa": heston_result["kappa"],
            "heston_theta": heston_result["theta"],
            "heston_xi": heston_result["xi"],
            "heston_rho": heston_result["rho"],
            "heston_v0": heston_result["v0"],
            "heston_feller_satisfied": heston_result["feller_satisfied"],
        }

        # --- Double Heston: warm-started from this window's own single-
        # Heston fit (no lookahead: both are calibrated on the exact same
        # calib_returns, nothing from the test window is used) ---
        if include_double_heston:
            dh_result = calibrate_double_heston_gmm(
                calib_returns, n_multistarts=double_heston_n_multistarts,
                seed=int(rng.integers(0, 1_000_000)), single_heston_result=heston_result)

            dh_forecast_vol = np.sqrt(
                heston_expected_avg_variance(dh_result["v1_0"], dh_result["theta1"], dh_result["kappa1"], T_test)
                + heston_expected_avg_variance(dh_result["v2_0"], dh_result["theta2"], dh_result["kappa2"], T_test)
            )

            _, S_1day_dh, _, _ = simulate_double_heston_paths(
                S0=1.0, v1_0=dh_result["v1_0"], v2_0=dh_result["v2_0"],
                T=1 / trading_days_per_year, n_steps=1, n_paths=mc_paths_for_var,
                kappa1=dh_result["kappa1"], theta1=dh_result["theta1"], xi1=dh_result["xi1"], rho1=dh_result["rho1"],
                kappa2=dh_result["kappa2"], theta2=dh_result["theta2"], xi2=dh_result["xi2"], rho2=dh_result["rho2"],
                r=mu_hat, q=0.0, seed=int(rng.integers(0, 1_000_000)), antithetic=True,
            )
            sim_1day_log_returns_dh = np.log(S_1day_dh[:, -1])
            dh_var = np.quantile(sim_1day_log_returns_dh, 1 - var_confidence)
            dh_breaches = int(np.sum(test_returns < dh_var))

            row.update({
                "dh_forecast_vol": dh_forecast_vol,
                "dh_var": dh_var,
                "dh_breaches": dh_breaches,
                "dh_kappa1": dh_result["kappa1"], "dh_theta1": dh_result["theta1"], "dh_xi1": dh_result["xi1"],
                "dh_kappa2": dh_result["kappa2"], "dh_theta2": dh_result["theta2"], "dh_xi2": dh_result["xi2"],
                "dh_rho": dh_result["rho1"], "dh_feller_satisfied": dh_result["feller_satisfied"],
            })

        rows.append(row)

    results_df = pd.DataFrame(rows)

    n_trials_total = results_df["n_test_days"].sum()
    model_breach_cols = {"gbm": "gbm_breaches", "heston": "heston_breaches"}
    if include_double_heston:
        model_breach_cols["double_heston"] = "dh_breaches"

    kupiec_summary = {
        "n_windows": len(windows),
        "n_trials_total": int(n_trials_total),
        "target_breach_rate": 1 - var_confidence,
    }
    for model_name, col in model_breach_cols.items():
        total_breaches = int(results_df[col].sum())
        LR, p_value = kupiec_test(total_breaches, n_trials_total, p=1 - var_confidence)
        kupiec_summary[model_name] = {
            "breaches": total_breaches,
            "breach_rate": total_breaches / n_trials_total,
            "LR_statistic": LR, "p_value": p_value,
            "rejected_at_5pct": p_value < 0.05,
        }

    return results_df, kupiec_summary
