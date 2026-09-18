"""
trading/vrp_backtest.py
=========================
33-year walk-forward backtest of systematically selling SPY straddles,
answering the central question of options trading: does the volatility
risk premium (options priced above the volatility that actually
materializes) exist, persist, and pay after costs -- and does a
Heston-based filter improve on blindly selling every month?

WHY THIS AVOIDS THE PROJECT'S CORE DATA LIMITATION (read this first):
The rest of this project (see README) explains that free historical
option-chain data does not exist -- yfinance only ever returns TODAY's
live chain (see data/loader.get_option_chain's docstring). That rules out
backtesting a strategy that trades real, individually-quoted listed
contracts across 33 years.

This module sidesteps that limitation with a standard, citable technique
from the volatility-risk-premium literature (Carr & Wu 2009, "Variance
Risk Premia", J. Financial Economics; the CBOE's own PUT and BXM
benchmark indices are built the same way): use the CBOE VIX INDEX -- which
IS free and has a genuine daily history back to 1990 (data/loader.
get_vix_history), and IS itself computed by the CBOE each day directly
from that day's REAL, live SPX option chain -- as the market's implied
volatility input to the Black-Scholes formula, in place of a strike-by-
strike chain we cannot obtain historically. This is not a workaround that
invents data: VIX_t IS real, contemporaneous option-market pricing
information for every single day since 1990, packaged as one number
instead of a full chain.

WHAT THIS APPROXIMATES: a monthly-rolled, AT-THE-MONEY, UNHEDGED short
straddle, struck exactly at spot (not the nearest real discrete strike)
and priced with a single flat vol (VIX) rather than the real chain's
strike-dependent skew. It is a benchmark-index-style overlay (exactly
what CBOE's own published benchmarks do), not a claim about what a real
listed-option fill would achieve. Every limitation is listed again in the
README section this module supports.

STRATEGY:
  Each ~1 calendar month (holding_days trading days), at the close:
    1. Observe VIX_t (real, that day's CBOE-published index level) and
       the Heston model's own physical-measure forecast of average
       volatility over the NEXT holding_days (closed-form, from a Heston
       fit calibrated on strictly PAST returns -- see calib window logic
       below).
    2. Volatility risk premium signal: vrp_t = VIX_t/100 - heston_forecast_vol.
       If vrp_t > vrp_threshold, SELL an ATM straddle (collect VIX-implied
       premium); otherwise stay in cash for the month.
    3. Hold to expiry (no interim delta hedging -- see module docstring
       above), settle P&L = premium_received - |S_{t+holding_days} - K|.

NO-LOOKAHEAD DISCIPLINE: Heston's (kappa, theta, xi) are re-calibrated
only once a year, on a trailing `calib_years`-year window of returns
STRICTLY BEFORE that year begins (identical walk-forward machinery to
analysis/backtest.py's _make_windows) -- never on data from the year being
traded. Within a traded year, v0 (today's instantaneous variance -- an
observable STATE, not a fitted structural constant, per models/heston.py's
own definition) is refreshed every month from the trailing `holding_days`-
day realized variance, which only ever uses data up to and including the
entry date.
"""

import numpy as np
import pandas as pd

from models.black_scholes import bs_price
from calibration.heston_calibration import calibrate_heston_gmm
from analysis.backtest import heston_expected_avg_variance, _make_windows


def _trailing_realized_var(log_returns_values, end_idx, window, trading_days_per_year=252):
    """Annualized realized variance of the `window` daily log returns
    ending at (and excluding, i.e. strictly before) index `end_idx`."""
    start_idx = max(0, end_idx - window)
    seg = log_returns_values[start_idx:end_idx]
    if len(seg) < 2:
        return np.nan
    return float(np.var(seg, ddof=1) * trading_days_per_year)


def run_vrp_backtest(prices, vix, r=0.04, q=0.0, calib_years=5, holding_days=21,
                      trading_days_per_year=252, transaction_cost_vol=0.01,
                      vrp_threshold=0.0, heston_n_multistarts=6, seed=0):
    """
    Parameters
    ----------
    prices : pd.Series of SPY (or any single-name/index) daily close,
        DatetimeIndex, chronological.
    vix : pd.Series of VIX close (in percent, e.g. 18.5), same date
        convention; aligned to `prices`' dates via reindex+ffill (VIX and
        SPY trade on the same US market calendar so this is almost always
        an exact-date match; ffill covers rare one-off data gaps in
        either feed).
    r, q : flat risk-free rate / dividend yield (same flat-rate
        simplification used everywhere else in this project -- see
        data/loader.get_risk_free_rate's docstring).
    transaction_cost_vol : round-trip cost, in ANNUALIZED VOLATILITY
        POINTS, subtracted from the entry vol before pricing BOTH legs of
        the straddle (i.e. filled as if the market's vol were
        `transaction_cost_vol` lower than the VIX quote) -- a standard way
        to haircut a benchmark-index backtest for the real bid/ask an
        at-the-money SPX straddle trades at (SPX options are among the
        most liquid in the world; 1 vol point, the default, is a
        conservative-but-not-punitive estimate, not a tight institutional
        fill).
    vrp_threshold : minimum vrp_t (VIX vol minus Heston-forecast vol) to
        trade that month; otherwise stay in cash. Pass -999 to always
        trade (the "always-sell" baseline used for comparison in the
        README/report).

    Returns
    -------
    pd.DataFrame, one row per traded-or-skipped month, with
    `.attrs['summary']` set to the headline performance-metric dict (kept
    attached to the DataFrame so callers can't separate one from the
    other by accident).
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    vix_aligned = vix.reindex(log_returns.index).ffill() / 100.0  # -> annualized vol fraction
    values = log_returns.values
    dates = log_returns.index
    n_obs = len(values)

    calib_days = int(calib_years * trading_days_per_year)
    windows = _make_windows(n_obs, calib_days, trading_days_per_year, trading_days_per_year)
    if len(windows) == 0:
        raise ValueError(f"Not enough history for even one annual walk-forward window "
                          f"(need >= {calib_days + trading_days_per_year} return obs, got {n_obs}).")

    rng = np.random.default_rng(seed)
    rows = []

    for calib_start, calib_end, test_end in windows:
        calib_returns = pd.Series(values[calib_start:calib_end], index=dates[calib_start:calib_end])
        heston_fit = calibrate_heston_gmm(calib_returns, n_multistarts=heston_n_multistarts,
                                           seed=int(rng.integers(0, 1_000_000)))
        kappa, theta_v = heston_fit["kappa"], heston_fit["theta"]

        month_start = calib_end
        while month_start + holding_days <= test_end:
            month_end = month_start + holding_days
            entry_date, exit_date = dates[month_start], dates[month_end - 1]

            S0 = float(prices.loc[entry_date])
            S_exit = float(prices.loc[exit_date])
            K = S0  # ATM, struck exactly at spot (see module docstring)
            T = holding_days / trading_days_per_year

            v0_t = _trailing_realized_var(values, month_start, holding_days, trading_days_per_year)
            heston_forecast_vol = float(np.sqrt(
                heston_expected_avg_variance(v0_t, theta_v, kappa, T))) if np.isfinite(v0_t) else np.nan
            vix_vol = float(vix_aligned.loc[entry_date])
            vrp = vix_vol - heston_forecast_vol if np.isfinite(heston_forecast_vol) else np.nan

            trade = bool(np.isfinite(vrp) and vrp > vrp_threshold)

            if trade:
                sigma_fill = max(vix_vol - transaction_cost_vol / 2, 1e-4)
                premium = (bs_price(S0, K, T, r, sigma_fill, q, "call")
                           + bs_price(S0, K, T, r, sigma_fill, q, "put"))
                payoff = abs(S_exit - K)
                pnl = premium - payoff
                ret = pnl / S0
            else:
                # Flat: idle cash earns the flat risk-free rate over the holding
                # period, for a fair comparison against staying active (not a
                # zero floor, which would silently favor the "always trade"
                # variant by starving the cash alternative of any return).
                ret, pnl, premium, payoff, sigma_fill = r * T, np.nan, np.nan, np.nan, np.nan

            rows.append({
                "entry_date": entry_date, "exit_date": exit_date, "S0": S0, "S_exit": S_exit,
                "K": K, "vix_vol": vix_vol, "heston_forecast_vol": heston_forecast_vol,
                "vrp": vrp, "traded": trade, "sigma_fill": sigma_fill, "premium": premium,
                "payoff": payoff, "pnl": pnl, "ret": ret,
                "heston_kappa": kappa, "heston_theta": theta_v,
            })

            month_start = month_end

    df = pd.DataFrame(rows)
    df.attrs["summary"] = _performance_summary(df, trading_days_per_year, holding_days)
    return df


def _performance_summary(df, trading_days_per_year, holding_days):
    """CAGR, annualized vol, Sharpe, max drawdown, win rate, active-month
    fraction -- from the monthly return series `df['ret']`."""
    periods_per_year = trading_days_per_year / holding_days
    rets = df["ret"].values
    n = len(rets)
    if n == 0:
        return {"n_periods": 0}

    equity = np.cumprod(1 + rets)
    total_return = float(equity[-1] - 1)
    n_years = n / periods_per_year
    cagr = float(equity[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan

    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(periods_per_year)) if n > 1 else np.nan
    ann_mean = float(np.mean(rets) * periods_per_year)
    sharpe = ann_mean / ann_vol if ann_vol and ann_vol > 0 else np.nan

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1
    max_drawdown = float(drawdown.min())

    active = df[df["traded"]]
    win_rate = float((active["pnl"] > 0).mean()) if len(active) else np.nan

    return {
        "n_periods": n, "n_active_periods": int(df["traded"].sum()),
        "active_fraction": float(df["traded"].mean()),
        "total_return": total_return, "cagr": cagr, "annualized_vol": ann_vol,
        "sharpe": sharpe, "max_drawdown": max_drawdown, "win_rate": win_rate,
        "equity_curve": equity,
    }


def buy_and_hold_summary(prices, trading_days_per_year=252):
    """Same performance-metric set as _performance_summary(), computed on
    a simple buy-and-hold of `prices` over its full span, as the
    benchmark comparison for run_vrp_backtest()'s equity curve."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    equity = np.exp(log_returns.cumsum())
    total_return = float(equity.iloc[-1] - 1)
    n_years = len(log_returns) / trading_days_per_year
    cagr = float(equity.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan
    ann_vol = float(log_returns.std(ddof=1) * np.sqrt(trading_days_per_year))
    ann_mean = float(log_returns.mean() * trading_days_per_year)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan
    running_max = equity.cummax()
    max_drawdown = float((equity / running_max - 1).min())
    return {"total_return": total_return, "cagr": cagr, "annualized_vol": ann_vol,
            "sharpe": sharpe, "max_drawdown": max_drawdown, "equity_curve": equity}
