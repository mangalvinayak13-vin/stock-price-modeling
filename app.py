"""
app.py
=======
Streamlit dashboard: the deployable "front end" for this project. Ties
together every module built so far -- real-time quotes, historical data,
GBM diagnostics, Heston/Double Heston calibration, and the 30-year
walk-forward backtest -- into one interactive page.

RUN LOCALLY:
    streamlit run app.py

DEPLOY (free): push this repo to GitHub, then deploy at
https://share.streamlit.io (Streamlit Community Cloud) pointing at this
file -- see README.md for full steps.

DESIGN NOTE: expensive computations (fetching 30 years of price history,
running the walk-forward backtest, which calibrates Heston/Double Heston
across ~24 rolling windows) are wrapped in @st.cache_data so that
switching between tabs, or re-running with the SAME ticker, doesn't
recompute from scratch every time Streamlit reruns the script (which it
does on every user interaction -- this is how Streamlit apps work, so
caching is not optional here, it's the difference between an app that
responds in milliseconds vs one that takes 20+ seconds on every click).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from analysis.plot_style import (new_dark_fig, apply_dark_style, COLOR_ACTUAL, COLOR_GBM,
                                  COLOR_HESTON, COLOR_DOUBLE_HESTON, COLOR_BLACK_SCHOLES,
                                  TEXT_SECONDARY, CAT)

from data.loader import (get_price_history, compute_log_returns, get_risk_free_rate,
                          get_dividend_yield, get_vix_history, get_option_chain, clean_option_chain)
from data.realtime import get_live_quote
from models.gbm import simulate_gbm_paths, estimate_gbm_params
from models.black_scholes import bs_price
from models.heston import heston_price
from models.double_heston import double_heston_price
from analysis.gbm_diagnostics import gbm_diagnostic_panel
from analysis.realized_variance import realized_variance, acf
from calibration.heston_calibration import calibrate_heston_gmm, _model_acf
from calibration.double_heston_calibration import calibrate_double_heston_gmm, _double_model_acf
from analysis.backtest import run_walk_forward_backtest
from analysis.model_comparison import summary_table, plot_rmse_bar_chart, plot_var_breach_comparison
from trading.strategies import (straddle, strangle, vertical_spread, iron_condor, bs_pricer,
                                 heston_pricer, bs_greeks_fn, heston_greeks_fn, strategy_summary,
                                 strategy_greeks)
from trading.scanner import scan_chain_vs_heston, top_mispricings
from trading.vrp_backtest import run_vrp_backtest, buy_and_hold_summary


st.set_page_config(page_title="Stock Price Modeling: GBM -> Heston -> Double Heston",
                    layout="wide")


# ---------------------------------------------------------------------------
# Cached data/computation layers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Fetching price history...")
def cached_price_history(ticker, years):
    return get_price_history(ticker, years=years)


@st.cache_data(ttl=3600)
def cached_risk_free_and_dividend(ticker):
    return get_risk_free_rate(), get_dividend_yield(ticker)


@st.cache_data(ttl=60, show_spinner=False)
def cached_live_quote(ticker):
    return get_live_quote(ticker)


@st.cache_data(ttl=3600, show_spinner="Calibrating Heston (return-based GMM)...")
def cached_heston_calibration(ticker, years):
    prices = get_price_history(ticker, years=years)
    log_returns = compute_log_returns(prices)
    return calibrate_heston_gmm(log_returns, n_multistarts=8, seed=1)


@st.cache_data(ttl=3600, show_spinner="Calibrating Double Heston (warm-started from single Heston)...")
def cached_double_heston_calibration(ticker, years, _single_result):
    prices = get_price_history(ticker, years=years)
    log_returns = compute_log_returns(prices)
    return calibrate_double_heston_gmm(log_returns, n_multistarts=10, seed=1,
                                        single_heston_result=_single_result)


@st.cache_data(ttl=3600, show_spinner="Running 30-year walk-forward backtest (calibrates ~24 rolling windows, takes ~20-30s)...")
def cached_backtest(ticker, years, calib_years, test_years, step_years):
    prices = get_price_history(ticker, years=years)
    log_returns = compute_log_returns(prices)
    return run_walk_forward_backtest(log_returns, calib_years=calib_years, test_years=test_years,
                                      step_years=step_years, seed=1)


@st.cache_data(ttl=1800, show_spinner="Fetching and cleaning today's live option chain...")
def cached_clean_chain(ticker, max_expiries):
    raw = get_option_chain(ticker, max_expiries=max_expiries)
    cleaned = clean_option_chain(raw, spot=raw["spot"], verbose=False)
    return raw["spot"], cleaned


@st.cache_data(ttl=3600, show_spinner="Walking forward through SPY/VIX, recalibrating Heston once per year of history...")
def cached_vrp_backtest(years, calib_years, holding_days, cost_vol, threshold):
    prices = get_price_history("SPY", years=years)["Close"]
    vix = get_vix_history(years=years)
    r = get_risk_free_rate()
    return run_vrp_backtest(prices, vix, r=r, calib_years=calib_years, holding_days=holding_days,
                             transaction_cost_vol=cost_vol, vrp_threshold=threshold,
                             heston_n_multistarts=6, seed=1)


@st.cache_data(ttl=3600)
def cached_buy_hold_spy(years):
    prices = get_price_history("SPY", years=years)["Close"]
    return buy_and_hold_summary(prices)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Configuration")

# Curated list of well-known, liquid tickers spanning sectors + major index
# ETFs -- click the dropdown and either pick one or type to search/filter it
# (Streamlit's selectbox supports type-ahead filtering natively). "Other"
# reveals a free-text box for any ticker not in this list, so the dropdown
# never actually limits what you can analyze, it just removes typing for
# the common case.
POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC",
    "JPM", "BAC", "GS", "V", "MA",
    "JNJ", "PFE", "UNH",
    "WMT", "KO", "PG", "MCD", "NKE", "DIS",
    "XOM", "CVX", "BA",
    "SPY", "QQQ", "DIA", "IWM", "VTI",
    "Other (type your own)",
]
ticker_choice = st.sidebar.selectbox("Ticker", options=POPULAR_TICKERS, index=0)
if ticker_choice == "Other (type your own)":
    ticker = st.sidebar.text_input("Enter ticker symbol", value="").strip().upper()
else:
    ticker = ticker_choice

history_years = st.sidebar.slider("Years of history", min_value=2, max_value=30, value=10)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Note on data sources:** historical prices via yfinance; real-time "
    "quote via Finnhub (needs a free `FINNHUB_API_KEY` in a `.env` file -- "
    "see README). All model calibration uses equity RETURNS only, no "
    "options data (see README for why)."
)

st.title("Stock Price Modeling: Brownian Motion -> Black-Scholes -> Heston -> Double Heston")
st.caption(f"Ticker: **{ticker}** | History: **{history_years} years**")

with st.container(border=True):
    st.markdown(
        "**What this app actually does, in plain terms:** every stock price model needs an "
        "assumption about how \"jumpy\" (volatile) the stock is. The simplest models assume "
        "that jumpiness never changes -- which is convenient but wrong, since real markets go "
        "through calm and panicky periods. This app builds up four models of increasing "
        "realism for that jumpiness, fits each one to this stock's real price history, and "
        "then tests -- honestly, on 30 years of data, without cheating by peeking ahead -- "
        "which one actually predicts risk better."
    )
    st.markdown(
        "1. **GBM** -- the baseline: constant volatility (visibly wrong, see the *GBM "
        "Diagnostics* tab)\n"
        "2. **Black-Scholes** -- the classic option-pricing formula built on top of GBM\n"
        "3. **Heston** -- lets volatility itself randomly rise and fall, pulled back toward "
        "a \"normal\" level over time\n"
        "4. **Double Heston** -- tests whether volatility actually has TWO different speeds "
        "of swinging (a fast one and a slow one) rather than just one\n\n"
        "**Where to look:** *Live Price* is just the raw data. *GBM Diagnostics* shows why "
        "the simplest model fails. *Calibration* shows the fitted parameters for Heston/"
        "Double Heston. **30-Year Backtest is the actual result** -- which model wins, tested "
        "honestly. *Theoretical Option Pricing* is a bonus demo, not a trading tool."
    )

if not ticker:
    st.stop()

try:
    prices = cached_price_history(ticker, history_years)
except Exception as e:
    st.error(f"Could not fetch price history for '{ticker}': {e}")
    st.stop()

log_returns = compute_log_returns(prices)
r_rate, div_yield = cached_risk_free_and_dividend(ticker)


(tab_results, tab_live, tab_gbm, tab_calib, tab_backtest, tab_pricing,
 tab_strategy, tab_scanner, tab_vrp) = st.tabs(
    ["Results (Start Here)", "Live Price", "GBM Diagnostics", "Heston / Double Heston Calibration",
     "Detailed Backtest (Technical)", "Theoretical Option Pricing",
     "Options Strategy Builder", "Live Mispricing Scanner", "Volatility Risk Premium Backtest"]
)


# ---------------------------------------------------------------------------
# Tab 0: Results -- the plain-language bottom line, loads automatically
# ---------------------------------------------------------------------------

with tab_results:
    st.subheader(f"Bottom Line: Does a Fancier Model Actually Predict {ticker}'s Risk Better?")
    st.caption(
        "This is the actual answer the whole project is trying to give, computed fresh for "
        "whichever ticker and history length you picked in the sidebar. Every number below is "
        "from a strict, no-lookahead test on real historical data: each model only ever sees "
        "data BEFORE the period it's being judged on -- no peeking ahead."
    )

    try:
        with st.spinner(f"Testing GBM, Heston and Double Heston on {history_years} years of "
                         f"{ticker} data, rolling forward 5-year windows at a time "
                         f"(~20-30s the first time; instant after that)..."):
            results_df, kupiec = cached_backtest(ticker, history_years, 5, 1, 1)

        table = summary_table(results_df, kupiec)
        best_row = table.loc[table["vol_RMSE_overall"].idxmin()]
        best_model = best_row["model"]
        passing_models = table[~table["kupiec_rejected"]]["model"].tolist()

        col1, col2, col3 = st.columns(3)
        col1.metric("Best volatility forecaster", best_model)
        col2.metric("Out-of-sample test periods", len(results_df))
        col3.metric("Models that pass the risk-check", f"{len(passing_models)} of {len(table)}")

        st.markdown("#### In plain English")
        if best_model == "GBM":
            st.markdown(
                f"**The simplest model (constant volatility) actually predicted {ticker}'s "
                f"future volatility best** among the three tested -- the fancier Heston and "
                f"Double Heston models didn't pay off here. This is a genuine, honest finding, "
                f"not a bug: extra model complexity doesn't automatically win, and testing that "
                f"honestly (rather than assuming a fancy model is always better) is the whole "
                f"point of this project."
            )
        else:
            st.markdown(
                f"**{best_model} predicted {ticker}'s future volatility most accurately** "
                f"among the three models tested, beating the simpler alternatives."
            )

        if len(passing_models) == 0:
            st.markdown(
                "**None of the three models correctly forecast tail risk** across the full "
                "test period (checked with the same formal statistical test bank regulators "
                "use to validate risk models). This is a known, real-world phenomenon: models "
                "like these, which assume shocks follow a smooth random pattern, tend to break "
                "down during genuine crises (2008, COVID) where losses are more extreme than "
                "the model expects."
            )
        else:
            st.markdown(f"**{', '.join(passing_models)} passed** the formal tail-risk check; "
                        f"the others didn't.")

        st.markdown("#### The picture")
        fig, ax = new_dark_fig(figsize=(11, 4.5))
        ax.plot(results_df["test_start_date"], results_df["realized_vol"], "-o", color=COLOR_ACTUAL,
                label="What actually happened", lw=2.5, ms=5)
        ax.plot(results_df["test_start_date"], results_df["gbm_forecast_vol"], "--s", color=COLOR_GBM,
                label="GBM predicted", alpha=0.9)
        ax.plot(results_df["test_start_date"], results_df["heston_forecast_vol"], "--^", color=COLOR_HESTON,
                label="Heston predicted", alpha=0.9)
        if "dh_forecast_vol" in results_df.columns:
            ax.plot(results_df["test_start_date"], results_df["dh_forecast_vol"], "--d", color=COLOR_DOUBLE_HESTON,
                    label="Double Heston predicted", alpha=0.9)
        ax.set_ylabel("Annualized volatility")
        ax.set_title(f"{ticker}: what each model predicted vs. what actually happened, year by year")
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)
        plt.close(fig)

        st.info("Want the full numbers behind this (RMSE tables, VaR breach rates, "
                "statistical p-values)? See the **Detailed Backtest (Technical)** tab.")
    except ValueError as e:
        st.error(f"Not enough history to run the backtest: {e}. Try increasing "
                 "'Years of history' in the sidebar.")


# ---------------------------------------------------------------------------
# Tab 1: Live price
# ---------------------------------------------------------------------------

with tab_live:
    st.subheader("Real-Time Quote")
    col1, col2 = st.columns([1, 2])

    with col1:
        try:
            quote = cached_live_quote(ticker)
            st.metric(label=f"{ticker} price", value=f"${quote['price']:.2f}",
                      delta=f"{quote['change']:+.2f} ({quote['percent_change']:+.2f}%)")
            st.caption(f"Day range: ${quote['low']:.2f} - ${quote['high']:.2f}  |  "
                       f"Prev close: ${quote['prev_close']:.2f}")
            st.button("Refresh quote", on_click=lambda: cached_live_quote.clear())
        except RuntimeError as e:
            st.warning(str(e))
            st.info(f"Showing most recent historical close instead: "
                     f"${float(prices['Close'].iloc[-1]):.2f} on {prices.index[-1].date()}")
        except Exception as e:
            st.error(f"Live quote fetch failed: {e}")

    with col2:
        st.caption(f"Risk-free rate (13-week T-bill proxy): {r_rate:.3%}  |  "
                   f"Dividend yield: {div_yield:.3%}")

    st.subheader(f"Historical Price ({history_years}y)")
    fig, ax = new_dark_fig(figsize=(12, 4))
    ax.plot(prices.index, prices["Close"], lw=1.3, color=COLOR_ACTUAL)
    ax.set_ylabel("Close price")
    ax.set_title(f"{ticker} daily close")
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 2: GBM diagnostics
# ---------------------------------------------------------------------------

with tab_gbm:
    st.subheader("Geometric Brownian Motion: Fit and Diagnostics")
    st.caption(
        "**In plain terms:** this tab asks \"what if this stock's volatility were constant "
        "forever?\" and simulates possible futures under that assumption. The bottom-right "
        "chart below is the key evidence that this assumption is unrealistic."
    )

    mu_hat, sigma_hat = estimate_gbm_params(log_returns)
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated annualized drift (mu)", f"{mu_hat:.2%}")
    col2.metric("Estimated annualized volatility (sigma)", f"{sigma_hat:.2%}")
    col3.metric("Observations used", f"{len(log_returns)}")

    n_paths = st.slider("Number of simulated paths to display", 10, 200, 50)
    S0 = float(prices["Close"].iloc[-1])
    t, S = simulate_gbm_paths(S0, mu_hat, sigma_hat, T=1.0, n_steps=252, n_paths=n_paths, seed=1)

    st.markdown(
        "The panel below is the key motivating result for this whole project: the **rolling "
        "realized volatility** plot (bottom-right) should be flat if GBM's constant-volatility "
        "assumption held -- in real data it visibly clusters and swings, which is exactly what "
        "Heston's mean-reverting stochastic variance is designed to capture."
    )
    fig = gbm_diagnostic_panel(t, S, log_returns)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 3: Heston / Double Heston calibration
# ---------------------------------------------------------------------------

with tab_calib:
    st.subheader("Return-Based GMM Calibration")
    st.caption(
        "**In plain terms:** this tab estimates HOW volatile the stock really is (theta), how "
        "quickly volatility snaps back to normal after a shock (kappa), how wildly volatility "
        "itself swings (xi), and whether price drops coincide with volatility spikes (rho -- "
        "usually negative for stocks). It then checks whether a SECOND, independent volatility "
        "factor (Double Heston) actually helps -- often it doesn't, and that's an honest, "
        "useful finding, not a failure. Calibrated from the historical RETURN series alone "
        "(rolling realized variance and its autocorrelation structure) -- no options data used. "
        "See README for the full methodology and honest limitations of this approach."
    )

    single = cached_heston_calibration(ticker, history_years)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("kappa", f"{single['kappa']:.3f}")
    col2.metric("theta", f"{single['theta']:.4f}")
    col3.metric("xi", f"{single['xi']:.3f}")
    col4.metric("rho", f"{single['rho']:.3f}")
    col5.metric("v0", f"{single['v0']:.4f}")

    feller_ok = single["feller_satisfied"]
    st.markdown(f"**Feller condition (2*kappa*theta >= xi^2):** "
                f"{'SATISFIED' if feller_ok else 'VIOLATED'} (slack = {single['feller_slack']:.4f}). "
                + ("" if feller_ok else "Violation is common for real calibrated equity parameters -- "
                                         "the model still works correctly, see models/heston.py docstring."))
    st.markdown(f"Long-run annualized volatility implied by theta: **{np.sqrt(single['theta']):.2%}**")

    double = cached_double_heston_calibration(ticker, history_years, single)
    st.markdown("---")
    st.markdown("**Double Heston (two-factor):**")
    dcol1, dcol2, dcol3 = st.columns(3)
    dcol1.markdown(f"Factor 1 (fast): kappa1={double['kappa1']:.3f}, theta1={double['theta1']:.4f}, "
                    f"xi1={double['xi1']:.3f}")
    dcol2.markdown(f"Factor 2 (slow): kappa2={double['kappa2']:.3f}, theta2={double['theta2']:.4f}, "
                    f"xi2={double['xi2']:.3f}")
    dcol3.markdown(f"Combined long-run vol: {np.sqrt(double['theta1']+double['theta2']):.2%}")

    if double["theta2"] < 0.05 * double["theta1"]:
        st.info(
            "Factor 2's long-run variance share is small relative to factor 1 -- the calibration "
            "is telling you a single mean-reverting factor already explains most of this ticker's "
            "realized-variance persistence. This is a genuine, honest finding (checked across "
            "multiple tickers during development), not a bug -- see README."
        )

    st.subheader("ACF Fit: Empirical vs Model-Implied")
    moments = single["moments"]
    lags_days = np.arange(1, len(moments["lags_years"]) + 1)
    single_acf_model = _model_acf(single["kappa"], moments["lags_years"])
    double_acf_model = _double_model_acf(double["kappa1"], double["theta1"], double["xi1"],
                                          double["kappa2"], double["theta2"], double["xi2"],
                                          moments["lags_years"])

    fig, ax = new_dark_fig(figsize=(10, 5))
    ax.plot(lags_days, moments["acf_emp"], ".", color=TEXT_SECONDARY, label="Empirical ACF (realized variance)", ms=5)
    ax.plot(lags_days, single_acf_model, "-", color=COLOR_HESTON, label="Single Heston fit", lw=1.8)
    ax.plot(lags_days, double_acf_model, "--", color=COLOR_DOUBLE_HESTON, label="Double Heston fit", lw=1.8)
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{ticker}: realized-variance ACF decay vs model fit")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 4: 30-year walk-forward backtest
# ---------------------------------------------------------------------------

with tab_backtest:
    st.subheader("30-Year Walk-Forward Out-of-Sample Backtest -- Full Technical Detail")
    st.caption(
        "The **Results (Start Here)** tab gives the plain-language version of this with fixed "
        "settings (5y calibration / 1y test). This tab is the same underlying test, but lets "
        "you adjust the window sizes and shows the full numeric tables (RMSE, MAE, VaR breach "
        "rates, Kupiec p-values) behind the headline finding."
    )

    calib_years = st.slider("Calibration window (years)", 2, 10, 5)
    test_years = st.slider("Test window (years)", 1, 3, 1)

    if st.button("Run backtest", type="primary"):
        try:
            results_df, kupiec = cached_backtest(ticker, history_years, calib_years, test_years, test_years)
            st.session_state["backtest_result"] = (results_df, kupiec)
        except ValueError as e:
            st.error(str(e))

    if "backtest_result" in st.session_state:
        results_df, kupiec = st.session_state["backtest_result"]

        st.subheader("Summary Table")
        table = summary_table(results_df, kupiec)
        st.dataframe(table.style.format({
            "vol_RMSE_overall": "{:.4f}", "vol_MAE_overall": "{:.4f}", "vol_bias_overall": "{:.4f}",
            "vol_RMSE_calm": "{:.4f}", "vol_RMSE_turbulent": "{:.4f}",
            "VaR_breach_rate": "{:.2%}", "VaR_target_rate": "{:.2%}", "kupiec_p_value": "{:.2e}",
        }))

        st.subheader("Volatility Forecast: Realized vs Model, Over Time")
        fig, ax = new_dark_fig(figsize=(12, 5))
        ax.plot(results_df["test_start_date"], results_df["realized_vol"], "-o", color=COLOR_ACTUAL, label="Realized (actual)", lw=2)
        ax.plot(results_df["test_start_date"], results_df["gbm_forecast_vol"], "--s", color=COLOR_GBM, label="GBM forecast", alpha=0.9)
        ax.plot(results_df["test_start_date"], results_df["heston_forecast_vol"], "--^", color=COLOR_HESTON, label="Heston forecast", alpha=0.9)
        if "dh_forecast_vol" in results_df.columns:
            ax.plot(results_df["test_start_date"], results_df["dh_forecast_vol"], "--d", color=COLOR_DOUBLE_HESTON, label="Double Heston forecast", alpha=0.9)
        ax.set_ylabel("Annualized volatility")
        ax.set_title(f"{ticker}: walk-forward out-of-sample volatility forecast")
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = new_dark_fig(figsize=(7, 5))
            plot_rmse_bar_chart(results_df, kupiec, ax=ax)
            apply_dark_style(fig, ax)
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, ax = new_dark_fig(figsize=(7, 5))
            plot_var_breach_comparison(results_df, kupiec, ax=ax)
            apply_dark_style(fig, ax)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Click 'Run backtest' to compute (takes ~20-30 seconds -- calibrates Heston and "
                "Double Heston across ~24 rolling 5-year windows).")


# ---------------------------------------------------------------------------
# Tab 5: Theoretical option pricing (illustrative, not fit to real quotes)
# ---------------------------------------------------------------------------

with tab_pricing:
    st.subheader("Theoretical Option Pricing (Illustrative)")
    st.caption(
        "Uses the calibrated model parameters to price a HYPOTHETICAL option -- these prices are "
        "NOT fit to or compared against any real market option quotes (this project calibrates "
        "purely from equity returns; see README). This tab exists to make the pricing formulas "
        "tangible, not as a trading tool."
    )

    S0 = float(prices["Close"].iloc[-1])
    col1, col2, col3 = st.columns(3)
    K = col1.number_input("Strike", value=float(round(S0)), step=1.0)
    T_days = col2.slider("Days to expiry", 7, 730, 90)
    option_type = col3.selectbox("Option type", ["call", "put"])
    T = T_days / 365.0

    bs_sigma = st.slider("Black-Scholes volatility (annualized)", 0.05, 1.0,
                          float(np.sqrt(single["theta"])), step=0.01)

    bs_p = bs_price(S0, K, T, r_rate, bs_sigma, div_yield, option_type)
    heston_p = heston_price(S0, K, T, r_rate, div_yield, single["kappa"], single["theta"],
                             single["xi"], single["rho"], single["v0"], option_type)
    dh_p = double_heston_price(S0, K, T, r_rate, div_yield,
                                double["kappa1"], double["theta1"], double["xi1"], double["rho1"], double["v1_0"],
                                double["kappa2"], double["theta2"], double["xi2"], double["rho2"], double["v2_0"],
                                option_type)

    col1, col2, col3 = st.columns(3)
    col1.metric("Black-Scholes price", f"${bs_p:.2f}")
    col2.metric("Heston price", f"${heston_p:.2f}")
    col3.metric("Double Heston price", f"${dh_p:.2f}")

    st.markdown("**Model-implied smile at this expiry** (illustrative, not compared to real quotes):")
    strikes = np.linspace(S0 * 0.7, S0 * 1.3, 25)
    from models.black_scholes import implied_volatility
    ivs_heston, ivs_dh = [], []
    for K_i in strikes:
        p_h = heston_price(S0, K_i, T, r_rate, div_yield, single["kappa"], single["theta"],
                            single["xi"], single["rho"], single["v0"], "call")
        p_dh = double_heston_price(S0, K_i, T, r_rate, div_yield,
                                    double["kappa1"], double["theta1"], double["xi1"], double["rho1"], double["v1_0"],
                                    double["kappa2"], double["theta2"], double["xi2"], double["rho2"], double["v2_0"],
                                    "call")
        ivs_heston.append(implied_volatility(p_h, S0, K_i, T, r_rate, div_yield, "call"))
        ivs_dh.append(implied_volatility(p_dh, S0, K_i, T, r_rate, div_yield, "call"))

    fig, ax = new_dark_fig(figsize=(9, 5))
    ax.plot(strikes, np.array(ivs_heston) * 100, "-o", color=COLOR_HESTON, ms=4, label="Heston-implied")
    ax.plot(strikes, np.array(ivs_dh) * 100, "--s", color=COLOR_DOUBLE_HESTON, ms=4, label="Double Heston-implied")
    ax.axhline(bs_sigma * 100, color=COLOR_BLACK_SCHOLES, ls=":", lw=2, label="Black-Scholes (flat)")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"Model-implied smile, T={T_days}d (calibrated params, illustrative)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tab 6: Options strategy builder
# ---------------------------------------------------------------------------

with tab_strategy:
    st.subheader("Options Strategy Builder")
    st.caption(
        "**In plain terms:** pick a strategy -- a straddle bets on a big move either way; a "
        "vertical spread caps both your risk and reward; an iron condor collects premium if "
        "the stock stays in a range -- and this tab prices every leg with either Black-Scholes "
        "(one flat volatility) or this ticker's calibrated Heston model, then shows exactly "
        "what you'd pay or collect, where you break even, and the full payoff at expiry."
    )

    pricer_choice = st.radio("Pricing model", ["Black-Scholes", "Heston"], horizontal=True,
                              key="strat_pricer")
    S0_strat = float(prices["Close"].iloc[-1])
    T_days_strat = st.slider("Days to expiry", 7, 365, 30, key="strat_T")
    T_strat = T_days_strat / 365.0

    if pricer_choice == "Black-Scholes":
        sigma_strat = st.slider("Volatility (annualized)", 0.05, 1.5,
                                 float(np.sqrt(single["theta"])), step=0.01, key="strat_sigma")
        pricer = bs_pricer(S0_strat, r_rate, sigma_strat, div_yield)
        greeks_fn = bs_greeks_fn(S0_strat, r_rate, sigma_strat, div_yield)
    else:
        pricer = heston_pricer(S0_strat, r_rate, div_yield, single["kappa"], single["theta"],
                                single["xi"], single["rho"], single["v0"])
        greeks_fn = heston_greeks_fn(S0_strat, r_rate, div_yield, single["kappa"], single["theta"],
                                      single["xi"], single["rho"], single["v0"])

    strategy_choice = st.selectbox("Strategy", [
        "Long Straddle", "Short Straddle", "Long Strangle", "Short Strangle",
        "Bull Call Spread", "Bear Put Spread", "Iron Condor",
    ], key="strat_strategy")

    legs = None
    if strategy_choice in ("Long Straddle", "Short Straddle"):
        K = st.number_input("Strike", value=float(round(S0_strat)), step=1.0, key="strat_K_straddle")
        legs = straddle(K, long=(strategy_choice == "Long Straddle"))
    elif strategy_choice in ("Long Strangle", "Short Strangle"):
        c1, c2 = st.columns(2)
        K_put = c1.number_input("Put strike", value=float(round(S0_strat * 0.95)), step=1.0,
                                 key="strat_K_put_strangle")
        K_call = c2.number_input("Call strike", value=float(round(S0_strat * 1.05)), step=1.0,
                                  key="strat_K_call_strangle")
        try:
            legs = strangle(K_put, K_call, long=(strategy_choice == "Long Strangle"))
        except ValueError as e:
            st.error(str(e))
    elif strategy_choice == "Bull Call Spread":
        c1, c2 = st.columns(2)
        K_long = c1.number_input("Long call strike", value=float(round(S0_strat * 0.97)), step=1.0,
                                  key="strat_K_long_bull")
        K_short = c2.number_input("Short call strike", value=float(round(S0_strat * 1.05)), step=1.0,
                                   key="strat_K_short_bull")
        legs = vertical_spread(K_long, K_short, "call")
    elif strategy_choice == "Bear Put Spread":
        c1, c2 = st.columns(2)
        K_long = c1.number_input("Long put strike", value=float(round(S0_strat * 1.03)), step=1.0,
                                  key="strat_K_long_bear")
        K_short = c2.number_input("Short put strike", value=float(round(S0_strat * 0.95)), step=1.0,
                                   key="strat_K_short_bear")
        legs = vertical_spread(K_long, K_short, "put")
    else:  # Iron Condor
        c1, c2, c3, c4 = st.columns(4)
        K_pl = c1.number_input("Put long", value=float(round(S0_strat * 0.85)), step=1.0, key="strat_K_pl")
        K_ps = c2.number_input("Put short", value=float(round(S0_strat * 0.93)), step=1.0, key="strat_K_ps")
        K_cs = c3.number_input("Call short", value=float(round(S0_strat * 1.07)), step=1.0, key="strat_K_cs")
        K_cl = c4.number_input("Call long", value=float(round(S0_strat * 1.15)), step=1.0, key="strat_K_cl")
        try:
            legs = iron_condor(K_pl, K_ps, K_cs, K_cl)
        except ValueError as e:
            st.error(str(e))

    if legs is not None:
        try:
            summary = strategy_summary(legs, T_strat, pricer, S0=S0_strat)
            greeks = strategy_greeks(legs, T_strat, greeks_fn)
        except Exception as e:
            st.error(f"Could not price this strategy: {e}")
            summary = None

        if summary is not None:
            col1, col2, col3 = st.columns(3)
            if summary["is_credit"]:
                col1.metric("Net credit received", f"${-summary['entry_cost']:.2f}/share")
            else:
                col1.metric("Net debit paid", f"${summary['entry_cost']:.2f}/share")
            col2.metric("Breakeven(s)", ", ".join(f"${b:.2f}" for b in summary["breakevens"]) or "none in range")
            max_p = "Unlimited*" if summary["profit_uncapped"] else f"${summary['max_profit']:.2f}"
            max_l = "Unlimited*" if summary["loss_uncapped"] else f"${summary['max_loss']:.2f}"
            col3.metric("Max profit / Max loss", f"{max_p} / {max_l}")
            if summary["profit_uncapped"] or summary["loss_uncapped"]:
                st.caption("*Still trending away from zero at the edge of the scanned price range "
                           "(+/-50% of spot) -- the true max is unbounded.")

            gcol1, gcol2, gcol3, gcol4 = st.columns(4)
            gcol1.metric("Delta", f"{greeks['delta']:.3f}")
            gcol2.metric("Gamma", f"{greeks['gamma']:.4f}")
            gcol3.metric("Vega", f"{greeks['vega']:.3f}")
            gcol4.metric("Theta/day", f"{greeks['theta_daily']:.3f}")

            fig, ax = new_dark_fig(figsize=(10, 5))
            ax.plot(summary["S_grid"], summary["payoff"], color=COLOR_ACTUAL, lw=2)
            ax.axhline(0, color=TEXT_SECONDARY, lw=1, ls="--")
            ax.axvline(S0_strat, color=COLOR_HESTON, lw=1, ls=":", label=f"Current price ${S0_strat:.2f}")
            for b in summary["breakevens"]:
                ax.axvline(b, color=COLOR_GBM, lw=1, ls=":", alpha=0.7)
            ax.set_xlabel("Underlying price at expiry")
            ax.set_ylabel("P&L per share ($)")
            ax.set_title(f"{strategy_choice}: payoff at expiry ({pricer_choice} pricing)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            st.caption("P&L is per share -- multiply by 100 for the standard "
                       "one-contract-per-100-shares convention on real US-listed options. "
                       "Not adjusted for commissions.")


# ---------------------------------------------------------------------------
# Tab 7: Live mispricing scanner
# ---------------------------------------------------------------------------

with tab_scanner:
    st.subheader("Live Mispricing Scanner")
    st.caption(
        "**What this compares:** today's REAL option chain (market prices) against this "
        "ticker's Heston model, calibrated PURELY from historical returns -- not fit to this "
        "chain at all (see README's methodology note). A big gap does NOT mean free money: it "
        "just as easily means the options market knows something about the future that trailing "
        "returns can't see as it means the market is mispricing the contract. Treat this as a "
        "diagnostic, not a trade signal -- see trading/scanner.py for the full reasoning."
    )
    max_expiries = st.slider("Number of expiries to scan", 1, 10, 4, key="scan_max_expiries")
    if st.button("Fetch live chain and scan", type="primary", key="scan_button"):
        try:
            with st.spinner("Fetching live chain and scanning against the Heston model..."):
                spot, chain = cached_clean_chain(ticker, max_expiries)
                scan = scan_chain_vs_heston(chain, spot, r_rate, div_yield, single)
            st.session_state["scan_result"] = (spot, scan)
        except Exception as e:
            st.error(f"Could not fetch/scan the live chain for '{ticker}': {e}")

    if "scan_result" in st.session_state:
        spot, scan = st.session_state["scan_result"]
        if scan.empty:
            st.warning("No liquid (bid/ask-quoted) contracts survived cleaning for this "
                       "ticker/expiry range -- try a more liquid ticker or more expiries.")
        else:
            top = top_mispricings(scan, n=15, min_abs_z=1.0)
            st.markdown(f"Spot: **${spot:.2f}** | {len(scan)} liquid contracts scanned | "
                        f"{len(top)} flagged as statistical outliers (|z| >= 1) against that "
                        f"day's own dispersion")
            if len(top) > 0:
                st.dataframe(top[["expiry", "strike", "option_type", "moneyness", "market_iv",
                                   "model_iv", "iv_gap", "iv_gap_z", "rich_or_cheap"]].style.format({
                    "strike": "{:.1f}", "moneyness": "{:.3f}", "market_iv": "{:.2%}",
                    "model_iv": "{:.2%}", "iv_gap": "{:.2%}", "iv_gap_z": "{:.2f}",
                }))
            else:
                st.info("No outlier contracts (|z| >= 1) today -- the market and the "
                        "returns-only Heston model broadly agree on this chain.")

            fig, ax = new_dark_fig(figsize=(10, 5))
            calls = scan[scan["option_type"] == "call"]
            ax.scatter(calls["strike"], calls["market_iv"] * 100, s=18, color=COLOR_ACTUAL,
                       label="Market IV (calls)", alpha=0.8)
            ax.scatter(calls["strike"], calls["model_iv"] * 100, s=18, color=COLOR_HESTON,
                       label="Heston model IV (calls)", alpha=0.8)
            ax.axvline(spot, color=TEXT_SECONDARY, ls=":", lw=1, label=f"Spot ${spot:.2f}")
            ax.set_xlabel("Strike")
            ax.set_ylabel("Implied volatility (%)")
            ax.set_title(f"{ticker}: market vs. Heston-model implied vol, all scanned expiries")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Click 'Fetch live chain and scan' to pull today's real option chain "
                "(a few seconds per expiry).")


# ---------------------------------------------------------------------------
# Tab 8: Volatility risk premium backtest (SPY, always -- see caption)
# ---------------------------------------------------------------------------

with tab_vrp:
    st.subheader("Volatility Risk Premium Backtest: Selling SPY Straddles, 1993-Today")
    st.caption(
        "This section always uses SPY, regardless of the ticker chosen in the sidebar -- it "
        "uses the CBOE VIX index, which is specific to SPX/SPY, as the market's implied-vol "
        "input (see trading/vrp_backtest.py for why this is how the project gets a REAL, "
        "decades-long options-market-based backtest without paid historical option-chain data)."
    )
    with st.container(border=True):
        st.markdown(
            "**In plain terms:** every month, this either sells an at-the-money 1-month SPY "
            "straddle (collecting the VIX-implied premium, betting SPY stays roughly where it "
            "is) or stays in cash, based on whether VIX is pricing in MORE volatility than this "
            "project's own Heston model expects over that month. It's compared against blindly "
            "selling every single month, and against simply buying and holding SPY."
        )
        st.markdown(
            "**Important limitations (read before drawing conclusions):** this is an "
            "AT-THE-MONEY, UNHEDGED, held-to-expiry straddle struck exactly at spot, priced "
            "with a single flat VIX-implied vol -- not a real chain's discrete strikes or skew, "
            "and not delta-hedged along the way. It approximates a benchmark-index-style options "
            "overlay (the same methodology CBOE's own published PUT/BXM indices use), not a "
            "claim about what a real brokerage fill would achieve."
        )

    c1, c2, c3, c4 = st.columns(4)
    vrp_years = c1.slider("Years of history", 6, 33, 33, key="vrp_years")
    vrp_calib_years = c2.slider("Heston calibration window (years)", 2, 10, 5, key="vrp_calib_years")
    vrp_cost = c3.slider("Round-trip cost (vol points)", 0.0, 3.0, 1.0, step=0.25, key="vrp_cost") / 100.0
    vrp_threshold = c4.slider("VRP signal threshold (vol points)", -5.0, 10.0, 0.0, step=0.5,
                               key="vrp_threshold") / 100.0

    if st.button("Run VRP backtest", type="primary", key="vrp_button"):
        try:
            with st.spinner(f"Walking forward through {vrp_years} years of SPY/VIX, "
                             f"recalibrating Heston once per year (roughly {max(vrp_years, 6)//2}s)..."):
                df_signal = cached_vrp_backtest(vrp_years, vrp_calib_years, 21, vrp_cost, vrp_threshold)
                df_always = cached_vrp_backtest(vrp_years, vrp_calib_years, 21, vrp_cost, -999.0)
                bh = cached_buy_hold_spy(vrp_years)
            st.session_state["vrp_result"] = (df_signal, df_always, bh)
        except ValueError as e:
            st.error(str(e))

    if "vrp_result" in st.session_state:
        df_signal, df_always, bh = st.session_state["vrp_result"]
        s_sig, s_alw = df_signal.attrs["summary"], df_always.attrs["summary"]

        metrics_df = pd.DataFrame([
            {"Strategy": "VRP-signal-filtered", "CAGR": s_sig["cagr"], "Ann. vol": s_sig["annualized_vol"],
             "Sharpe": s_sig["sharpe"], "Max drawdown": s_sig["max_drawdown"],
             "Win rate": s_sig["win_rate"], "Active months": f"{s_sig['n_active_periods']}/{s_sig['n_periods']}"},
            {"Strategy": "Always sell", "CAGR": s_alw["cagr"], "Ann. vol": s_alw["annualized_vol"],
             "Sharpe": s_alw["sharpe"], "Max drawdown": s_alw["max_drawdown"],
             "Win rate": s_alw["win_rate"], "Active months": f"{s_alw['n_active_periods']}/{s_alw['n_periods']}"},
            {"Strategy": "Buy & hold SPY", "CAGR": bh["cagr"], "Ann. vol": bh["annualized_vol"],
             "Sharpe": bh["sharpe"], "Max drawdown": bh["max_drawdown"], "Win rate": np.nan,
             "Active months": "-"},
        ]).set_index("Strategy")
        st.dataframe(metrics_df.style.format({
            "CAGR": "{:.2%}", "Ann. vol": "{:.2%}", "Sharpe": "{:.2f}",
            "Max drawdown": "{:.2%}", "Win rate": "{:.1%}",
        }))

        fig, ax = new_dark_fig(figsize=(11, 5))
        ax.plot(df_signal["exit_date"], s_sig["equity_curve"], color=CAT[5], lw=2, label="VRP-signal-filtered")
        ax.plot(df_always["exit_date"], s_alw["equity_curve"], color=CAT[4], lw=2, label="Always sell")
        ax.plot(bh["equity_curve"].index, bh["equity_curve"].values, color=COLOR_ACTUAL, lw=1.5,
                alpha=0.8, label="Buy & hold SPY")
        ax.set_yscale("log")
        ax.set_ylabel("Growth of $1 (log scale)")
        ax.set_title(f"SPY short-straddle overlay vs. buy & hold, {vrp_years} years")
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

        best_sharpe = max([("VRP-signal-filtered", s_sig["sharpe"]), ("Always sell", s_alw["sharpe"]),
                            ("Buy & hold SPY", bh["sharpe"])], key=lambda x: x[1])[0]
        st.markdown(
            f"**Best risk-adjusted return (Sharpe) over this window: {best_sharpe}.** Selling "
            "volatility has historically collected a real premium here -- VIX has priced in "
            "more volatility than materialized, on average, for most of this history -- but it "
            "comes with fat-tail risk a monthly Sharpe ratio doesn't fully capture (a single bad "
            "month can erase many months of collected premium; see max drawdown above)."
        )
    else:
        st.info("Click 'Run VRP backtest' to compute (roughly 5-20 seconds depending on the "
                "years selected -- recalibrates Heston once per year of history).")
