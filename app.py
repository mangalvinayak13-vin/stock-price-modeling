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

from data.loader import get_price_history, compute_log_returns, get_risk_free_rate, get_dividend_yield
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Configuration")
ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
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


tab_live, tab_gbm, tab_calib, tab_backtest, tab_pricing = st.tabs(
    ["Live Price", "GBM Diagnostics", "Heston / Double Heston Calibration",
     "30-Year Walk-Forward Backtest", "Theoretical Option Pricing"]
)


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
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(prices.index, prices["Close"], lw=1)
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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lags_days, moments["acf_emp"], "k.", label="Empirical ACF (realized variance)", ms=4)
    ax.plot(lags_days, single_acf_model, "b-", label="Single Heston fit", lw=1.5)
    ax.plot(lags_days, double_acf_model, "r--", label="Double Heston fit", lw=1.5)
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
    st.subheader("30-Year Walk-Forward Out-of-Sample Backtest")
    st.caption(
        "**In plain terms -- this is the actual result of the whole project.** Each model is "
        "trained only on a 5-year window, then graded on the year immediately after, which it "
        "never got to see in advance (no cheating). This repeats across 30 years, spanning the "
        "dot-com crash, 2008, and COVID, and asks two questions: (1) how close was each model's "
        "volatility forecast to what actually happened, and (2) if each model set a daily "
        "\"this is the worst 1-in-20-day loss I'd expect\" threshold, did reality actually "
        "breach it about 1 time in 20, as it should? Spoiler: the answers aren't flattering to "
        "the fancier models, and that's reported honestly -- see README for full methodology."
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
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(results_df["test_start_date"], results_df["realized_vol"], "k-o", label="Realized (actual)", lw=2)
        ax.plot(results_df["test_start_date"], results_df["gbm_forecast_vol"], "b--s", label="GBM forecast", alpha=0.8)
        ax.plot(results_df["test_start_date"], results_df["heston_forecast_vol"], "r--^", label="Heston forecast", alpha=0.8)
        if "dh_forecast_vol" in results_df.columns:
            ax.plot(results_df["test_start_date"], results_df["dh_forecast_vol"], "g--d", label="Double Heston forecast", alpha=0.8)
        ax.set_ylabel("Annualized volatility")
        ax.set_title(f"{ticker}: walk-forward out-of-sample volatility forecast")
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            plot_rmse_bar_chart(results_df, kupiec, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 5))
            plot_var_breach_comparison(results_df, kupiec, ax=ax)
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

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(strikes, np.array(ivs_heston) * 100, "r-o", ms=3, label="Heston-implied")
    ax.plot(strikes, np.array(ivs_dh) * 100, "g--s", ms=3, label="Double Heston-implied")
    ax.axhline(bs_sigma * 100, color="b", ls=":", label="Black-Scholes (flat)")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"Model-implied smile, T={T_days}d (calibrated params, illustrative)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
