"""
report/generate_figures.py
=============================
Generates every figure used in report/build_report.py, using the real
project pipeline on real AAPL data (30 years). Run this before
build_report.py whenever the report needs refreshing with new numbers.

Not part of the core deliverable -- a one-off script to produce report
assets from already-validated project code.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

from data.loader import get_price_history, compute_log_returns, get_risk_free_rate, get_dividend_yield
from models.gbm import simulate_gbm_paths, estimate_gbm_params
from models.black_scholes import bs_price, implied_volatility
from models.heston import heston_price
from models.heston_mc import heston_mc_price
from models.double_heston import double_heston_price
from analysis.gbm_diagnostics import gbm_diagnostic_panel
from calibration.heston_calibration import calibrate_heston_gmm, _model_acf
from calibration.double_heston_calibration import calibrate_double_heston_gmm, _double_model_acf
from analysis.backtest import run_walk_forward_backtest
from analysis.model_comparison import summary_table, plot_rmse_bar_chart, plot_var_breach_comparison

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TICKER = "AAPL"
HISTORY_YEARS = 30

numbers = {}  # collected key numbers for the report text, dumped to json at the end

print(f"Fetching {HISTORY_YEARS}y history for {TICKER}...")
prices = get_price_history(TICKER, years=HISTORY_YEARS)
log_returns = compute_log_returns(prices)
r_rate = get_risk_free_rate()
div_yield = get_dividend_yield(TICKER)
S0 = float(prices["Close"].iloc[-1])

numbers["ticker"] = TICKER
numbers["history_years"] = HISTORY_YEARS
numbers["history_start"] = str(prices.index.min().date())
numbers["history_end"] = str(prices.index.max().date())
numbers["n_obs"] = len(prices)
numbers["S0"] = S0
numbers["risk_free_rate"] = r_rate
numbers["dividend_yield"] = div_yield

# --- Figure 1: historical price ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(prices.index, prices["Close"], lw=1)
ax.set_title(f"{TICKER}: {HISTORY_YEARS}-year daily close price")
ax.set_ylabel("Price (USD)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/01_price_history.png", dpi=150)
plt.close(fig)
print("saved 01_price_history.png")

# --- Figure 2: GBM diagnostic panel ---
mu_hat, sigma_hat = estimate_gbm_params(log_returns)
numbers["gbm_mu"] = mu_hat
numbers["gbm_sigma"] = sigma_hat
t, S = simulate_gbm_paths(S0, mu_hat, sigma_hat, T=1.0, n_steps=252, n_paths=200, seed=1)
fig = gbm_diagnostic_panel(t, S, log_returns)
fig.savefig(f"{FIG_DIR}/02_gbm_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 02_gbm_diagnostics.png")

# --- Black-Scholes validation numbers ---
C = bs_price(100, 100, 1.0, r_rate, sigma_hat, div_yield, "call")
P = bs_price(100, 100, 1.0, r_rate, sigma_hat, div_yield, "put")
numbers["bs_parity_lhs"] = C - P
numbers["bs_parity_rhs"] = 100 * np.exp(-div_yield * 1.0) - 100 * np.exp(-r_rate * 1.0)
price_demo = bs_price(S0, S0 * 1.05, 0.25, r_rate, sigma_hat, div_yield, "call")
iv_demo = implied_volatility(price_demo, S0, S0 * 1.05, 0.25, r_rate, div_yield, "call")
numbers["bs_iv_roundtrip_input_sigma"] = sigma_hat
numbers["bs_iv_roundtrip_recovered"] = iv_demo

# --- Heston calibration (5-year window for a SENSIBLE calibration, per the
# stationarity finding) ---
prices_5y = get_price_history(TICKER, years=5)
log_returns_5y = compute_log_returns(prices_5y)
heston_result = calibrate_heston_gmm(log_returns_5y, n_multistarts=8, seed=1)
numbers["heston_5y"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                         for k, v in heston_result.items() if k in
                         ["kappa", "theta", "xi", "rho", "v0", "feller_satisfied", "feller_slack"]}

# Also the 30-year (deliberately non-stationary) calibration, for the honest caveat figure
heston_result_30y = calibrate_heston_gmm(log_returns, n_multistarts=8, seed=1)
numbers["heston_30y"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in heston_result_30y.items() if k in
                          ["kappa", "theta", "xi", "rho", "v0", "feller_satisfied", "feller_slack"]}

# --- Figure 3: ACF fit (5-year calibration) ---
moments = heston_result["moments"]
lags_days = np.arange(1, len(moments["lags_years"]) + 1)
acf_model_single = _model_acf(heston_result["kappa"], moments["lags_years"])

double_result = calibrate_double_heston_gmm(log_returns_5y, n_multistarts=10, seed=1,
                                             single_heston_result=heston_result)
numbers["double_heston_5y"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                for k, v in double_result.items() if k in
                                ["kappa1", "theta1", "xi1", "kappa2", "theta2", "xi2", "rho1",
                                 "v1_0", "v2_0", "feller_satisfied"]}

acf_model_double = _double_model_acf(double_result["kappa1"], double_result["theta1"], double_result["xi1"],
                                      double_result["kappa2"], double_result["theta2"], double_result["xi2"],
                                      moments["lags_years"])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(lags_days, moments["acf_emp"], "k.", label="Empirical ACF (realized variance)", ms=4)
ax.plot(lags_days, acf_model_single, "b-", label="Single Heston fit", lw=1.5)
ax.plot(lags_days, acf_model_double, "r--", label="Double Heston fit", lw=1.5)
ax.set_xlabel("Lag (trading days)")
ax.set_ylabel("Autocorrelation")
ax.set_title(f"{TICKER} (5y calibration window): realized-variance ACF vs model fits")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/03_acf_fit.png", dpi=150)
plt.close(fig)
print("saved 03_acf_fit.png")

theta2_share = double_result["theta2"] / (double_result["theta1"] + double_result["theta2"])
numbers["double_heston_5y"]["theta2_share"] = float(theta2_share)

# --- CF vs MC cross-validation numbers ---
K_demo, T_demo = S0 * 1.02, 0.5
cf_price = heston_price(S0, K_demo, T_demo, r_rate, div_yield, heston_result["kappa"],
                         heston_result["theta"], heston_result["xi"], heston_result["rho"],
                         heston_result["v0"], "call")
mc_price, mc_stderr = heston_mc_price(S0, K_demo, T_demo, r_rate, div_yield, heston_result["kappa"],
                                       heston_result["theta"], heston_result["xi"], heston_result["rho"],
                                       heston_result["v0"], "call", n_steps=500, n_paths=100_000, seed=1)
numbers["cf_vs_mc"] = {"cf_price": float(cf_price), "mc_price": float(mc_price),
                        "mc_stderr": float(mc_stderr), "K": K_demo, "T": T_demo}

# --- Figure 4: illustrative Heston/Double Heston/BS smile from calibrated params ---
strikes = np.linspace(S0 * 0.7, S0 * 1.3, 25)
ivs_heston, ivs_dh = [], []
for K_i in strikes:
    p_h = heston_price(S0, K_i, 0.5, r_rate, div_yield, heston_result["kappa"], heston_result["theta"],
                        heston_result["xi"], heston_result["rho"], heston_result["v0"], "call")
    p_dh = double_heston_price(S0, K_i, 0.5, r_rate, div_yield,
                                double_result["kappa1"], double_result["theta1"], double_result["xi1"],
                                double_result["rho1"], double_result["v1_0"],
                                double_result["kappa2"], double_result["theta2"], double_result["xi2"],
                                double_result["rho2"], double_result["v2_0"], "call")
    ivs_heston.append(implied_volatility(p_h, S0, K_i, 0.5, r_rate, div_yield, "call"))
    ivs_dh.append(implied_volatility(p_dh, S0, K_i, 0.5, r_rate, div_yield, "call"))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(strikes, np.array(ivs_heston) * 100, "r-o", ms=3, label="Heston-implied")
ax.plot(strikes, np.array(ivs_dh) * 100, "g--s", ms=3, label="Double Heston-implied")
ax.axhline(np.sqrt(heston_result["theta"]) * 100, color="b", ls=":", label="Black-Scholes (flat, sqrt(theta))")
ax.set_xlabel("Strike")
ax.set_ylabel("Implied volatility (%)")
ax.set_title(f"Model-implied volatility smile, T=0.5y (illustrative -- calibrated params, not fit to market quotes)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/04_illustrative_smile.png", dpi=150)
plt.close(fig)
print("saved 04_illustrative_smile.png")

# --- Walk-forward backtest ---
print("Running 30-year walk-forward backtest (~20-30s)...")
results_df, kupiec_summary = run_walk_forward_backtest(
    log_returns, calib_years=5, test_years=1, step_years=1, seed=1, include_double_heston=True)

numbers["kupiec"] = kupiec_summary
table = summary_table(results_df, kupiec_summary)
numbers["comparison_table"] = table.to_dict(orient="records")

# --- Figure 5: walk-forward vol forecast ---
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(results_df["test_start_date"], results_df["realized_vol"], "k-o", label="Realized (actual)", lw=2)
ax.plot(results_df["test_start_date"], results_df["gbm_forecast_vol"], "b--s", label="GBM forecast", alpha=0.8)
ax.plot(results_df["test_start_date"], results_df["heston_forecast_vol"], "r--^", label="Heston forecast", alpha=0.8)
ax.plot(results_df["test_start_date"], results_df["dh_forecast_vol"], "g--d", label="Double Heston forecast", alpha=0.8)
ax.set_ylabel("Annualized volatility")
ax.set_title(f"{TICKER}: walk-forward out-of-sample volatility forecast, "
             f"{results_df.test_start_date.min().year}-{results_df.test_end_date.max().year}")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/05_backtest_vol_forecast.png", dpi=150)
plt.close(fig)
print("saved 05_backtest_vol_forecast.png")

# --- Figure 6: comparison bar charts ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_rmse_bar_chart(results_df, kupiec_summary, ax=axes[0])
plot_var_breach_comparison(results_df, kupiec_summary, ax=axes[1])
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/06_model_comparison.png", dpi=150)
plt.close(fig)
print("saved 06_model_comparison.png")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers.json"), "w") as f:
    json.dump(numbers, f, indent=2, default=str)
print("\nSaved report/numbers.json with all figures for the report text.")
print("All figures saved to report/figures/")
