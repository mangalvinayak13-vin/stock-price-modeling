"""
analysis/model_comparison.py
===============================
Summary tables and plots comparing GBM, Heston, and Double Heston on the
walk-forward backtest results from analysis/backtest.py. This is the
"headline results" module for the report -- everything here consumes the
(results_df, kupiec_summary) tuple that run_walk_forward_backtest()
produces.

REGIME BREAKDOWN (the equity-return analogue of the original project
spec's "short-dated vs long-dated maturity" comparison): since we
calibrate from equity returns rather than an option chain, there's no
option maturity axis to split by. The natural equivalent split for a
return-based comparison is CALM vs TURBULENT periods -- does a model's
relative accuracy hold up uniformly, or does one model specifically win
during high-volatility regimes (where mean-reversion-aware Heston-type
forecasts should, in principle, have the most room to add value over
GBM's constant-vol assumption)? We split walk-forward windows into
"calm" (realized_vol below the sample median) and "turbulent" (above
median) groups and report each model's forecast RMSE within each group
separately.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def summary_table(results_df, kupiec_summary, include_double_heston=True):
    """
    Build the headline summary table: for each model, volatility-forecast
    RMSE/MAE/bias (overall, and split into calm vs turbulent regimes), and
    the pooled VaR breach rate + Kupiec test p-value.

    Returns
    -------
    pd.DataFrame, one row per model.
    """
    models = [("GBM", "gbm_forecast_vol"), ("Heston", "heston_forecast_vol")]
    if include_double_heston and "dh_forecast_vol" in results_df.columns:
        models.append(("Double Heston", "dh_forecast_vol"))

    median_vol = results_df["realized_vol"].median()
    calm_mask = results_df["realized_vol"] <= median_vol
    turbulent_mask = ~calm_mask

    rows = []
    for name, col in models:
        err = results_df[col] - results_df["realized_vol"]
        err_calm = err[calm_mask]
        err_turb = err[turbulent_mask]

        kupiec_key = {"GBM": "gbm", "Heston": "heston", "Double Heston": "double_heston"}[name]
        k = kupiec_summary[kupiec_key]

        rows.append({
            "model": name,
            "vol_RMSE_overall": np.sqrt((err**2).mean()),
            "vol_MAE_overall": err.abs().mean(),
            "vol_bias_overall": err.mean(),
            "vol_RMSE_calm": np.sqrt((err_calm**2).mean()),
            "vol_RMSE_turbulent": np.sqrt((err_turb**2).mean()),
            "VaR_breach_rate": k["breach_rate"],
            "VaR_target_rate": kupiec_summary["target_breach_rate"],
            "kupiec_p_value": k["p_value"],
            "kupiec_rejected": k["rejected_at_5pct"],
        })

    return pd.DataFrame(rows)


def print_summary(results_df, kupiec_summary, include_double_heston=True):
    """Pretty-print the summary table with sensible rounding, for quick
    inspection in a notebook or console (the full-precision DataFrame from
    summary_table() is what should actually go in the report/analysis)."""
    table = summary_table(results_df, kupiec_summary, include_double_heston)
    display_table = table.copy()
    for col in ["vol_RMSE_overall", "vol_MAE_overall", "vol_bias_overall",
                "vol_RMSE_calm", "vol_RMSE_turbulent"]:
        display_table[col] = display_table[col].round(4)
    display_table["VaR_breach_rate"] = (display_table["VaR_breach_rate"] * 100).round(2).astype(str) + "%"
    display_table["VaR_target_rate"] = (display_table["VaR_target_rate"] * 100).round(2).astype(str) + "%"
    display_table["kupiec_p_value"] = display_table["kupiec_p_value"].apply(lambda p: f"{p:.2e}")
    print(display_table.to_string(index=False))
    return table


def plot_rmse_bar_chart(results_df, kupiec_summary, include_double_heston=True, ax=None):
    """Bar chart comparing volatility-forecast RMSE across models, split
    into calm vs turbulent regimes -- the key comparison figure."""
    table = summary_table(results_df, kupiec_summary, include_double_heston)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(table))
    width = 0.35
    ax.bar(x - width / 2, table["vol_RMSE_calm"], width, label="Calm regime", color="steelblue")
    ax.bar(x + width / 2, table["vol_RMSE_turbulent"], width, label="Turbulent regime", color="firebrick")

    ax.set_xticks(x)
    ax.set_xticklabels(table["model"])
    ax.set_ylabel("Volatility forecast RMSE")
    ax.set_title("Out-of-sample volatility forecast RMSE: calm vs turbulent regimes")
    ax.legend()
    return ax


def plot_var_breach_comparison(results_df, kupiec_summary, include_double_heston=True, ax=None):
    """Bar chart of VaR breach rates vs the target rate, one bar per
    model -- visualizes the Kupiec coverage test results directly."""
    table = summary_table(results_df, kupiec_summary, include_double_heston)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(table))
    colors = ["crimson" if rej else "seagreen" for rej in table["kupiec_rejected"]]
    ax.bar(x, table["VaR_breach_rate"] * 100, color=colors)
    ax.axhline(table["VaR_target_rate"].iloc[0] * 100, color="black", ls="--",
               label=f"Target rate ({table['VaR_target_rate'].iloc[0]*100:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"])
    ax.set_ylabel("Observed VaR breach rate (%)")
    ax.set_title("VaR backtest: observed breach rate vs target (red = Kupiec-rejected at 5%)")
    ax.legend()
    return ax
