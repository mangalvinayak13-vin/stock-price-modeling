"""
analysis/gbm_diagnostics.py
=============================
Diagnostic plots for the GBM module. The point of these plots (for the
report) is twofold:
  1. Show that the exact-solution simulator works and looks like real
     stock-path behavior.
  2. Visually motivate WHY the constant-sigma assumption of GBM /
     Black-Scholes is unrealistic -- this sets up the rest of the report
     (Heston, Double Heston) which relax exactly this assumption.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_simulated_paths(t, S, n_show=30, ax=None, title="Simulated GBM Paths"):
    """Plot a subset of simulated GBM sample paths against time."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    n_show = min(n_show, S.shape[0])
    for i in range(n_show):
        ax.plot(t, S[i], lw=0.8, alpha=0.7)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price")
    ax.set_title(title)
    return ax


def plot_return_histogram(log_returns, ax=None, title="Log Returns vs Fitted Normal"):
    """
    Histogram of historical daily log returns overlaid with the Normal PDF
    fitted to the same sample mean/std -- this is the distribution GBM
    assumes log returns follow. Visually, real return histograms usually
    show fatter tails than the fitted Normal (excess kurtosis), which is a
    first hint that GBM understates the probability of large moves.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    mu_hat, std_hat = np.mean(log_returns), np.std(log_returns, ddof=1)
    ax.hist(log_returns, bins=60, density=True, alpha=0.6, color="steelblue", label="Empirical")

    x = np.linspace(log_returns.min(), log_returns.max(), 400)
    ax.plot(x, stats.norm.pdf(x, mu_hat, std_hat), "r-", lw=2, label="Fitted Normal")

    ax.set_xlabel("Daily log return")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_qq(log_returns, ax=None, title="QQ Plot vs Normal"):
    """
    QQ-plot of standardized log returns against the standard Normal
    quantiles. Points that curve away from the y=x reference line at the
    tails indicate the "fat tails" real returns have relative to the
    Normal distribution GBM assumes -- another motivation for stochastic
    volatility models.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(log_returns, dist="norm", plot=ax)
    ax.set_title(title)
    return ax


def plot_rolling_volatility(log_returns, window=21, trading_days_per_year=252, ax=None,
                             title="Rolling Realized Volatility (Annualized)"):
    """
    Rolling `window`-day realized volatility (annualized), plotted over
    time. This is the key motivating plot for the whole project: if sigma
    were truly constant (as GBM/Black-Scholes assume), this line would be
    flat. In real data it visibly clusters and varies over time (volatility
    clustering, per Mandelbrot/Engle) -- exactly what Heston's mean-
    reverting stochastic variance process is designed to capture.

    JUDGMENT CALL: window=21 trading days (~1 calendar month) is a standard
    choice for a "short-term realized vol" window -- short enough to show
    time-variation, long enough to not be pure noise from a handful of
    returns.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    rolling_std = log_returns.rolling(window=window).std()
    rolling_vol = rolling_std * np.sqrt(trading_days_per_year)

    ax.plot(rolling_vol.index, rolling_vol.values, color="darkorange", lw=1.2)
    ax.axhline(log_returns.std() * np.sqrt(trading_days_per_year), color="gray",
               ls="--", lw=1, label="Full-sample constant sigma")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility")
    ax.set_title(title)
    ax.legend()
    return ax


def gbm_diagnostic_panel(t, S, log_returns, window=21):
    """
    Convenience function: produces the full 2x2 diagnostic panel used in
    the report (simulated paths, return histogram, QQ-plot, rolling vol).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_simulated_paths(t, S, ax=axes[0, 0])
    plot_return_histogram(log_returns, ax=axes[0, 1])
    plot_qq(log_returns, ax=axes[1, 0])
    plot_rolling_volatility(log_returns, window=window, ax=axes[1, 1])
    fig.tight_layout()
    return fig
