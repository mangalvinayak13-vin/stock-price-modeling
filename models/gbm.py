"""
models/gbm.py
==============
Geometric Brownian Motion (GBM): the model behind the Black-Scholes world.

The SDE is:
    dS(t) = mu * S(t) dt + sigma * S(t) dW(t)

where W(t) is a standard Brownian motion (Wiener process). Applying Ito's
lemma to X(t) = ln S(t) gives the EXACT (not approximate) solution:

    S(t) = S(0) * exp( (mu - sigma^2 / 2) * t + sigma * W(t) )

We simulate this directly using the exact solution rather than an Euler
discretization of the SDE. This matters because Euler-Maruyama on GBM has
O(sqrt(dt)) discretization error and can even (for large steps) produce a
qualitatively wrong path; the exact solution above has ZERO discretization
error at the simulated time points -- the only randomness is from sampling
the Brownian increments, not from approximating the dynamics.
"""

import numpy as np


def simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=None):
    """
    Simulate `n_paths` sample paths of GBM over [0, T] using the exact
    solution S(t) = S0 * exp((mu - sigma^2/2)*t + sigma*W(t)).

    Parameters
    ----------
    S0      : float, initial price
    mu      : float, annualized drift (real-world drift; use r for
              risk-neutral simulation if that's ever needed elsewhere)
    sigma   : float, annualized volatility
    T       : float, time horizon in years
    n_steps : int, number of time steps (path will have n_steps+1 points
              including t=0)
    n_paths : int, number of independent sample paths
    seed    : int, optional RNG seed for reproducibility

    Returns
    -------
    t : np.ndarray, shape (n_steps+1,), the time grid
    S : np.ndarray, shape (n_paths, n_steps+1), simulated price paths
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    # Brownian increments: dW_i ~ N(0, dt), independent across steps.
    # W(t_k) = sum_{i<=k} dW_i is then exact (not approximate) Brownian
    # motion sampled at the grid points, since Gaussian increments are the
    # TRUE distribution of Wiener process increments (not a discretization).
    dW = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)

    drift = (mu - 0.5 * sigma**2) * t  # shape (n_steps+1,), broadcasts over paths
    S = S0 * np.exp(drift[np.newaxis, :] + sigma * W)

    return t, S


def estimate_gbm_params(log_returns, trading_days_per_year=252):
    """
    Estimate GBM's (mu, sigma) from a series of historical DAILY log
    returns, annualized.

    Under GBM, log returns over a small step dt are i.i.d. Normal with
        mean = (mu - sigma^2/2) * dt,   variance = sigma^2 * dt.
    So from the sample mean and sample std of daily log returns
    (mean_hat, std_hat) with dt = 1/252:

        sigma_hat = std_hat * sqrt(252)                     (annualized vol)
        mu_hat    = mean_hat * 252 + sigma_hat^2 / 2         (annualized drift,
                                                               correcting for
                                                               the -sigma^2/2
                                                               Ito term)

    JUDGMENT CALL: we annualize using 252 trading days/year (the standard
    convention in equity markets), not 365 calendar days -- option markets
    and most textbook treatments (e.g. Hull) use 252 for equities.

    Returns
    -------
    mu_hat, sigma_hat : floats, annualized drift and volatility
    """
    mean_hat = np.mean(log_returns)
    std_hat = np.std(log_returns, ddof=1)  # ddof=1: unbiased sample variance

    sigma_hat = std_hat * np.sqrt(trading_days_per_year)
    mu_hat = mean_hat * trading_days_per_year + 0.5 * sigma_hat**2

    return mu_hat, sigma_hat
