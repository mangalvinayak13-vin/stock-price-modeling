"""
trading/greeks.py
====================
Option Greeks: the sensitivities of an option's price to its inputs, used
throughout options trading for hedging and risk sizing.

    delta = dPrice/dS       -- shares of underlying the option behaves like
    gamma = d^2Price/dS^2   -- how fast delta itself changes
    vega  = dPrice/dSigma   -- sensitivity to volatility (already in
                               models/black_scholes.py as bs_vega; repeated
                               here for a single consistent Greeks API)
    theta = -dPrice/dT      -- time decay (price lost per year/day as time
                               passes, at fixed S/sigma; the minus sign is
                               a convention: theta is usually quoted as
                               "how much the option loses per day")
    rho   = dPrice/dr       -- sensitivity to the risk-free rate

BLACK-SCHOLES: all five have closed-form formulas (Hull, ch. 19) --
implemented directly, no numerical differentiation needed.

HESTON: no simple closed form for delta/gamma/theta/rho in terms of the
same P1/P2 integrals used for price (they exist via differentiating the
characteristic-function integral under the integral sign, but that adds a
second layer of numerical integration per Greek, which is unnecessary
complexity for a student project). JUDGMENT CALL: we instead use CENTRAL
FINITE DIFFERENCES on the validated heston_price() function itself --
bump each input by a small relative step, reprice, and divide. This is
slower than a closed form but exactly as accurate to O(h^2) as any
scheme needs to be for reporting/hedging-demo purposes, and it reuses the
already-tested pricer instead of introducing new, unverified analytic
formulas. Vega here is genuinely ambiguous for Heston (there is no single
volatility parameter) -- we report d(Price)/d(v0) instead, i.e.
sensitivity to a shock in TODAY's instantaneous variance, which is the
natural Heston analogue of "spot vol" and the one hedgers actually care
about intraday.
"""

import numpy as np
from scipy.stats import norm

from models.black_scholes import _d1_d2, bs_price, bs_vega
from models.heston import heston_price


# ---------------------------------------------------------------------------
# Black-Scholes Greeks (closed form)
# ---------------------------------------------------------------------------

def bs_greeks(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Full Black-Scholes Greeks for one option, per underlying share.

    Returns
    -------
    dict with keys: price, delta, gamma, vega, theta (per YEAR), theta_daily
    (per calendar day, theta/365 -- the number usually quoted by brokers),
    rho.
    """
    price = bs_price(S, K, T, r, sigma, q, option_type)

    if T <= 0 or sigma <= 0:
        # Degenerate limit: intrinsic value only, no time value left to decay
        # and delta collapses to 0/1 depending on moneyness -- gamma/vega/
        # theta/rho are all exactly zero (no sensitivity left to anything
        # but the payoff itself).
        if option_type == "call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {"price": price, "delta": delta, "gamma": 0.0, "vega": 0.0,
                "theta": 0.0, "theta_daily": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = bs_vega(S, K, T, r, sigma, q)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-(S * disc_q * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    elif option_type == "put":
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = (-(S * disc_q * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "theta_daily": theta / 365.0, "rho": rho}


# ---------------------------------------------------------------------------
# Heston Greeks (central finite differences on the validated CF pricer)
# ---------------------------------------------------------------------------

def heston_greeks(S0, K, T, r, q, kappa, theta_v, xi, rho, v0, option_type="call",
                   rel_step=1e-3):
    """
    Heston Greeks via central finite differences on heston_price().

    STEP SIZE (rel_step=1e-3, i.e. 0.1% relative bumps): small enough that
    the finite-difference truncation error (O(h^2) for central differences)
    is negligible, large enough to stay well clear of the numerical-
    integration noise floor of heston_price()'s own quad() call (an
    excessively tiny h would amplify that noise when divided by h). This
    was checked empirically (see tests/test_greeks.py) by confirming the
    finite-difference delta/gamma agree with the Black-Scholes closed form
    when Heston's parameters are collapsed to the constant-volatility limit
    (xi ~ 0, v0 = theta_v).

    Returns
    -------
    dict with keys: price, delta, gamma, vega (d/dv0), theta (per year,
    same 'lost value as T shrinks' sign convention as bs_greeks), theta_daily,
    rho.
    """
    def price(S_, T_, r_, v0_):
        return heston_price(S_, K, T_, r_, q, kappa, theta_v, xi, rho, v0_, option_type)

    p0 = price(S0, T, r, v0)

    # --- delta, gamma: bump S0 ---
    hS = rel_step * S0
    p_up, p_dn = price(S0 + hS, T, r, v0), price(S0 - hS, T, r, v0)
    delta = (p_up - p_dn) / (2 * hS)
    gamma = (p_up - 2 * p0 + p_dn) / (hS**2)

    # --- vega analogue: bump v0 (today's instantaneous variance) ---
    hv = max(rel_step * v0, 1e-6)
    vega = (price(S0, T, r, v0 + hv) - price(S0, T, r, v0 - hv)) / (2 * hv)

    # --- theta: bump T (theta = -dPrice/dT, so flip sign after the
    # forward-looking finite difference; use a one-sided step for T since
    # T-h could go to/below 0 for short-dated options) ---
    hT = min(rel_step * T, T * 0.25) if T > 0 else 1e-4
    hT = max(hT, 1e-5)
    theta = -(price(S0, T + hT, r, v0) - price(S0, max(T - hT, 1e-6), r, v0)) / (
        (T + hT) - max(T - hT, 1e-6))

    # --- rho: bump r ---
    hr = 1e-4
    rho_greek = (price(S0, T, r + hr, v0) - price(S0, T, r - hr, v0)) / (2 * hr)

    return {"price": p0, "delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "theta_daily": theta / 365.0, "rho": rho_greek}
