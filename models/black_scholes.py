"""
models/black_scholes.py
=========================
The Black-Scholes-Merton (1973) closed-form European option pricer, plus
an implied volatility solver.

Model assumptions (all of which the later Heston/Double Heston modules
relax): the underlying follows GBM with CONSTANT volatility sigma under
the risk-neutral measure, no arbitrage, continuous trading, and (here) a
constant continuous dividend yield q (the "Merton" extension of the
original 1973 paper, which assumed q=0).

Risk-neutral price of a European call:
    C = S * e^{-qT} * N(d1) - K * e^{-rT} * N(d2)
and a European put (via put-call parity, or the symmetric formula):
    P = K * e^{-rT} * N(-d2) - S * e^{-qT} * N(-d1)
where
    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
and N(.) is the standard normal CDF.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """Compute d1, d2 for the Black-Scholes formula (shared by call/put)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Black-Scholes-Merton European option price.

    Parameters
    ----------
    S : float, current underlying price
    K : float, strike
    T : float, time to expiry in years
    r : float, continuously-compounded risk-free rate
    sigma : float, volatility (annualized)
    q : float, continuous dividend yield
    option_type : 'call' or 'put'

    Returns
    -------
    float, theoretical option price

    Edge case handling: as T -> 0 or sigma -> 0, d1/d2 blow up (division
    by ~0). We handle this by returning the option's intrinsic value in
    that limit, which is the correct economic answer (an option an instant
    before expiry, or with zero volatility, is worth exactly its payoff
    discounted by an infinitesimal amount).
    """
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_vega(S, K, T, r, sigma, q=0.0):
    """
    Vega: dPrice/dSigma, the sensitivity of the option price to volatility.
    Same formula for calls and puts (vega is identical for a call and put
    at the same strike/expiry, a consequence of put-call parity since
    d(Put)/dSigma - d(Call)/dSigma = d/dSigma[discount*(K-S)] = 0).

        vega = S * e^{-qT} * phi(d1) * sqrt(T)

    where phi is the standard normal PDF. Used as the derivative in the
    Newton-Raphson implied-vol solver below.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_volatility(price, S, K, T, r, q=0.0, option_type="call",
                        sigma_init=0.3, tol=1e-6, max_iter=100,
                        bracket=(1e-4, 5.0)):
    """
    Solve for the Black-Scholes implied volatility sigma such that
    bs_price(S, K, T, r, sigma, q, option_type) == price.

    METHOD: Newton-Raphson, with a bisection fallback for robustness.

    WHY BOTH METHODS (this is a real numerical issue, not over-engineering):
    Newton-Raphson update is  sigma_{n+1} = sigma_n - (f(sigma_n) / vega(sigma_n)).
    This converges fast (quadratically) when it works, but it can fail in
    two well-known ways:
      1. Vega -> 0: this happens for deep ITM/OTM options and for very
         short-dated options, where the price is nearly flat in sigma over
         a wide range -- dividing by a near-zero vega sends the next
         iterate flying off to a nonsensical (even negative) sigma.
      2. Non-convergence / oscillation: a poor starting guess can overshoot
         into a region where the local derivative sends it further away
         rather than closer, especially for options far from ATM.

    So we run Newton-Raphson, but after every step reject any iterate that
    leaves the valid vol range (bracket) or fails to improve, and fall back
    to bisection -- which is slower (linear convergence) but mathematically
    guaranteed to converge as long as a sign change exists in the bracket,
    since bs_price(sigma) is monotonically increasing in sigma (vega >= 0
    always) so price - target has at most one root in the bracket.

    Parameters
    ----------
    price : float, observed market price (mid) to invert
    bracket : (low, high) tuple, sigma search range for bisection fallback.
        JUDGMENT CALL: (0.0001, 5.0) i.e. 0.01% to 500% annualized vol.
        This is deliberately wide -- real equities rarely exceed 300% vol
        even in extreme stress, but a wide bracket costs nothing extra for
        bisection (log2 iterations to a given tolerance) and guarantees we
        don't accidentally exclude a legitimate high-vol short-dated quote.

    Returns
    -------
    float, implied volatility, or np.nan if no solution exists in the
    bracket (e.g. the quoted price violates a no-arbitrage bound and is
    below intrinsic value or above the model's max attainable price --
    this can happen with noisy/stale quotes and should be filtered
    upstream, but we return nan rather than raising so a batch smile
    computation can skip bad points instead of crashing).
    """
    if T <= 0 or price <= 0:
        return np.nan

    # No-arbitrage bounds check: if the market price is outside what ANY
    # sigma (including sigma->0 or sigma->infinity) can produce, there is
    # no solution -- catch this up front rather than let the root-finders
    # fail silently or loop to max_iter.
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0) if option_type == "call" \
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    upper_bound = S * np.exp(-q * T) if option_type == "call" else K * np.exp(-r * T)
    if price < intrinsic - 1e-8 or price > upper_bound + 1e-8:
        return np.nan

    def f(sigma):
        return bs_price(S, K, T, r, sigma, q, option_type) - price

    # --- Try Newton-Raphson first ---
    sigma = sigma_init
    newton_converged = False
    for _ in range(max_iter):
        price_diff = f(sigma)
        if abs(price_diff) < tol:
            newton_converged = True
            break

        vega = bs_vega(S, K, T, r, sigma, q)
        if vega < 1e-8:
            break  # vega too small -- bail out to bisection below

        sigma_next = sigma - price_diff / vega

        # Reject a step that leaves the sane search range: this is exactly
        # the failure mode described above (near-zero vega or a bad
        # starting point sending sigma negative or absurdly large).
        if sigma_next <= bracket[0] or sigma_next >= bracket[1] or not np.isfinite(sigma_next):
            break

        sigma = sigma_next

    if newton_converged:
        return sigma

    # --- Bisection fallback (Newton either didn't converge within
    # max_iter, or broke out early due to a bad step) ---
    lo, hi = bracket
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        # No sign change in the bracket -> price is not attainable by any
        # sigma in [lo, hi] under this model (shouldn't happen given the
        # no-arbitrage check above, but guards against pathological inputs).
        return np.nan

    for _ in range(200):  # 200 iterations is vastly more than needed for
                           # double-precision convergence on a bracket this
                           # wide (log2((5-0.0001)/1e-10) ~ 55 iterations),
                           # kept generous since bisection is cheap per step.
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)
