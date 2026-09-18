"""
trading/strategies.py
========================
Multi-leg options strategies: payoff diagrams, entry cost, breakevens,
max profit/loss, and aggregated Greeks -- built on top of any pricer
(Black-Scholes or Heston) via a small common interface.

DESIGN: a strategy is just a list of "legs":
    (strike, option_type, qty)
where qty is signed: +1 = long one contract, -1 = short one contract
(qty is in UNITS OF ONE OPTION on one share of underlying -- multiply by
100 yourself for the standard "1 contract = 100 shares" convention if you
want dollar P&L on real US-listed contracts; this module works per-share
throughout, matching bs_price/heston_price's own per-share convention).

A "pricer" is any callable pricer(K, T, option_type) -> price per share.
Use bs_pricer(...) or heston_pricer(...) below to build one closed over
the spot/rate/model parameters -- this is what lets the exact same
strategy-construction code price a straddle under either model.
"""

import numpy as np
from scipy.optimize import brentq

from models.black_scholes import bs_price
from models.heston import heston_price
from trading.greeks import bs_greeks, heston_greeks


# ---------------------------------------------------------------------------
# Pricer factories
# ---------------------------------------------------------------------------

def bs_pricer(S, r, sigma, q=0.0):
    """Return pricer(K, T, option_type) -> Black-Scholes price."""
    def pricer(K, T, option_type):
        return bs_price(S, K, T, r, sigma, q, option_type)
    return pricer


def heston_pricer(S, r, q, kappa, theta, xi, rho, v0):
    """Return pricer(K, T, option_type) -> Heston price, using a fixed
    calibrated parameter set (e.g. from calibrate_heston_gmm)."""
    def pricer(K, T, option_type):
        return heston_price(S, K, T, r, q, kappa, theta, xi, rho, v0, option_type)
    return pricer


def bs_greeks_fn(S, r, sigma, q=0.0):
    """Return greeks_fn(K, T, option_type) -> dict, for strategy_greeks()."""
    def fn(K, T, option_type):
        return bs_greeks(S, K, T, r, sigma, q, option_type)
    return fn


def heston_greeks_fn(S, r, q, kappa, theta, xi, rho, v0):
    def fn(K, T, option_type):
        return heston_greeks(S, K, T, r, q, kappa, theta, xi, rho, v0, option_type)
    return fn


# ---------------------------------------------------------------------------
# Strategy builders -- each returns a list of (strike, option_type, qty)
# ---------------------------------------------------------------------------

def straddle(K, long=True):
    """Long/short straddle: same strike, one call + one put, same side.
    Bet on a big move (long) or on the underlying staying pinned (short)."""
    sign = 1 if long else -1
    return [(K, "call", sign), (K, "put", sign)]


def strangle(K_put, K_call, long=True):
    """Long/short strangle: OTM put + OTM call (K_put < K_call). Cheaper
    than a straddle (long) / less premium but tighter breakevens (short)."""
    if K_put >= K_call:
        raise ValueError("strangle requires K_put < K_call")
    sign = 1 if long else -1
    return [(K_put, "put", sign), (K_call, "call", sign)]


def vertical_spread(K_long, K_short, option_type="call"):
    """
    Vertical (debit/credit) spread: long one strike, short another, same
    expiry, same option type.
      - Bull call spread: option_type='call', K_long < K_short (net debit,
        capped upside, defined max loss = the debit paid).
      - Bear put spread: option_type='put', K_long > K_short (net debit,
        profits as price falls).
    Caps both max profit and max loss relative to an outright long option
    -- the whole point of the structure.
    """
    return [(K_long, option_type, 1), (K_short, option_type, -1)]


def iron_condor(K_put_long, K_put_short, K_call_short, K_call_long):
    """
    Iron condor: sell a put spread + sell a call spread, strikes ordered
        K_put_long < K_put_short < K_call_short < K_call_long
    Collects a net credit; profits if the underlying stays between the two
    short strikes through expiry; max loss is capped by the long "wings".
    A defined-risk way to sell volatility (the strategy this project's VRP
    backtest in trading/vrp_backtest.py approximates with a straddle for
    tractability -- an iron condor is the version a real desk would prefer
    for its capped downside).
    """
    if not (K_put_long < K_put_short < K_call_short < K_call_long):
        raise ValueError("iron_condor requires K_put_long < K_put_short < K_call_short < K_call_long")
    return [
        (K_put_long, "put", 1), (K_put_short, "put", -1),
        (K_call_short, "call", -1), (K_call_long, "call", 1),
    ]


# ---------------------------------------------------------------------------
# Evaluation: entry cost, payoff, breakevens, Greeks
# ---------------------------------------------------------------------------

def _leg_intrinsic(S_T, K, option_type):
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def entry_cost(legs, T, pricer):
    """
    Net cash paid (positive) or received (negative, i.e. a credit) to put
    the strategy on today, per share. Sums qty * price over every leg,
    using `pricer(K, T, option_type)`.
    """
    return float(sum(qty * pricer(K, T, option_type) for K, option_type, qty in legs))


def payoff_at_expiry(legs, S_T, cost):
    """
    P&L per share at expiry, for a scalar or array of terminal prices S_T:
        sum_legs( qty * intrinsic_value(S_T) )  -  entry_cost
    (entry_cost already carries the right sign: a credit strategy has
    cost < 0, which ADDS to P&L here, exactly as it should -- money
    received up front is profit unless given back at expiry.)
    """
    S_T = np.asarray(S_T, dtype=float)
    total = np.zeros_like(S_T)
    for K, option_type, qty in legs:
        total = total + qty * _leg_intrinsic(S_T, K, option_type)
    return total - cost


def breakevens(legs, cost, S_lo, S_hi, n_grid=2000):
    """
    Find all breakeven prices (payoff_at_expiry == 0) between S_lo and
    S_hi, by scanning a fine grid for sign changes and root-finding each
    one with brentq (the payoff function is piecewise LINEAR in S_T, so
    this is exact up to grid resolution -- a bracket always contains the
    true root if the grid is fine enough to catch the sign change, which
    n_grid=2000 comfortably is for any sane strike range).
    """
    grid = np.linspace(S_lo, S_hi, n_grid)
    vals = payoff_at_expiry(legs, grid, cost)

    roots = []
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            roots.append(grid[i])
        elif vals[i] * vals[i + 1] < 0:
            root = brentq(lambda s: payoff_at_expiry(legs, s, cost), grid[i], grid[i + 1])
            roots.append(root)
    return sorted(set(round(r, 4) for r in roots))


def strategy_summary(legs, T, pricer, S0, price_range=(0.5, 1.5), n_grid=2000):
    """
    One-call summary of a strategy: entry cost, breakevens, max
    profit/loss over the scanned price range, and the payoff curve itself
    (for plotting).

    NOTE on max profit/loss: for strategies with an unbounded leg (e.g. a
    long straddle's upside, or a naked short call), "max profit"/"max
    loss" over a FINITE scanned range is a lower/upper bound, not the true
    unbounded value -- this is flagged in the returned dict
    (`profit_uncapped`, `loss_uncapped`) by checking whether the payoff is
    still trending away from zero at the edges of the scanned range.
    """
    cost = entry_cost(legs, T, pricer)
    S_lo, S_hi = S0 * price_range[0], S0 * price_range[1]
    S_grid = np.linspace(S_lo, S_hi, n_grid)
    payoff = payoff_at_expiry(legs, S_grid, cost)

    bes = breakevens(legs, cost, S_lo, S_hi, n_grid)
    max_profit = float(payoff.max())
    max_loss = float(payoff.min())

    # Uncapped-at-the-edges check: is the payoff still moving away from
    # zero at either boundary of the scanned range? (A capped strategy's
    # payoff is flat at the boundaries; an uncapped one keeps sloping.)
    edge_slope_right = payoff[-1] - payoff[-5]
    edge_slope_left = payoff[4] - payoff[0]
    profit_uncapped = (payoff[-1] == max_profit and edge_slope_right > 1e-6) or \
                       (payoff[0] == max_profit and edge_slope_left < -1e-6)
    loss_uncapped = (payoff[-1] == max_loss and edge_slope_right < -1e-6) or \
                     (payoff[0] == max_loss and edge_slope_left > 1e-6)

    return {
        "legs": legs, "entry_cost": cost, "is_credit": cost < 0,
        "breakevens": bes, "max_profit": max_profit, "max_loss": max_loss,
        "profit_uncapped": profit_uncapped, "loss_uncapped": loss_uncapped,
        "S_grid": S_grid, "payoff": payoff,
    }


def strategy_greeks(legs, T, greeks_fn):
    """
    Aggregate Greeks across every leg: sum(qty * per-share greek).
    `greeks_fn(K, T, option_type)` -> dict (from trading/greeks.py's
    bs_greeks_fn / heston_greeks_fn closures).
    """
    total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0,
              "theta_daily": 0.0, "rho": 0.0}
    for K, option_type, qty in legs:
        g = greeks_fn(K, T, option_type)
        for key in total:
            total[key] += qty * g[key]
    return total
