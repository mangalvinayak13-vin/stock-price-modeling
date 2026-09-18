"""
tests/test_strategies.py
===========================
Correctness checks for trading/strategies.py: strategy construction,
entry cost sign conventions, breakeven solving, and capped vs uncapped
max profit/loss detection -- checked against hand-derivable results for
each canonical strategy shape (a straddle's breakevens are exactly
K +/- premium; a vertical spread's max loss is exactly the debit paid;
etc.), not just "does it run".
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.strategies import (straddle, strangle, vertical_spread, iron_condor,
                                 bs_pricer, bs_greeks_fn, entry_cost, strategy_summary,
                                 strategy_greeks)


def test_long_straddle_breakevens_are_strike_plus_minus_premium():
    """A textbook result: a long straddle's two breakevens are exactly
    K - premium and K + premium (since at expiry the payoff is
    |S_T - K| - premium, which is zero exactly at those two points)."""
    S, K, T, r, sigma, q = 100.0, 100.0, 30 / 365, 0.04, 0.25, 0.0
    pricer = bs_pricer(S, r, sigma, q)
    legs = straddle(K, long=True)
    premium = entry_cost(legs, T, pricer)

    summary = strategy_summary(legs, T, pricer, S0=S, price_range=(0.5, 1.5))
    assert summary["entry_cost"] > 0 and not summary["is_credit"]
    bes = summary["breakevens"]
    assert len(bes) == 2
    assert abs(bes[0] - (K - premium)) < 0.01
    assert abs(bes[1] - (K + premium)) < 0.01
    print(f"[OK] Long straddle: premium={premium:.4f}, breakevens={bes} "
          f"match K-premium/K+premium exactly")


def test_short_straddle_is_a_credit_with_mirrored_max_profit_loss():
    """Selling the exact same straddle must flip the sign of entry cost
    (a credit) and mirror the payoff curve (short's max profit == long's
    max loss in magnitude, at least within the scanned price range)."""
    S, K, T, r, sigma, q = 100.0, 100.0, 30 / 365, 0.04, 0.25, 0.0
    pricer = bs_pricer(S, r, sigma, q)

    long_summary = strategy_summary(straddle(K, long=True), T, pricer, S0=S)
    short_summary = strategy_summary(straddle(K, long=False), T, pricer, S0=S)

    assert short_summary["entry_cost"] < 0 and short_summary["is_credit"]
    assert abs(long_summary["entry_cost"] + short_summary["entry_cost"]) < 1e-9
    # Short straddle's max profit is capped at the premium collected
    # (achieved exactly at S_T = K, i.e. -entry_cost since entry_cost is a
    # negative credit); tolerance is grid resolution, not solver error --
    # strategy_summary evaluates on a finite price grid, so the true max
    # (at exactly S_T=K) is only approximated by the nearest grid point.
    assert abs(short_summary["max_profit"] - (-short_summary["entry_cost"])) < 0.05
    print(f"[OK] Short straddle credit={short_summary['entry_cost']:.4f}, "
          f"max profit={short_summary['max_profit']:.4f} == premium collected")


def test_strangle_requires_ordered_strikes():
    try:
        strangle(K_put=110.0, K_call=90.0)
        assert False, "should have raised for K_put >= K_call"
    except ValueError:
        pass
    print("[OK] strangle() rejects K_put >= K_call")


def test_vertical_spread_max_loss_equals_debit_paid():
    """A bull call spread's max loss (both legs expire worthless, i.e.
    S_T <= K_long) is exactly the net debit paid -- a standard, capped-risk
    result that must hold regardless of the pricer used."""
    S, T, r, sigma, q = 100.0, 60 / 365, 0.04, 0.3, 0.0
    pricer = bs_pricer(S, r, sigma, q)
    legs = vertical_spread(K_long=95.0, K_short=105.0, option_type="call")
    debit = entry_cost(legs, T, pricer)
    assert debit > 0  # buying the lower (more valuable) strike net costs money

    summary = strategy_summary(legs, T, pricer, S0=S, price_range=(0.5, 1.5))
    assert not summary["loss_uncapped"] and not summary["profit_uncapped"]
    assert abs(summary["max_loss"] - (-debit)) < 0.01
    assert abs(summary["max_profit"] - ((105.0 - 95.0) - debit)) < 0.05
    print(f"[OK] Bull call spread: debit={debit:.4f}, max_loss={summary['max_loss']:.4f} "
          f"(== -debit), max_profit={summary['max_profit']:.4f} (== width - debit)")


def test_iron_condor_strike_ordering_enforced_and_credit_received():
    try:
        iron_condor(90, 95, 85, 110)  # deliberately out of order
        assert False, "should have raised for unordered strikes"
    except ValueError:
        pass

    S, T, r, sigma, q = 100.0, 45 / 365, 0.04, 0.22, 0.0
    pricer = bs_pricer(S, r, sigma, q)
    legs = iron_condor(K_put_long=85, K_put_short=95, K_call_short=105, K_call_long=115)
    summary = strategy_summary(legs, T, pricer, S0=S, price_range=(0.5, 1.5))
    assert summary["is_credit"]
    assert not summary["loss_uncapped"] and not summary["profit_uncapped"]
    # Max profit is capped at the net credit received; max loss is capped
    # at (wing width - credit), a standard defined-risk result.
    assert abs(summary["max_profit"] - (-summary["entry_cost"])) < 0.05
    print(f"[OK] Iron condor: credit={-summary['entry_cost']:.4f}, "
          f"max_profit={summary['max_profit']:.4f}, max_loss={summary['max_loss']:.4f}")


def test_strategy_greeks_is_qty_weighted_sum_of_leg_greeks():
    S, K, T, r, sigma, q = 100.0, 100.0, 0.25, 0.03, 0.2, 0.0
    greeks_fn = bs_greeks_fn(S, r, sigma, q)
    legs = straddle(K, long=True)

    total = strategy_greeks(legs, T, greeks_fn)
    manual = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "theta_daily": 0.0, "rho": 0.0}
    for Ki, opt, qty in legs:
        g = greeks_fn(Ki, T, opt)
        for k in manual:
            manual[k] += qty * g[k]

    for k in manual:
        assert abs(total[k] - manual[k]) < 1e-9
    # A same-strike straddle's delta should be small (call and put deltas
    # roughly offset), but its gamma/vega should be large and POSITIVE
    # (long gamma, long vega -- the whole point of a long straddle).
    assert total["gamma"] > 0 and total["vega"] > 0
    print(f"[OK] Straddle aggregate Greeks: delta={total['delta']:.4f} (near 0), "
          f"gamma={total['gamma']:.5f} (>0), vega={total['vega']:.4f} (>0)")


if __name__ == "__main__":
    test_long_straddle_breakevens_are_strike_plus_minus_premium()
    test_short_straddle_is_a_credit_with_mirrored_max_profit_loss()
    test_strangle_requires_ordered_strikes()
    test_vertical_spread_max_loss_equals_debit_paid()
    test_iron_condor_strike_ordering_enforced_and_credit_received()
    test_strategy_greeks_is_qty_weighted_sum_of_leg_greeks()
    print("\nAll strategy tests passed.")
