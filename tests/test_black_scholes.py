"""
tests/test_black_scholes.py
=============================
Correctness checks for the Black-Scholes pricer and implied vol solver.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.black_scholes import bs_price, bs_vega, implied_volatility


def test_put_call_parity():
    """
    Put-call parity (no-arbitrage identity, holds regardless of the model
    used to price -- it's a static replication argument):
        C - P = S*e^{-qT} - K*e^{-rT}
    """
    S, K, T, r, sigma, q = 100.0, 95.0, 0.75, 0.03, 0.25, 0.01
    C = bs_price(S, K, T, r, sigma, q, "call")
    P = bs_price(S, K, T, r, sigma, q, "put")
    lhs = C - P
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-10, f"parity violated: {lhs} vs {rhs}"
    print(f"[OK] put-call parity: C-P={lhs:.6f}, S*e^-qT - K*e^-rT={rhs:.6f}")


def test_iv_roundtrip_wide_range_of_strikes_and_maturities():
    """
    The core robustness test: price options at many (K, T) combinations
    (including deep ITM/OTM and very short-dated, the cases known to break
    naive Newton-Raphson), then invert to recover the implied vol. It
    should recover the TRUE sigma used to generate the price, for every
    single case -- this validates the Newton+bisection fallback actually
    works where plain Newton would fail.

    ONE GENUINE EXCEPTION, not a solver bug: when vega is truly ~0 (deep
    ITM/OTM combined with very short time to expiry), the option's price
    is numerically FLAT in sigma -- price is indistinguishable from
    intrinsic value to machine precision across a wide range of sigma.
    In that regime implied vol is mathematically unidentifiable from
    price (this is a real, well-known phenomenon in options markets: you
    will never see a quoted/traded IV for a 60%-ITM option expiring
    tomorrow, for exactly this reason). We still require the solver to
    return SOME finite sigma (not nan/crash) and, crucially, that
    reproducing the price at that sigma matches the target price exactly
    -- we just don't require it to recover the specific true_sigma in
    that regime, since infinitely many sigmas are equally "correct".
    """
    S, r, q = 100.0, 0.04, 0.01
    true_sigma = 0.35
    # Below this vega, a fixed price tolerance of 1e-6 implies a sigma
    # uncertainty of price_tol/vega >= 1e-3 -- i.e. sigma is no longer
    # tightly pinned down by price, so we don't hold the solver to a tight
    # true_sigma match there (we still always require correct repricing).
    VEGA_FLOOR = 1e-3

    strikes = [40, 60, 80, 95, 100, 105, 120, 150, 200]         # deep ITM to deep OTM
    maturities = [1/365, 7/365, 30/365, 0.25, 1.0, 2.0]         # 1 day to 2 years

    max_err_identifiable = 0.0
    n_tested, n_low_vega = 0, 0
    for T in maturities:
        for K in strikes:
            for opt_type in ["call", "put"]:
                price = bs_price(S, K, T, r, true_sigma, q, opt_type)
                if price < 1e-8:
                    continue  # skip prices too close to zero to be numerically meaningful
                vega = bs_vega(S, K, T, r, true_sigma, q)
                iv = implied_volatility(price, S, K, T, r, q, opt_type)
                assert not np.isnan(iv), f"IV solver failed to converge: K={K}, T={T}, type={opt_type}"

                # The one thing that must ALWAYS hold, even in the
                # unidentifiable regime: re-pricing at the recovered iv
                # must reproduce the target price (i.e. we found *a* root).
                repriced = bs_price(S, K, T, r, iv, q, opt_type)
                assert abs(repriced - price) < 1e-4, \
                    f"solver returned a sigma that doesn't reprice correctly: K={K}, T={T}"

                n_tested += 1
                if vega < VEGA_FLOOR:
                    n_low_vega += 1
                    continue  # unidentifiable regime -- see docstring, don't check vs true_sigma

                err = abs(iv - true_sigma)
                max_err_identifiable = max(max_err_identifiable, err)
                assert err < 1e-4, \
                    f"IV mismatch at K={K}, T={T}, {opt_type}: got {iv}, true {true_sigma}"

    print(f"[OK] IV roundtrip: {n_tested} (K,T,type) combos tested "
          f"({n_low_vega} in the vega~0 unidentifiable regime, correctly excluded from the "
          f"true-sigma check but still verified to reprice correctly), "
          f"max abs error on identifiable cases = {max_err_identifiable:.2e}")


def test_iv_solver_near_zero_vega_deep_otm_short_dated():
    """
    Specifically stress-test the failure mode called out in the docstring:
    a deep OTM, very short-dated option has vega ~ 0 (the option is almost
    certain to expire worthless, so price is nearly flat in sigma). Plain
    Newton-Raphson is known to diverge here -- confirm our solver still
    returns a sane answer instead of nan/inf/crashing.
    """
    # 3-day, 20%-OTM call: vega ~0.015 here (checked numerically), small
    # enough to break plain Newton-Raphson (near-zero-vega regime) but
    # still large enough that implied vol IS well-identified from price
    # (unlike the far-OTM/deep-ITM cases in the roundtrip test above,
    # which are genuinely unidentifiable and excluded there for that
    # reason) -- this is the right case to test Newton-fails/bisection-saves.
    S, K, T, r, q = 100.0, 120.0, 3/365, 0.04, 0.0
    true_sigma = 0.6
    price = bs_price(S, K, T, r, true_sigma, q, "call")
    assert price > 0

    iv = implied_volatility(price, S, K, T, r, q, "call")
    assert not np.isnan(iv), "solver failed on near-zero-vega case"
    assert abs(iv - true_sigma) < 1e-3, f"got {iv}, expected close to {true_sigma}"
    print(f"[OK] near-zero-vega stress test: recovered sigma={iv:.5f} (true {true_sigma})")


def test_iv_no_solution_returns_nan_not_crash():
    """A price violating no-arbitrage bounds should return nan, not raise."""
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.03, 0.0
    # A call price below intrinsic value (S - K*e^-rT) is impossible.
    intrinsic = S - K * np.exp(-r * T)
    bad_price = intrinsic - 1.0
    iv = implied_volatility(bad_price, S, K, T, r, q, "call")
    assert np.isnan(iv)
    print(f"[OK] out-of-bounds price correctly returns nan")


if __name__ == "__main__":
    test_put_call_parity()
    test_iv_roundtrip_wide_range_of_strikes_and_maturities()
    test_iv_solver_near_zero_vega_deep_otm_short_dated()
    test_iv_no_solution_returns_nan_not_crash()
    print("\nAll Black-Scholes tests passed.")
