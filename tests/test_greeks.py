"""
tests/test_greeks.py
=======================
Correctness checks for trading/greeks.py: Black-Scholes closed-form
Greeks cross-checked against finite differences of bs_price (an
independent sanity check that doesn't reuse the same formulas), and
Heston's finite-difference Greeks cross-checked against the Black-Scholes
closed form in the constant-volatility limit (xi -> 0, v0 = theta), where
Heston degenerates to GBM/Black-Scholes and the two should agree closely.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.black_scholes import bs_price
from trading.greeks import bs_greeks, heston_greeks


def test_bs_delta_put_call_relationship():
    """delta_call - delta_put == e^{-qT} (a direct consequence of put-call
    parity: C - P = S e^{-qT} - K e^{-rT}, differentiate both sides w.r.t. S)."""
    S, K, T, r, sigma, q = 100.0, 105.0, 0.5, 0.03, 0.25, 0.015
    call = bs_greeks(S, K, T, r, sigma, q, "call")
    put = bs_greeks(S, K, T, r, sigma, q, "put")
    expected = np.exp(-q * T)
    assert abs((call["delta"] - put["delta"]) - expected) < 1e-10
    print(f"[OK] BS delta_call - delta_put = {call['delta']-put['delta']:.6f} == e^-qT = {expected:.6f}")


def test_bs_gamma_vega_identical_for_call_and_put():
    """Gamma and vega are identical for a call and put at the same
    strike/expiry (a standard put-call-parity consequence, see
    models/black_scholes.bs_vega's docstring)."""
    S, K, T, r, sigma, q = 100.0, 95.0, 1.0, 0.04, 0.3, 0.0
    call = bs_greeks(S, K, T, r, sigma, q, "call")
    put = bs_greeks(S, K, T, r, sigma, q, "put")
    assert abs(call["gamma"] - put["gamma"]) < 1e-10
    assert abs(call["vega"] - put["vega"]) < 1e-10
    print(f"[OK] BS gamma/vega match for call and put: gamma={call['gamma']:.6f}, vega={call['vega']:.4f}")


def test_bs_delta_matches_finite_difference():
    """Cross-check the closed-form delta against a central finite
    difference of bs_price itself -- an independent check that doesn't
    share any code path with bs_greeks()."""
    S, K, T, r, sigma, q = 100.0, 100.0, 0.75, 0.035, 0.22, 0.01
    analytic = bs_greeks(S, K, T, r, sigma, q, "call")["delta"]
    h = 1e-4 * S
    fd = (bs_price(S + h, K, T, r, sigma, q, "call") - bs_price(S - h, K, T, r, sigma, q, "call")) / (2 * h)
    assert abs(analytic - fd) < 1e-4
    print(f"[OK] BS delta closed-form={analytic:.6f} matches finite-difference={fd:.6f}")


def test_bs_gamma_matches_finite_difference():
    S, K, T, r, sigma, q = 100.0, 100.0, 0.75, 0.035, 0.22, 0.01
    analytic = bs_greeks(S, K, T, r, sigma, q, "call")["gamma"]
    h = 1e-3 * S
    p_up = bs_price(S + h, K, T, r, sigma, q, "call")
    p0 = bs_price(S, K, T, r, sigma, q, "call")
    p_dn = bs_price(S - h, K, T, r, sigma, q, "call")
    fd = (p_up - 2 * p0 + p_dn) / h**2
    assert abs(analytic - fd) < 1e-3
    print(f"[OK] BS gamma closed-form={analytic:.6f} matches finite-difference={fd:.6f}")


def test_bs_degenerate_T_zero_returns_intrinsic_delta():
    itm_call = bs_greeks(110.0, 100.0, 0.0, 0.03, 0.2, 0.0, "call")
    otm_call = bs_greeks(90.0, 100.0, 0.0, 0.03, 0.2, 0.0, "call")
    assert itm_call["delta"] == 1.0 and itm_call["gamma"] == 0.0
    assert otm_call["delta"] == 0.0
    print("[OK] T=0 degenerate limit: delta collapses to 0/1, other Greeks to 0")


def test_heston_greeks_matches_bs_in_constant_vol_limit():
    """With xi -> 0 (variance barely moves) and v0 = theta (already at
    its long-run level), Heston's variance process is effectively
    constant -- so its price and Greeks should closely match Black-Scholes
    with sigma = sqrt(theta). This is the key validation for the
    finite-difference Heston Greeks, since Heston has no independent
    closed-form Greeks to check against directly."""
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.03, 0.0
    kappa, theta, xi, rho, v0 = 3.0, 0.04, 0.001, 0.0, 0.04
    sigma = np.sqrt(theta)

    hg = heston_greeks(S, K, T, r, q, kappa, theta, xi, rho, v0, "call")
    bg = bs_greeks(S, K, T, r, sigma, q, "call")

    assert abs(hg["price"] - bg["price"]) < 0.05
    assert abs(hg["delta"] - bg["delta"]) < 0.01
    assert abs(hg["gamma"] - bg["gamma"]) < 0.005
    print(f"[OK] Heston Greeks (xi->0, v0=theta) match BS: "
          f"delta {hg['delta']:.4f} vs {bg['delta']:.4f}, "
          f"gamma {hg['gamma']:.5f} vs {bg['gamma']:.5f}")


def test_heston_greeks_price_matches_heston_price():
    """heston_greeks()['price'] must exactly match a direct call to
    heston_price() with the same parameters (both should use the same
    underlying pricer)."""
    from models.heston import heston_price
    S, K, T, r, q = 100.0, 105.0, 1.0, 0.03, 0.01
    kappa, theta, xi, rho, v0 = 2.0, 0.05, 0.4, -0.5, 0.06
    direct = heston_price(S, K, T, r, q, kappa, theta, xi, rho, v0, "call")
    via_greeks = heston_greeks(S, K, T, r, q, kappa, theta, xi, rho, v0, "call")["price"]
    assert abs(direct - via_greeks) < 1e-9
    print(f"[OK] heston_greeks price ({via_greeks:.6f}) matches heston_price directly ({direct:.6f})")


if __name__ == "__main__":
    test_bs_delta_put_call_relationship()
    test_bs_gamma_vega_identical_for_call_and_put()
    test_bs_delta_matches_finite_difference()
    test_bs_gamma_matches_finite_difference()
    test_bs_degenerate_T_zero_returns_intrinsic_delta()
    test_heston_greeks_matches_bs_in_constant_vol_limit()
    test_heston_greeks_price_matches_heston_price()
    print("\nAll Greeks tests passed.")
