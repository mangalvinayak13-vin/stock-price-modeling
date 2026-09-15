"""
tests/test_double_heston.py
==============================
Correctness checks for Double Heston, mirroring tests/test_heston.py.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.double_heston import (double_heston_char_func, double_heston_price,
                                   double_heston_feller_conditions)
from models.double_heston_mc import simulate_double_heston_paths, double_heston_mc_price
from models.heston import heston_price


def test_double_heston_char_func_at_zero_is_one():
    val = double_heston_char_func(0.0, S0=100, v1_0=0.03, v2_0=0.02,
                                   kappa1=2.0, theta1=0.04, xi1=0.3, rho1=-0.6,
                                   kappa2=0.5, theta2=0.03, xi2=0.4, rho2=-0.4,
                                   r=0.03, q=0.0, T=1.0)
    assert abs(val - 1.0) < 1e-10
    print(f"[OK] Double Heston phi(0) = {val:.10f} (expected 1)")


def test_double_heston_matches_forward_price_identity():
    S0, r, q, T = 100, 0.04, 0.01, 2.0
    phi_neg_i = double_heston_char_func(-1j, S0, v1_0=0.03, v2_0=0.02,
                                         kappa1=1.5, theta1=0.05, xi1=0.4, rho1=-0.6,
                                         kappa2=0.4, theta2=0.03, xi2=0.5, rho2=-0.3,
                                         r=r, q=q, T=T)
    forward = S0 * np.exp((r - q) * T)
    assert abs(phi_neg_i.imag) < 1e-6
    assert abs(phi_neg_i.real - forward) / forward < 1e-6
    print(f"[OK] Double Heston phi(-i) = {phi_neg_i.real:.6f}, matches forward {forward:.6f}")


def test_double_heston_feller_conditions():
    both_ok, s1, s2 = double_heston_feller_conditions(2.0, 0.04, 0.2, 2.0, 0.04, 0.2)
    assert both_ok and s1 > 0 and s2 > 0
    both_ok, s1, s2 = double_heston_feller_conditions(2.0, 0.04, 0.2, 1.0, 0.04, 0.9)
    assert not both_ok and s1 > 0 and s2 < 0  # factor 1 fine, factor 2 violated
    print("[OK] Double Heston Feller conditions correctly evaluated per-factor")


def test_double_heston_reduces_to_single_heston_when_factor2_off():
    """
    CORE MODEL-CONSISTENCY TEST requested for the viva: if factor 2 is
    switched off (v2_0 -> 0 AND theta2 -> 0, so it never contributes any
    variance), Double Heston must collapse EXACTLY to single Heston using
    only factor 1's parameters -- this directly validates the factorized
    characteristic function's construction (f2 -> 1 when its factor
    carries zero variance for the whole path).
    """
    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    kappa1, theta1, xi1, rho1, v1_0 = 2.0, 0.05, 0.35, -0.6, 0.045

    # Factor 2 "off": tiny theta2 and v2_0 (exactly 0 would make D2's
    # xi2^2 denominator terms degenerate at u=0 in edge cases, so use a
    # very small but nonzero value -- same numerical-limit caveat as the
    # xi->0 test in test_heston.py).
    kappa2, theta2, xi2, rho2, v2_0 = 3.0, 1e-8, 0.3, -0.5, 1e-8

    dh_price = double_heston_price(S0, K, T, r, q, kappa1, theta1, xi1, rho1, v1_0,
                                    kappa2, theta2, xi2, rho2, v2_0, "call")
    single_price = heston_price(S0, K, T, r, q, kappa1, theta1, xi1, rho1, v1_0, "call")

    rel_err = abs(dh_price - single_price) / single_price
    assert rel_err < 1e-3, f"Double Heston (factor2 off)={dh_price}, single Heston={single_price}"
    print(f"[OK] Double Heston (factor 2 off) call = {dh_price:.6f} vs single Heston = {single_price:.6f} "
          f"(rel err {rel_err:.2e})")

    # Check across several strikes too.
    for K_test in [75, 90, 100, 110, 130]:
        dhp = double_heston_price(S0, K_test, T, r, q, kappa1, theta1, xi1, rho1, v1_0,
                                   kappa2, theta2, xi2, rho2, v2_0, "call")
        sp = heston_price(S0, K_test, T, r, q, kappa1, theta1, xi1, rho1, v1_0, "call")
        err = abs(dhp - sp) / max(sp, 1e-6)
        assert err < 5e-3, f"K={K_test}: DH={dhp}, single={sp}, err={err}"
    print("[OK] Double Heston -> single Heston collapse holds across ITM/ATM/OTM strikes")


def test_double_heston_cf_price_matches_monte_carlo():
    """Cross-validation: CF pricer vs MC simulator, with BOTH factors
    carrying meaningful (and Feller-violating) variance, to stress-test
    the two-factor full-truncation scheme."""
    S0, K, T, r, q = 100.0, 105.0, 1.0, 0.03, 0.0
    kappa1, theta1, xi1, rho1, v1_0 = 3.0, 0.04, 0.5, -0.6, 0.03  # fast factor
    kappa2, theta2, xi2, rho2, v2_0 = 0.5, 0.03, 0.4, -0.4, 0.02  # slow factor

    cf_price = double_heston_price(S0, K, T, r, q, kappa1, theta1, xi1, rho1, v1_0,
                                    kappa2, theta2, xi2, rho2, v2_0, "call")
    mc_price, mc_stderr = double_heston_mc_price(S0, K, T, r, q, kappa1, theta1, xi1, rho1, v1_0,
                                                  kappa2, theta2, xi2, rho2, v2_0, "call",
                                                  n_steps=500, n_paths=200_000, seed=42)

    diff = abs(cf_price - mc_price)
    assert diff < 4 * mc_stderr, f"CF price {cf_price:.4f} vs MC price {mc_price:.4f} +/- {mc_stderr:.4f}"
    print(f"[OK] CF price = {cf_price:.4f}, MC price = {mc_price:.4f} +/- {mc_stderr:.4f}")


def test_double_heston_mc_variance_paths_stay_nonnegative():
    _, S, v1, v2 = simulate_double_heston_paths(
        S0=100, v1_0=0.02, v2_0=0.01, T=2.0, n_steps=500, n_paths=5000,
        kappa1=0.5, theta1=0.08, xi1=1.2, rho1=-0.8,
        kappa2=0.3, theta2=0.05, xi2=1.0, rho2=-0.5,
        r=0.03, q=0.0, seed=1)
    assert np.all(v1 >= 0) and np.all(v2 >= 0)
    assert np.all(np.isfinite(S))
    print("[OK] both Double Heston variance factors stay non-negative under full truncation")


if __name__ == "__main__":
    test_double_heston_char_func_at_zero_is_one()
    test_double_heston_matches_forward_price_identity()
    test_double_heston_feller_conditions()
    test_double_heston_reduces_to_single_heston_when_factor2_off()
    test_double_heston_mc_variance_paths_stay_nonnegative()
    test_double_heston_cf_price_matches_monte_carlo()
    print("\nAll Double Heston tests passed.")
