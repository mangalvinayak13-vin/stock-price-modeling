"""
tests/test_heston.py
======================
Correctness checks for the Heston characteristic-function pricer and
Monte Carlo simulator. These are the tests that matter most for viva
credibility: they don't just check the code runs, they check it's RIGHT.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.heston import heston_char_func, heston_price, feller_condition
from models.heston_mc import simulate_heston_paths, heston_mc_price
from models.black_scholes import bs_price


def test_char_func_at_zero_is_one():
    """phi(0) = E[e^0] = 1 always, for any characteristic function -- a
    basic sanity identity any correct implementation must satisfy."""
    val = heston_char_func(0.0, S0=100, v0=0.04, kappa=2.0, theta=0.04, xi=0.3,
                            rho=-0.7, r=0.03, q=0.0, T=1.0)
    assert abs(val - 1.0) < 1e-10, f"phi(0) = {val}, expected 1"
    print(f"[OK] phi(0) = {val:.10f} (expected 1)")


def test_char_func_matches_forward_price_identity():
    """
    phi(-i) = E[e^{-i*(-i)*ln(S_T)}] = E[e^{-ln(S_T)... }] wait -- by
    definition phi(u) = E[e^{iu ln S_T}], so phi(-i) = E[e^{i*(-i)*ln S_T}]
    = E[e^{ln S_T}] = E[S_T] = S0*e^{(r-q)T} under the risk-neutral
    measure. This is the exact identity heston_price() relies on internally
    to normalize P1 without a second quad call -- verify it actually holds
    for the little-trap CF as implemented (this is also implicitly a
    continuity/branch-cut check, since a broken CF would violate it).
    """
    S0, v0, kappa, theta, xi, rho, r, q, T = 100, 0.04, 1.5, 0.05, 0.4, -0.6, 0.04, 0.01, 2.0
    phi_neg_i = heston_char_func(-1j, S0, v0, kappa, theta, xi, rho, r, q, T)
    forward = S0 * np.exp((r - q) * T)
    assert abs(phi_neg_i.imag) < 1e-6, f"phi(-i) should be real, got {phi_neg_i}"
    assert abs(phi_neg_i.real - forward) / forward < 1e-6, f"phi(-i)={phi_neg_i.real}, forward={forward}"
    print(f"[OK] phi(-i) = {phi_neg_i.real:.6f}, matches forward price {forward:.6f}")


def test_char_func_continuous_across_long_maturity():
    """
    THE classic 'little Heston trap' failure mode: the ORIGINAL 1993 CF
    formula can be discontinuous in T for long maturities / certain u,
    because a complex log term crosses a branch cut. Verify the
    little-trap CF we implemented varies SMOOTHLY as T increases across a
    long horizon with high vol-of-vol (the regime where the bug famously
    shows up) -- no sudden jumps.
    """
    S0, v0, kappa, theta, xi, rho, r, q = 100, 0.04, 0.5, 0.09, 0.9, -0.7, 0.03, 0.0
    u = 5.0  # a moderately large u, where the original formula's bug is most visible
    T_grid = np.linspace(0.05, 10.0, 500)
    vals = np.array([heston_char_func(u, S0, v0, kappa, theta, xi, rho, r, q, T) for T in T_grid])

    # A discontinuity would show up as an abnormally large jump between
    # consecutive T points relative to the typical step-to-step change.
    diffs = np.abs(np.diff(vals))
    median_diff = np.median(diffs)
    max_diff = np.max(diffs)
    # A genuine branch-cut jump is a discrete, large jump -- typically
    # orders of magnitude above the local (smooth) rate of change.
    assert max_diff < 50 * median_diff, \
        f"possible discontinuity detected: max step {max_diff} vs median step {median_diff}"
    print(f"[OK] characteristic function is smooth across T in [0.05, 10]: "
          f"max/median step ratio = {max_diff / median_diff:.2f}")


def test_feller_condition():
    satisfied, slack = feller_condition(kappa=2.0, theta=0.04, xi=0.2)
    assert satisfied and slack > 0
    satisfied, slack = feller_condition(kappa=1.0, theta=0.04, xi=0.9)
    assert not satisfied and slack < 0
    print("[OK] Feller condition check behaves correctly on satisfied/violated cases")


def test_heston_reduces_to_black_scholes_as_xi_to_zero():
    """
    CORE MODEL-CONSISTENCY TEST requested for the viva: as xi (vol-of-vol)
    -> 0, the variance process has no randomness left -- v(t) deterministically
    decays from v0 toward theta at rate kappa, but with essentially zero
    stochastic fluctuation. If additionally v0 = theta, variance is
    CONSTANT at theta for the whole horizon, and Heston must collapse
    exactly to Black-Scholes with sigma = sqrt(theta).

    We use a small but nonzero xi (not literally 0) because the
    characteristic function has an xi^2 in several denominators (division
    by zero at exactly xi=0) -- this is a standard limit-taking numerical
    caveat, not a modeling error: economically xi=0 is a degenerate
    special case (no stochastic vol at all, just literally Black-Scholes),
    and any implementation that evaluates the SDE symbolically would need
    the same care.
    """
    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    theta = 0.04  # v0 = theta = 0.04 => constant variance => sigma = sqrt(0.04) = 0.2
    kappa = 2.0
    rho = -0.5  # correlation is irrelevant when there's no vol-of-vol to correlate with
    xi = 1e-4   # "nearly zero" vol-of-vol

    heston_c = heston_price(S0, K, T, r, q, kappa, theta, xi, rho, v0=theta, option_type="call")
    bs_c = bs_price(S0, K, T, r, sigma=np.sqrt(theta), q=q, option_type="call")

    rel_err = abs(heston_c - bs_c) / bs_c
    assert rel_err < 1e-3, f"Heston(xi->0)={heston_c}, BS={bs_c}, rel err={rel_err}"
    print(f"[OK] Heston(xi=1e-4) call = {heston_c:.6f} vs Black-Scholes = {bs_c:.6f} "
          f"(rel err {rel_err:.2e})")

    # Also check across several strikes (ITM/ATM/OTM) since a bug could
    # plausibly only manifest away from ATM.
    for K_test in [70, 85, 100, 115, 140]:
        hc = heston_price(S0, K_test, T, r, q, kappa, theta, xi, rho, v0=theta, option_type="call")
        bc = bs_price(S0, K_test, T, r, np.sqrt(theta), q, "call")
        err = abs(hc - bc) / max(bc, 1e-6)
        assert err < 5e-3, f"K={K_test}: Heston={hc}, BS={bc}, err={err}"
    print("[OK] Heston->BS collapse holds across ITM/ATM/OTM strikes")


def test_heston_cf_price_insensitive_to_truncation_bound():
    """Sanity check on the u_max judgment call: price shouldn't meaningfully
    change if we push the integration truncation further out."""
    S0, K, T, r, q = 100, 110, 1.5, 0.03, 0.0
    kappa, theta, xi, rho, v0 = 2.0, 0.05, 0.3, -0.6, 0.04

    p1 = heston_price(S0, K, T, r, q, kappa, theta, xi, rho, v0, "call", u_max=200)
    p2 = heston_price(S0, K, T, r, q, kappa, theta, xi, rho, v0, "call", u_max=500)
    assert abs(p1 - p2) < 1e-4, f"price changed from {p1} to {p2} when increasing u_max"
    print(f"[OK] price insensitive to u_max: {p1:.6f} (u_max=200) vs {p2:.6f} (u_max=500)")


def test_heston_cf_price_matches_monte_carlo():
    """
    THE key cross-validation test requested: characteristic-function price
    and Monte Carlo price must agree within Monte Carlo sampling error,
    for parameters that VIOLATE the Feller condition (the regime that
    breaks naive Euler simulation) -- this both validates the CF pricer
    AND validates the full-truncation MC scheme handles the hard case.
    """
    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    kappa, theta, xi, rho, v0 = 1.0, 0.06, 0.5, -0.7, 0.05  # xi^2=0.25 > 2*kappa*theta=0.12: Feller VIOLATED

    satisfied, slack = feller_condition(kappa, theta, xi)
    assert not satisfied, "test should use Feller-violating params to stress the MC scheme"

    cf_price = heston_price(S0, K, T, r, q, kappa, theta, xi, rho, v0, "call")
    mc_price, mc_stderr = heston_mc_price(S0, K, T, r, q, kappa, theta, xi, rho, v0,
                                           "call", n_steps=500, n_paths=200_000, seed=42)

    diff = abs(cf_price - mc_price)
    # 4 standard errors is a generous but still meaningful bound (>99.99%
    # confidence under CLT if both estimators are unbiased and correct).
    assert diff < 4 * mc_stderr, \
        f"CF price {cf_price:.4f} vs MC price {mc_price:.4f} +/- {mc_stderr:.4f} (diff {diff:.4f})"
    print(f"[OK] CF price = {cf_price:.4f}, MC price = {mc_price:.4f} +/- {mc_stderr:.4f} "
          f"(Feller violated, full-truncation Euler handles it correctly)")


def test_mc_variance_paths_stay_nonnegative():
    """The whole point of full truncation: v must never go negative in the
    stored/used path, even with Feller badly violated."""
    _, S, v = simulate_heston_paths(S0=100, v0=0.02, T=2.0, n_steps=500, n_paths=5000,
                                     kappa=0.5, theta=0.08, xi=1.2, rho=-0.8, r=0.03, q=0.0,
                                     seed=1)
    assert np.all(v >= 0), "variance went negative -- full truncation failed"
    assert np.all(np.isfinite(S)), "price paths contain non-finite values"
    print(f"[OK] all simulated variance paths stayed non-negative "
          f"(min={v.min():.6f}, this is a badly Feller-violating case: 2*kappa*theta={2*0.5*0.08}, xi^2={1.2**2})")


if __name__ == "__main__":
    test_char_func_at_zero_is_one()
    test_char_func_matches_forward_price_identity()
    test_char_func_continuous_across_long_maturity()
    test_feller_condition()
    test_heston_reduces_to_black_scholes_as_xi_to_zero()
    test_heston_cf_price_insensitive_to_truncation_bound()
    test_mc_variance_paths_stay_nonnegative()
    test_heston_cf_price_matches_monte_carlo()
    print("\nAll Heston tests passed.")
