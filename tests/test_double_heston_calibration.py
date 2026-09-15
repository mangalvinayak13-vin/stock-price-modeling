"""
tests/test_double_heston_calibration.py
==========================================
Parameter-recovery tests for Double Heston's return-based GMM calibration,
mirroring tests/test_heston_calibration.py's methodology: simulate KNOWN
parameters, calibrate as if it were real data, check recovery is
reasonable both in parameter space and (more importantly) in terms of
option prices computed from the recovered parameters.
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.double_heston_mc import simulate_double_heston_paths
from models.double_heston import double_heston_price
from calibration.double_heston_calibration import calibrate_double_heston_gmm


def test_double_heston_gmm_recovers_two_factor_structure():
    """
    Simulate a genuine two-factor process (kappa1 >> kappa2, i.e. one
    fast-reverting and one slow-reverting factor, clearly separated) and
    check the calibration (a) preserves the fast/slow ordering, (b)
    recovers a sensible total long-run variance, and (c) prices a
    representative option reasonably close to the true-parameter price.
    """
    S0, r, q = 100.0, 0.03, 0.0
    kappa1, theta1, xi1, rho1, v1_0 = 8.0, 0.03, 0.5, -0.6, 0.025   # fast factor
    kappa2, theta2, xi2, rho2, v2_0 = 0.8, 0.02, 0.3, -0.4, 0.015   # slow factor

    n_years = 15
    t, S, v1, v2 = simulate_double_heston_paths(
        S0, v1_0, v2_0, T=n_years, n_steps=252 * n_years, n_paths=1,
        kappa1=kappa1, theta1=theta1, xi1=xi1, rho1=rho1,
        kappa2=kappa2, theta2=theta2, xi2=xi2, rho2=rho2,
        r=r, q=q, seed=55, antithetic=False)

    log_returns = pd.Series(np.diff(np.log(S[0])))
    achieved_theta_total = (v1[0] + v2[0]).mean()  # accounts for MC discretization bias, as in single-Heston test

    result = calibrate_double_heston_gmm(log_returns, n_multistarts=10, seed=3)
    theta_total_hat = result["theta1"] + result["theta2"]

    print(f"True (nominal) theta1+theta2 = {theta1 + theta2:.4f}, achieved (simulated path mean) = {achieved_theta_total:.4f}")
    print(f"Recovered: kappa1={result['kappa1']:.3f}, theta1={result['theta1']:.4f}, xi1={result['xi1']:.3f}")
    print(f"           kappa2={result['kappa2']:.3f}, theta2={result['theta2']:.4f}, xi2={result['xi2']:.3f}")
    print(f"           theta1+theta2={theta_total_hat:.4f}, rho={result['rho1']:.3f}")

    assert result["kappa1"] >= result["kappa2"], \
        "factor ordering violated: kappa1 should be >= kappa2 by construction of the calibration"

    assert abs(theta_total_hat - achieved_theta_total) / achieved_theta_total < 0.35, \
        f"total variance level recovery off: got {theta_total_hat}, achieved {achieved_theta_total}"

    # Economic check: price a representative option with true vs recovered params.
    K, T = 105.0, 0.75
    true_price = double_heston_price(S0, K, T, r, q, kappa1, theta1, xi1, rho1, v1_0,
                                      kappa2, theta2, xi2, rho2, v2_0, "call")
    recovered_price = double_heston_price(S0, K, T, r, q,
                                           result["kappa1"], result["theta1"], result["xi1"],
                                           result["rho1"], result["v1_0"],
                                           result["kappa2"], result["theta2"], result["xi2"],
                                           result["rho2"], result["v2_0"], "call")
    rel_err = abs(recovered_price - true_price) / true_price
    print(f"True-param price: {true_price:.4f}, recovered-param price: {recovered_price:.4f} (rel err {rel_err:.2%})")
    assert rel_err < 0.6, f"recovered-parameter price too far off: {rel_err:.2%}"
    print("[OK] Double Heston GMM calibration recovers sensible two-factor structure")


def test_double_heston_calibration_degenerates_gracefully_for_single_factor_data():
    """
    THE requested consistency check applied to CALIBRATION (not just
    pricing, which test_double_heston.py already covers): if the TRUE
    data-generating process only has ONE active factor (factor 2 truly
    off), Double Heston's calibration should still produce a total
    variance level and option price close to the single-factor truth --
    i.e. it shouldn't need a "real" second factor to fit single-factor
    data well, it should just let factor 2 end up small/negligible.
    """
    S0, r, q = 100.0, 0.03, 0.0
    kappa1, theta1, xi1, rho1, v1_0 = 3.0, 0.045, 0.4, -0.55, 0.04
    # True factor 2 is OFF.
    kappa2, theta2, xi2, rho2, v2_0 = 2.0, 1e-8, 0.3, -0.4, 1e-8

    n_years = 15
    t, S, v1, v2 = simulate_double_heston_paths(
        S0, v1_0, v2_0, T=n_years, n_steps=252 * n_years, n_paths=1,
        kappa1=kappa1, theta1=theta1, xi1=xi1, rho1=rho1,
        kappa2=kappa2, theta2=theta2, xi2=xi2, rho2=rho2,
        r=r, q=q, seed=21, antithetic=False)
    log_returns = pd.Series(np.diff(np.log(S[0])))

    result = calibrate_double_heston_gmm(log_returns, n_multistarts=10, seed=4)

    K, T = 100.0, 0.5
    # "True" price here uses the single active factor only (factor 2 contributes ~0).
    from models.heston import heston_price
    true_price = heston_price(S0, K, T, r, q, kappa1, theta1, xi1, rho1, v1_0, "call")
    recovered_price = double_heston_price(S0, K, T, r, q,
                                           result["kappa1"], result["theta1"], result["xi1"],
                                           result["rho1"], result["v1_0"],
                                           result["kappa2"], result["theta2"], result["xi2"],
                                           result["rho2"], result["v2_0"], "call")
    rel_err = abs(recovered_price - true_price) / true_price
    print(f"Single-factor-truth price: {true_price:.4f}, Double Heston recovered price: {recovered_price:.4f} "
          f"(rel err {rel_err:.2%})")
    assert rel_err < 0.6, f"Double Heston failed to degenerate gracefully on single-factor data: {rel_err:.2%}"
    print("[OK] Double Heston calibration degenerates gracefully when the true process has only one active factor")


if __name__ == "__main__":
    test_double_heston_gmm_recovers_two_factor_structure()
    test_double_heston_calibration_degenerates_gracefully_for_single_factor_data()
    print("\nAll Double Heston calibration tests passed.")
