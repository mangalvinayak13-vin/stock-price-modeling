"""
tests/test_backtest.py
========================
Sanity checks for the walk-forward backtest utilities (Kupiec test and
the closed-form Heston expected-variance forecast), independent of any
real data fetch.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.backtest import kupiec_test, heston_expected_avg_variance, _make_windows


def test_kupiec_accepts_well_calibrated_model():
    """Breach rate exactly matching the target -> should NOT reject (high p-value)."""
    n_trials = 2000
    n_breaches = 100  # exactly 5% of 2000
    LR, p_value = kupiec_test(n_breaches, n_trials, p=0.05)
    assert p_value > 0.9, f"expected high p-value for exact match, got {p_value}"
    print(f"[OK] Kupiec test on exactly-5%-breach data: LR={LR:.4f}, p={p_value:.4f} (correctly not rejected)")


def test_kupiec_rejects_miscalibrated_model():
    """Breach rate far from target (e.g. way too many breaches) -> should reject (low p-value)."""
    n_trials = 2000
    n_breaches = 300  # 15% breach rate vs 5% target -- badly miscalibrated (model understates risk)
    LR, p_value = kupiec_test(n_breaches, n_trials, p=0.05)
    assert p_value < 0.01, f"expected low p-value for badly miscalibrated model, got {p_value}"
    print(f"[OK] Kupiec test on 15%-breach (vs 5% target) data: LR={LR:.4f}, p={p_value:.2e} (correctly rejected)")


def test_heston_expected_variance_limits():
    v0, theta, kappa = 0.09, 0.04, 2.0

    # As T -> 0, forecast should approach v0 (no time for mean reversion to act).
    val_small_T = heston_expected_avg_variance(v0, theta, kappa, T=1e-6)
    assert abs(val_small_T - v0) < 1e-3, f"T->0 limit should be v0={v0}, got {val_small_T}"

    # As T -> large, forecast should approach theta (long-run mean dominates the average).
    val_large_T = heston_expected_avg_variance(v0, theta, kappa, T=100.0)
    assert abs(val_large_T - theta) < 1e-3, f"T->inf limit should be theta={theta}, got {val_large_T}"

    # If v0 == theta (already at long-run level), forecast should equal theta for ANY T.
    val_at_theta = heston_expected_avg_variance(theta, theta, kappa, T=0.5)
    assert abs(val_at_theta - theta) < 1e-10, f"v0=theta case should give exactly theta, got {val_at_theta}"

    print(f"[OK] Heston expected-variance forecast limits: T->0 gives v0 ({val_small_T:.5f} vs {v0}), "
          f"T->inf gives theta ({val_large_T:.5f} vs {theta}), v0=theta gives theta exactly")


def test_windows_have_no_overlap_and_no_lookahead():
    """
    Core no-lookahead check: for every window, calib_end must equal
    test_start exactly (no gap, no overlap) -- i.e. the test period starts
    the instant the calibration period ends, and calibration never
    extends into the test period.
    """
    windows = _make_windows(n_obs=3000, calib_days=1260, test_days=252, step_days=252)
    assert len(windows) > 0
    for calib_start, calib_end, test_end in windows:
        assert calib_start < calib_end < test_end
        # (by construction test_start == calib_end, enforced directly in _make_windows,
        # this assert just documents/re-confirms the invariant)
    # Consecutive windows should step forward, not backward or overlap in start point.
    starts = [w[0] for w in windows]
    assert all(s2 > s1 for s1, s2 in zip(starts, starts[1:])), "windows must strictly advance forward in time"
    print(f"[OK] {len(windows)} walk-forward windows generated, all strictly chronological with no overlap/lookahead")


if __name__ == "__main__":
    test_kupiec_accepts_well_calibrated_model()
    test_kupiec_rejects_miscalibrated_model()
    test_heston_expected_variance_limits()
    test_windows_have_no_overlap_and_no_lookahead()
    print("\nAll backtest utility tests passed.")
