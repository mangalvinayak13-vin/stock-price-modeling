"""
tests/test_vrp_backtest.py
=============================
Correctness/plumbing checks for trading/vrp_backtest.py, run on
SYNTHETIC GBM price + a hand-built VIX-like series (not a live network
fetch, so these run fast and deterministically in CI/offline) -- checking
the walk-forward mechanics, sign conventions, and summary-statistic
plumbing are right, not making any claim about a real edge (that's what
report/generate_figures.py's real-SPY run, and the README's reported
numbers, are for).
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.vrp_backtest import run_vrp_backtest, buy_and_hold_summary, _trailing_realized_var


def _synthetic_data(n_years=8, sigma_true=0.20, mu_true=0.06, seed=7):
    """8 years of synthetic GBM 'SPY' prices, plus a synthetic 'VIX' that
    oscillates above and below the true realized vol (so the backtest
    exercises BOTH the trade and skip branches, not just one)."""
    n_days = n_years * 260
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)

    dt = 1 / 252
    log_rets = rng.normal((mu_true - 0.5 * sigma_true**2) * dt, sigma_true * np.sqrt(dt), n_days)
    log_rets[0] = 0.0
    prices = pd.Series(100.0 * np.exp(np.cumsum(log_rets)), index=dates, name="Close")

    t = np.arange(n_days)
    vix_frac = sigma_true + 0.06 * np.sin(t / 45.0)  # oscillates the "market vol" around the true vol
    vix = pd.Series(np.maximum(vix_frac, 0.05) * 100.0, index=dates, name="VIX")

    return prices, vix


def test_trailing_realized_var_matches_manual_computation():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 0.01, 100)
    window = 21
    end_idx = 60
    result = _trailing_realized_var(values, end_idx, window, trading_days_per_year=252)
    manual = np.var(values[end_idx - window:end_idx], ddof=1) * 252
    assert abs(result - manual) < 1e-12
    print(f"[OK] _trailing_realized_var matches manual np.var computation: {result:.6f}")


def test_vrp_backtest_runs_end_to_end_and_produces_valid_summary():
    prices, vix = _synthetic_data()
    df = run_vrp_backtest(prices, vix, r=0.03, calib_years=5, holding_days=21,
                           transaction_cost_vol=0.01, vrp_threshold=0.0,
                           heston_n_multistarts=2, seed=1)

    assert len(df) > 0
    summary = df.attrs["summary"]
    for key in ["n_periods", "cagr", "annualized_vol", "sharpe", "max_drawdown",
                "win_rate", "active_fraction", "equity_curve"]:
        assert key in summary
    assert len(summary["equity_curve"]) == len(df)
    assert 0.0 <= summary["active_fraction"] <= 1.0
    # No-lookahead sanity: every entry date must precede its own exit date.
    assert (df["exit_date"] > df["entry_date"]).all()
    print(f"[OK] VRP backtest ran end-to-end: {len(df)} months, "
          f"active_fraction={summary['active_fraction']:.2f}, CAGR={summary['cagr']:.2%}")


def test_vrp_threshold_minus_999_forces_always_trade():
    """vrp_threshold=-999 should trade in literally every month (used as
    the 'always sell' baseline in the report)."""
    prices, vix = _synthetic_data()
    df = run_vrp_backtest(prices, vix, calib_years=5, holding_days=21,
                           vrp_threshold=-999, heston_n_multistarts=2, seed=1)
    assert df["traded"].all()
    assert df["pnl"].notna().all()
    print(f"[OK] vrp_threshold=-999 trades all {len(df)} months (always-sell baseline)")


def test_short_straddle_pnl_sign_convention():
    """When traded, pnl = premium_received - |move|, so pnl should be
    positive whenever the realized move is smaller than the premium
    collected, and negative when the move blows through it -- check both
    occur across a run with plenty of months, i.e. the P&L isn't
    accidentally always positive or always negative (a common sign-flip bug)."""
    prices, vix = _synthetic_data(n_years=10, seed=3)
    df = run_vrp_backtest(prices, vix, calib_years=5, holding_days=21,
                           vrp_threshold=-999, heston_n_multistarts=2, seed=2)
    active = df[df["traded"]]
    assert len(active) > 5
    recomputed = active["premium"] - active["payoff"]
    assert np.allclose(recomputed.values, active["pnl"].values)
    assert (active["pnl"] > 0).any() and (active["pnl"] < 0).any()
    print(f"[OK] Short-straddle P&L sign convention correct; "
          f"{(active['pnl']>0).sum()}/{len(active)} months profitable")


def test_buy_and_hold_summary_recovers_approximate_true_drift():
    """CAGR of a buy-and-hold on synthetic GBM should be in the right
    ballpark of the TRUE simulated drift mu_true=0.06 (won't match
    exactly -- one 8-year path has real sampling noise -- but should be
    within a wide, sanity-check tolerance, not wildly off)."""
    prices, _ = _synthetic_data(n_years=8, mu_true=0.06, seed=11)
    summary = buy_and_hold_summary(prices)
    assert -0.05 < summary["cagr"] < 0.20
    assert summary["max_drawdown"] <= 0.0
    print(f"[OK] Buy-and-hold on synthetic GBM: CAGR={summary['cagr']:.2%} "
          f"(true drift 6%), max_drawdown={summary['max_drawdown']:.2%}")


def test_not_enough_history_raises_clear_error():
    prices, vix = _synthetic_data(n_years=2)  # far short of the 6y (5 calib + 1 test) minimum
    try:
        run_vrp_backtest(prices, vix, calib_years=5, holding_days=21)
        assert False, "should have raised for insufficient history"
    except ValueError as e:
        assert "Not enough history" in str(e)
    print("[OK] Insufficient history raises a clear ValueError instead of silently misbehaving")


if __name__ == "__main__":
    test_trailing_realized_var_matches_manual_computation()
    test_vrp_backtest_runs_end_to_end_and_produces_valid_summary()
    test_vrp_threshold_minus_999_forces_always_trade()
    test_short_straddle_pnl_sign_convention()
    test_buy_and_hold_summary_recovers_approximate_true_drift()
    test_not_enough_history_raises_clear_error()
    print("\nAll VRP backtest tests passed.")
