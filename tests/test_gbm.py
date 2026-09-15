"""
tests/test_gbm.py
===================
Sanity checks for the GBM simulator and parameter estimator.

Core correctness check: under the exact solution
    S(t) = S0 * exp((mu - sigma^2/2) t + sigma W(t))
log S(T) - log S(0) ~ Normal( (mu - sigma^2/2) T, sigma^2 T ).
We simulate many paths and check the sample mean/variance of log-returns
at T match this analytical distribution -- this validates the simulator
is implemented correctly (not just "runs without crashing").

We also check that estimate_gbm_params recovers the true (mu, sigma) used
to generate synthetic data, within Monte Carlo sampling error.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gbm import simulate_gbm_paths, estimate_gbm_params


def test_gbm_terminal_distribution_matches_theory():
    S0, mu, sigma, T = 100.0, 0.08, 0.25, 1.0
    n_steps, n_paths = 252, 200_000

    t, S = simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=42)

    log_ret_T = np.log(S[:, -1] / S0)
    theo_mean = (mu - 0.5 * sigma**2) * T
    theo_std = sigma * np.sqrt(T)

    sample_mean = log_ret_T.mean()
    sample_std = log_ret_T.std(ddof=1)

    # With 200,000 paths, Monte Carlo std error of the mean is
    # sigma/sqrt(N) ~ 0.25/sqrt(200000) ~ 0.00056, so a tolerance of 0.01
    # is generous (>15 standard errors) while still being a real check.
    assert abs(sample_mean - theo_mean) < 0.01, f"mean {sample_mean} vs theory {theo_mean}"
    assert abs(sample_std - theo_std) < 0.01, f"std {sample_std} vs theory {theo_std}"
    print(f"[OK] terminal distribution: sample mean={sample_mean:.5f} (theory {theo_mean:.5f}), "
          f"sample std={sample_std:.5f} (theory {theo_std:.5f})")


def test_gbm_martingale_property_under_risk_neutral_drift():
    """
    If we simulate with mu = r (risk-neutral drift, no dividends), then
    E[S(T)] should equal S0 * exp(r*T) -- this is the defining property
    that makes GBM usable for risk-neutral option pricing later.
    """
    S0, r, sigma, T = 100.0, 0.05, 0.2, 1.0
    n_steps, n_paths = 100, 200_000
    t, S = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=7)

    sample_mean_ST = S[:, -1].mean()
    theo_mean_ST = S0 * np.exp(r * T)

    rel_err = abs(sample_mean_ST - theo_mean_ST) / theo_mean_ST
    assert rel_err < 0.01, f"E[S(T)] sample {sample_mean_ST} vs theory {theo_mean_ST}"
    print(f"[OK] martingale check: E[S(T)] sample={sample_mean_ST:.4f}, theory={theo_mean_ST:.4f}")


def test_estimate_gbm_params_recovers_truth():
    true_mu, true_sigma = 0.10, 0.30
    S0, T, n_steps = 100.0, 1.0, 252 * 10  # 10 years of daily data

    _, S = simulate_gbm_paths(S0, true_mu, true_sigma, T * 10, n_steps, n_paths=1, seed=123)
    log_returns = np.diff(np.log(S[0]))

    mu_hat, sigma_hat = estimate_gbm_params(log_returns, trading_days_per_year=252)

    assert abs(sigma_hat - true_sigma) < 0.02, f"sigma_hat={sigma_hat} vs true {true_sigma}"
    # mu estimates are noisy (need very long samples to pin down drift precisely) --
    # allow a wider tolerance, this is a known statistical fact (not a bug):
    # estimating drift from returns requires far more data than estimating vol.
    assert abs(mu_hat - true_mu) < 0.15, f"mu_hat={mu_hat} vs true {true_mu}"
    print(f"[OK] param recovery: mu_hat={mu_hat:.4f} (true {true_mu}), "
          f"sigma_hat={sigma_hat:.4f} (true {true_sigma})")


if __name__ == "__main__":
    test_gbm_terminal_distribution_matches_theory()
    test_gbm_martingale_property_under_risk_neutral_drift()
    test_estimate_gbm_params_recovers_truth()
    print("\nAll GBM tests passed.")
