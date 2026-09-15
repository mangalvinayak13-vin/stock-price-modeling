"""
models/double_heston_mc.py
=============================
Monte Carlo simulator for the Double Heston SDE system, used to
cross-validate the factorized characteristic-function pricer in
models/double_heston.py -- same role, and same Full Truncation Euler
scheme (see models/heston_mc.py's docstring for the full justification of
why naive Euler is unsafe for CIR-type variance processes), extended to
two INDEPENDENT variance factors instead of one.

Each factor gets its own pair of correlated Brownian increments
(Z_vi drives factor i's variance, Z_Si = rho_i*Z_vi + sqrt(1-rho_i^2)*Z_indep_i
drives factor i's contribution to the price), and the two factors'
increments are drawn independently of each other (4 independent standard
normals consumed per step, per path). The price process combines both
factors' diffusion contributions:

    d ln S = (r - q - 0.5*(v1+v2)) dt + sqrt(v1) dW1_S + sqrt(v2) dW2_S

which is exact given Var(d ln S) = (v1+v2)*dt = Var(sqrt(v1)*dW1_S) +
Var(sqrt(v2)*dW2_S) (the two contributions are independent so variances
add directly).
"""

import numpy as np


def simulate_double_heston_paths(S0, v1_0, v2_0, T, n_steps, n_paths,
                                  kappa1, theta1, xi1, rho1,
                                  kappa2, theta2, xi2, rho2,
                                  r, q=0.0, seed=None, antithetic=True):
    """
    Simulate Double Heston (S, v1, v2) paths via Full Truncation Euler.
    Same parameters/conventions as models/heston_mc.simulate_heston_paths,
    duplicated per factor.

    Returns
    -------
    t : np.ndarray, shape (n_steps+1,)
    S : np.ndarray, shape (n_paths, n_steps+1)
    v1, v2 : np.ndarray, shape (n_paths, n_steps+1), each factor's
        (truncated, non-negative) simulated variance path
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    t = np.linspace(0.0, T, n_steps + 1)

    if antithetic:
        half = (n_paths + 1) // 2
        n_sim = half
    else:
        n_sim = n_paths

    def _init_arrays():
        S_ = np.zeros((n_sim, n_steps + 1)); S_[:, 0] = S0
        v1_ = np.zeros((n_sim, n_steps + 1)); v1_[:, 0] = v1_0
        v2_ = np.zeros((n_sim, n_steps + 1)); v2_[:, 0] = v2_0
        return S_, v1_, v2_

    S, v1, v2 = _init_arrays()
    if antithetic:
        S2, v1b, v2b = _init_arrays()

    for step in range(n_steps):
        Zv1 = rng.standard_normal(n_sim)
        Zind1 = rng.standard_normal(n_sim)
        ZS1 = rho1 * Zv1 + np.sqrt(1 - rho1**2) * Zind1

        Zv2 = rng.standard_normal(n_sim)
        Zind2 = rng.standard_normal(n_sim)
        ZS2 = rho2 * Zv2 + np.sqrt(1 - rho2**2) * Zind2

        v1_pos = np.maximum(v1[:, step], 0.0)
        v2_pos = np.maximum(v2[:, step], 0.0)
        sqrt_v1_pos = np.sqrt(v1_pos)
        sqrt_v2_pos = np.sqrt(v2_pos)

        v1[:, step + 1] = v1[:, step] + kappa1 * (theta1 - v1_pos) * dt + xi1 * sqrt_v1_pos * Zv1 * sqrt_dt
        v2[:, step + 1] = v2[:, step] + kappa2 * (theta2 - v2_pos) * dt + xi2 * sqrt_v2_pos * Zv2 * sqrt_dt

        S[:, step + 1] = S[:, step] * np.exp(
            (r - q - 0.5 * (v1_pos + v2_pos)) * dt
            + sqrt_v1_pos * ZS1 * sqrt_dt + sqrt_v2_pos * ZS2 * sqrt_dt
        )

        if antithetic:
            v1b_pos = np.maximum(v1b[:, step], 0.0)
            v2b_pos = np.maximum(v2b[:, step], 0.0)
            sqrt_v1b_pos = np.sqrt(v1b_pos)
            sqrt_v2b_pos = np.sqrt(v2b_pos)

            v1b[:, step + 1] = v1b[:, step] + kappa1 * (theta1 - v1b_pos) * dt + xi1 * sqrt_v1b_pos * (-Zv1) * sqrt_dt
            v2b[:, step + 1] = v2b[:, step] + kappa2 * (theta2 - v2b_pos) * dt + xi2 * sqrt_v2b_pos * (-Zv2) * sqrt_dt
            S2[:, step + 1] = S2[:, step] * np.exp(
                (r - q - 0.5 * (v1b_pos + v2b_pos)) * dt
                + sqrt_v1b_pos * (-ZS1) * sqrt_dt + sqrt_v2b_pos * (-ZS2) * sqrt_dt
            )

    if antithetic:
        S = np.concatenate([S, S2], axis=0)[:n_paths]
        v1 = np.maximum(np.concatenate([v1, v1b], axis=0)[:n_paths], 0.0)
        v2 = np.maximum(np.concatenate([v2, v2b], axis=0)[:n_paths], 0.0)

    return t, S, v1, v2


def double_heston_mc_price(S0, K, T, r, q, kappa1, theta1, xi1, rho1, v1_0,
                            kappa2, theta2, xi2, rho2, v2_0, option_type="call",
                            n_steps=252, n_paths=100_000, seed=None):
    """
    Price a European option via Monte Carlo simulation of the Double
    Heston SDE, used purely as a cross-check against the characteristic-
    function pricer in models/double_heston.py (not used for calibration).
    """
    _, S, _, _ = simulate_double_heston_paths(S0, v1_0, v2_0, T, n_steps, n_paths,
                                               kappa1, theta1, xi1, rho1,
                                               kappa2, theta2, xi2, rho2, r, q, seed=seed)
    S_T = S[:, -1]

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_price = np.exp(-r * T) * payoff.mean()
    std_error = np.exp(-r * T) * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return discounted_price, std_error
