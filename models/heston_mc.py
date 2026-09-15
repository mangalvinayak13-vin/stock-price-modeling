"""
models/heston_mc.py
=====================
Monte Carlo simulator for the Heston SDE system, used to cross-validate
the characteristic-function pricer in models/heston.py (they must agree
closely -- if they don't, one of the two implementations has a bug, and
in a viva that agreement IS your evidence the pricer is correct).

WHY NOT NAIVE EULER ON THE VARIANCE PROCESS:
The variance SDE is dv = kappa(theta-v)dt + xi*sqrt(v)dW2. A naive Euler
step v_{t+dt} = v_t + kappa(theta-v_t)dt + xi*sqrt(v_t)*dW2 can send v
NEGATIVE (dW2 is Gaussian, unbounded below), at which point sqrt(v) is
undefined (complex) on the next step -- the simulation literally breaks.
This isn't a rare edge case: whenever the Feller condition
2*kappa*theta >= xi^2 is violated (common for real calibrated equity
parameters, see models/heston.py), the true process spends non-negligible
time near zero, making this failure mode hit often, not just in
pathological corners.

SCHEME USED HERE: Full Truncation Euler (Lord, Koekkoek, van Dijk, 2010,
"A comparison of biased simulation schemes for stochastic volatility
models"). At each step, wherever v appears inside a sqrt() or as a drift
input, we use v+ = max(v, 0) instead of v itself, but we let the *stored*
value of v itself go negative before truncating it back to zero for
future access:
    v_next = v + kappa*(theta - v+)*dt + xi*sqrt(v+)*dW2
    v_next_used = max(v_next, 0)   <- what's actually carried forward and
                                       used as "v" in the next step

JUDGMENT CALL: Andersen's (2008) Quadratic-Exponential (QE) scheme is more
accurate (lower bias for large time steps) and is the "gold standard" in
industry, but is substantially more involved to implement correctly (it
switches between two different sampling schemes depending on a
psi-threshold test, with several derived constants). Full Truncation
Euler is simpler, still correctly handles the negative-variance problem
(unlike naive Euler), and its bias becomes small with a reasonably fine
time grid (e.g. daily steps, dt=1/252) -- verified empirically here by
checking MC-vs-characteristic-function price agreement converges as
n_steps increases (see tests/test_heston.py). Given the project's time
budget, Full Truncation Euler is the right complexity/accuracy tradeoff;
this tradeoff (and that QE would reduce discretization bias further were
finer accuracy needed) is worth stating explicitly in the viva.

We also use ANTITHETIC VARIATES (each Brownian draw is used for one path
and its negation for another) to reduce Monte Carlo variance at no extra
random-number cost -- a standard variance-reduction technique.
"""

import numpy as np


def simulate_heston_paths(S0, v0, T, n_steps, n_paths, kappa, theta, xi, rho, r, q=0.0,
                           seed=None, antithetic=True):
    """
    Simulate Heston (S, v) paths via Full Truncation Euler discretization.

    Parameters
    ----------
    S0, v0 : float, initial price and variance
    T : float, horizon in years
    n_steps : int, number of time steps
    n_paths : int, number of simulated paths (if antithetic=True, this is
        rounded up to an even number and half are antithetic pairs of the
        other half)
    kappa, theta, xi, rho : Heston variance-process parameters
    r, q : risk-free rate, dividend yield (drift of S under risk-neutral
        measure is (r-q))
    seed : RNG seed
    antithetic : bool, use antithetic variance reduction (recommended;
        set False only if you specifically need i.i.d. paths, e.g. for
        certain moment estimators that assume independence)

    Returns
    -------
    t : np.ndarray, shape (n_steps+1,)
    S : np.ndarray, shape (n_paths, n_steps+1)
    v : np.ndarray, shape (n_paths, n_steps+1)  -- the (truncated,
        non-negative) simulated variance paths
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

    S = np.zeros((n_sim, n_steps + 1))
    v = np.zeros((n_sim, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    if antithetic:
        S2 = np.zeros((n_sim, n_steps + 1))
        v2 = np.zeros((n_sim, n_steps + 1))
        S2[:, 0] = S0
        v2[:, 0] = v0

    for step in range(n_steps):
        # Correlated Gaussian increments: Z1 drives the price, Z2 drives
        # variance, with Corr(Z1,Z2) = rho built via Cholesky-style mixing:
        # Z2 = rho*Z1 + sqrt(1-rho^2)*Z_indep.
        Z1 = rng.standard_normal(n_sim)
        Z_indep = rng.standard_normal(n_sim)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z_indep

        v_pos = np.maximum(v[:, step], 0.0)  # v+ = max(v,0), the "full truncation"
        sqrt_v_pos = np.sqrt(v_pos)

        # Variance step (Full Truncation Euler): drift and diffusion use
        # v+, but the updated value itself is allowed to go negative
        # before being truncated on the NEXT iteration's v_pos computation
        # (this matches Lord-Koekkoek-van Dijk's exact prescription).
        v[:, step + 1] = v[:, step] + kappa * (theta - v_pos) * dt + xi * sqrt_v_pos * Z2 * sqrt_dt

        # Price step: exact-in-drift log-Euler using v+ as the
        # instantaneous variance for this step (standard practice paired
        # with full truncation -- using log S keeps S > 0 automatically,
        # unlike stepping S directly).
        S[:, step + 1] = S[:, step] * np.exp(
            (r - q - 0.5 * v_pos) * dt + sqrt_v_pos * Z1 * sqrt_dt
        )

        if antithetic:
            v2_pos = np.maximum(v2[:, step], 0.0)
            sqrt_v2_pos = np.sqrt(v2_pos)
            # Antithetic pair uses the NEGATED Gaussian draws.
            v2[:, step + 1] = v2[:, step] + kappa * (theta - v2_pos) * dt + xi * sqrt_v2_pos * (-Z2) * sqrt_dt
            S2[:, step + 1] = S2[:, step] * np.exp(
                (r - q - 0.5 * v2_pos) * dt + sqrt_v2_pos * (-Z1) * sqrt_dt
            )

    if antithetic:
        S = np.concatenate([S, S2], axis=0)[:n_paths]
        v = np.concatenate([v, v2], axis=0)[:n_paths]
        # Truncate/pad the stored variance to non-negative for reporting
        # consistency with what was actually used in the last step's drift
        # (purely cosmetic -- doesn't affect the simulated S paths, which
        # already used max(v,0) at each step as required).
        v = np.maximum(v, 0.0)

    return t, S, v


def heston_mc_price(S0, K, T, r, q, kappa, theta, xi, rho, v0, option_type="call",
                     n_steps=252, n_paths=100_000, seed=None):
    """
    Price a European option via Monte Carlo simulation of the Heston SDE
    (discounted average terminal payoff). Used purely as a cross-check
    against the characteristic-function pricer in models/heston.py --
    NOT used for calibration (far too slow to call inside an optimizer's
    inner loop; the CF pricer is used there instead).
    """
    _, S, _ = simulate_heston_paths(S0, v0, T, n_steps, n_paths, kappa, theta, xi, rho, r, q, seed=seed)
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
